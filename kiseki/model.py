"""Kiseki: hybrid Gated-DeltaNet / GQA-attention backbone with fine-grained shared-expert MoE,
aux-loss-free balancing and multi-token-prediction. Pure PyTorch reference implementation."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import KisekiConfig

class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.eps = eps; s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * s.w

def rope_cache(T, dim, theta, device, offset=0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(offset, offset + T, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin, rot):
    # x: B,H,T,D ; rotate first `rot` dims (partial RoPE)
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., ::2], xr[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    o = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
    return torch.cat([o, xp], -1)

class GQAttention(nn.Module):
    def __init__(s, c: KisekiConfig):
        super().__init__(); s.c = c
        s.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        s.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        s.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        s.o = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        s.gate = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)   # output gating (Qwen3-Next / gated attn)
        s.qn, s.kn = RMSNorm(c.head_dim), RMSNorm(c.head_dim)
        s.rot = int(c.head_dim * c.rope_frac) // 2 * 2
    def forward(s, x, cache=None):
        B, T, _ = x.shape; c = s.c
        q = s.q(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = s.k(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = s.v(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = s.qn(q), s.kn(k)
        off = 0 if cache is None else cache.get("len", 0)
        cos, sin = rope_cache(T, s.rot, c.rope_theta, x.device, off)
        q, k = apply_rope(q, cos, sin, s.rot), apply_rope(k, cos, sin, s.rot)
        if cache is not None:
            if "k" in cache: k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache.update(k=k, v=v, len=k.shape[2])
        rep = c.n_heads // c.n_kv_heads
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        causal = cache is None or T > 1 and off == 0
        o = F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)
        o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(s.gate(x))
        return s.o(o)

def l2n(x, eps=1e-6): return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)

def delta_rule_recurrent(q, k, v, g, beta, S=None):
    """Reference (sequential) gated delta rule. q,k: B,H,T,Dk  v: B,H,T,Dv  g: log-decay B,H,T  beta: B,H,T
    S_t = exp(g_t) S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T ;  o_t = S_t q_t"""
    B, H, T, Dk = q.shape; Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv) if S is None else S
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        S = S * g[:, :, t].exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, (vt - pred) * beta[:, :, t, None])
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, 2), S

def delta_rule_chunk(q, k, v, g, beta, chunk=32, S=None):
    """Chunkwise-parallel gated delta rule (WY representation), numerically equal to the recurrent form."""
    B, H, T, Dk = q.shape; Dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:
        q, k, v = [F.pad(a, (0, 0, 0, pad)) for a in (q, k, v)]
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    L = q.shape[2]; N = L // chunk
    r = lambda a: a.reshape(B, H, N, chunk, *a.shape[3:])
    q, k, v, g, beta = map(r, (q, k, v, g, beta))
    g = g.cumsum(-1)
    vb, kb = v * beta[..., None], k * beta[..., None]
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    strict = torch.triu(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool), 0)
    dm = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    A = -((kb @ k.transpose(-1, -2)) * dm).masked_fill(strict, 0)
    for i in range(1, chunk):   # forward substitution of (I + L)^-1
        A[..., i, :i] = A[..., i, :i] + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = A + eye
    value = A @ vb
    kcd = A @ (kb * g.exp()[..., None])
    S = q.new_zeros(B, H, Dk, Dv) if S is None else S
    out = torch.zeros_like(v)
    upper = torch.triu(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool), 1)
    for i in range(N):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        attn = (qi @ ki.transpose(-1, -2) * dm[:, :, i]).masked_fill(upper, 0)
        vnew = value[:, :, i] - kcd[:, :, i] @ S
        out[:, :, i] = (qi * gi.exp()[..., None]) @ S + attn @ vnew
        S = S * gi[..., -1, None, None].exp() + (ki * (gi[..., -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ vnew
    return out.reshape(B, H, L, Dv)[:, :, :T], S

class GatedDeltaNet(nn.Module):
    def __init__(s, c: KisekiConfig):
        super().__init__(); s.c = c; H = c.dn_heads
        s.qkv = nn.Linear(c.d_model, H * (2 * c.dn_head_k + c.dn_head_v), bias=False)
        s.conv = nn.Conv1d(H * (2 * c.dn_head_k + c.dn_head_v), H * (2 * c.dn_head_k + c.dn_head_v), c.dn_conv,
                           groups=H * (2 * c.dn_head_k + c.dn_head_v), padding=0, bias=False)
        s.ab = nn.Linear(c.d_model, 2 * H, bias=False)
        s.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        s.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.empty(H).uniform_(1e-3, 1e-1))))
        s.gate = nn.Linear(c.d_model, H * c.dn_head_v, bias=False)
        s.on = RMSNorm(c.dn_head_v)
        s.o = nn.Linear(H * c.dn_head_v, c.d_model, bias=False)
    def forward(s, x, cache=None):
        B, T, _ = x.shape; c = s.c; H = c.dn_heads
        raw = s.qkv(x).transpose(1, 2)                      # B,C,T
        prev = cache["conv"] if (cache is not None and "conv" in cache) else raw.new_zeros(B, raw.shape[1], c.dn_conv - 1)
        full = torch.cat([prev, raw], 2)
        if cache is not None: cache["conv"] = full[..., -(c.dn_conv - 1):]
        z = F.silu(s.conv(full))                              # causal: padding=0 on left-padded input
        z = z.transpose(1, 2)
        q, k, v = torch.split(z, [H * c.dn_head_k, H * c.dn_head_k, H * c.dn_head_v], -1)
        sh = lambda a, d: a.view(B, T, H, d).transpose(1, 2)
        q, k, v = l2n(sh(q, c.dn_head_k)) * c.dn_head_k ** -0.5, l2n(sh(k, c.dn_head_k)), sh(v, c.dn_head_v)
        a, b = s.ab(x).float().chunk(2, -1)
        g = (-s.A_log.exp() * F.softplus(a + s.dt_bias)).transpose(1, 2)   # log decay <= 0
        beta = torch.sigmoid(b).transpose(1, 2)
        S0 = cache.get("S") if cache is not None else None
        fn = delta_rule_recurrent if (T == 1) else delta_rule_chunk
        o, S = fn(q.float(), k.float(), v.float(), g, beta, S=S0)
        if cache is not None: cache["S"] = S
        o = s.on(o.type_as(x).transpose(1, 2)) * torch.sigmoid(s.gate(x)).view(B, T, H, -1)
        return s.o(o.reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(s, d, f):
        super().__init__(); s.w1 = nn.Linear(d, f, bias=False); s.w3 = nn.Linear(d, f, bias=False); s.w2 = nn.Linear(f, d, bias=False)
    def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))

class MoE(nn.Module):
    """Fine-grained routed experts + always-on shared expert; sigmoid router, aux-loss-free bias balancing."""
    def __init__(s, c: KisekiConfig):
        super().__init__(); s.c = c
        s.router = nn.Linear(c.d_model, c.n_experts, bias=False)
        s.register_buffer("bias", torch.zeros(c.n_experts))
        s.register_buffer("load", torch.zeros(c.n_experts))
        s.experts = nn.ModuleList(SwiGLU(c.d_model, c.expert_ff) for _ in range(c.n_experts))
        s.shared = SwiGLU(c.d_model, c.expert_ff * c.n_shared)
    def forward(s, x):
        B, T, D = x.shape; xf = x.reshape(-1, D); c = s.c
        scores = torch.sigmoid(s.router(xf).float())
        _, idx = (scores + s.bias).topk(c.top_k, -1)           # bias affects selection only
        w = scores.gather(-1, idx); w = w / (w.sum(-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(xf)
        for e in range(c.n_experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel(): out.index_add_(0, tok, (s.experts[e](xf[tok]) * w[tok, slot, None]).type_as(xf))
        if s.training:
            with torch.no_grad():
                cnt = torch.bincount(idx.flatten(), minlength=c.n_experts).float()
                s.load.copy_(cnt / cnt.sum())
        return (out + s.shared(xf)).view(B, T, D)
    @torch.no_grad()
    def update_bias(s):   # call once per optimizer step
        err = 1.0 / s.c.n_experts - s.load
        s.bias += s.c.bias_update_rate * torch.sign(err)

class Block(nn.Module):
    def __init__(s, c: KisekiConfig, i: int):
        super().__init__()
        s.is_attn = (i + 1) % c.attn_every == 0
        s.n1, s.n2 = RMSNorm(c.d_model, c.norm_eps), RMSNorm(c.d_model, c.norm_eps)
        s.mix = GQAttention(c) if s.is_attn else GatedDeltaNet(c)
        s.ffn = SwiGLU(c.d_model, c.dense_ff) if i < c.dense_layers else MoE(c)
    def forward(s, x, cache=None):
        x = x + s.mix(s.n1(x), cache)
        return x + s.ffn(s.n2(x))

class Kiseki(nn.Module):
    def __init__(s, c: KisekiConfig):
        super().__init__(); s.c = c
        s.emb = nn.Embedding(c.vocab_size, c.d_model)
        s.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layers))
        s.norm = RMSNorm(c.d_model, c.norm_eps)
        s.head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        # MTP: predict token t+1+d from [h_t ; emb(x_{t+d})] through one light transformer block
        s.mtp_proj = nn.ModuleList(nn.Linear(2 * c.d_model, c.d_model, bias=False) for _ in range(c.mtp_depth))
        s.mtp_blk = nn.ModuleList(Block(c, c.attn_every - 1) for _ in range(c.mtp_depth))
        s.apply(s._init)
        for n, p in s.named_parameters():   # depth-scaled init for residual outputs
            if n.endswith(("mix.o.weight", "w2.weight")): nn.init.normal_(p, 0, c.init_std / math.sqrt(2 * c.n_layers))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, s.c.init_std)
    def logits(s, h): return F.linear(h, s.emb.weight) if s.head is None else s.head(h)
    def forward(s, ids, targets=None, cache=None):
        x = s.emb(ids)
        for i, b in enumerate(s.blocks): x = b(x, None if cache is None else cache[i])
        h = s.norm(x); lg = s.logits(h)
        if targets is None: return lg
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), targets.view(-1), ignore_index=-100)
        mtp = 0.0
        for d in range(s.c.mtp_depth):          # token t+2+d predicted from h_t and emb(x_{t+1+d})
            sh = d + 1
            if ids.shape[1] <= sh + 1: break
            hh = s.mtp_proj[d](torch.cat([h[:, :-sh], s.emb(ids[:, sh:])], -1))
            hh = s.mtp_blk[d](hh)
            l2 = s.logits(s.norm(hh))[:, :-1]; tgt = targets[:, sh:][:, 1:] if False else targets[:, sh + 1:]
            l2 = l2[:, :tgt.shape[1]]
            mtp = mtp + F.cross_entropy(l2.float().reshape(-1, l2.shape[-1]), tgt.reshape(-1))
        return loss + s.c.mtp_weight * mtp, loss
    def new_cache(s): return [dict() for _ in s.blocks]
    @torch.no_grad()
    def generate(s, ids, n, temp=0.8, top_k=40):
        s.eval(); cache = s.new_cache(); lg = s(ids, cache=cache)[:, -1]
        for _ in range(n):
            lg = lg / temp
            if top_k: v, _ = lg.topk(top_k); lg[lg < v[:, -1:]] = -1e9
            nxt = torch.multinomial(F.softmax(lg, -1), 1); ids = torch.cat([ids, nxt], 1)
            lg = s(nxt, cache=cache)[:, -1]
        return ids

def count_params(c: KisekiConfig):
    with torch.device("meta"): m = Kiseki(c)
    total = sum(p.numel() for p in m.parameters())
    mtp = sum(p.numel() for n, p in m.named_parameters() if n.startswith("mtp"))
    inactive = 0
    for b in m.blocks:
        if isinstance(b.ffn, MoE):
            inactive += (c.n_experts - c.top_k) * 3 * c.d_model * c.expert_ff
    return dict(total=total, total_wo_mtp=total - mtp, active=total - mtp - inactive)
