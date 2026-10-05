from dataclasses import dataclass, field

@dataclass
class KisekiConfig:
    vocab_size: int = 128_000
    d_model: int = 2560
    n_layers: int = 32
    # --- layer schedule: 3 linear (Gated DeltaNet) : 1 global attention, repeating
    attn_every: int = 4
    # --- Gated DeltaNet
    dn_heads: int = 20
    dn_head_k: int = 128
    dn_head_v: int = 128
    dn_conv: int = 4
    # --- attention (GQA + QK-norm + partial RoPE, NoPE-free global layers)
    n_heads: int = 20
    n_kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rope_frac: float = 0.5
    # --- FFN: first `dense_layers` are dense SwiGLU, rest are fine-grained MoE w/ shared expert
    dense_layers: int = 2
    dense_ff: int = 6912
    n_experts: int = 64
    top_k: int = 6
    n_shared: int = 1
    expert_ff: int = 512
    # --- extras
    mtp_depth: int = 1          # multi-token-prediction heads
    mtp_weight: float = 0.3
    tie_embeddings: bool = True
    max_seq: int = 131_072
    norm_eps: float = 1e-6
    bias_update_rate: float = 1e-3   # aux-loss-free balancing
    init_std: float = 0.02

def kiseki_9b() -> KisekiConfig:
    return KisekiConfig()

def kiseki_tiny() -> KisekiConfig:
    return KisekiConfig(vocab_size=256, d_model=128, n_layers=8, attn_every=4, dn_heads=4, dn_head_k=32,
                        dn_head_v=32, n_heads=4, n_kv_heads=2, head_dim=32, dense_layers=1, dense_ff=256,
                        n_experts=8, top_k=2, n_shared=1, expert_ff=64, max_seq=1024)
