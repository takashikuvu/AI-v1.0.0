"""Tiny-scale CPU training run of the Kiseki architecture on byte-level TinyStories (sanity / learning-signal check)."""
import torch, time, json, math, sys
from config import kiseki_tiny; from model import Kiseki, MoE
torch.manual_seed(1); torch.set_num_threads(2)
data = torch.frombuffer(bytearray(open("data/tinystories.txt","rb").read()), dtype=torch.uint8).long()
n = int(len(data)*0.98); tr, va = data[:n], data[n:]
c = kiseki_tiny(); m = Kiseki(c)
print("params", sum(p.numel() for p in m.parameters())/1e6, "M", flush=True)
steps, B, T = int(sys.argv[1]) if len(sys.argv)>1 else 400, 8, 128
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9,0.95), weight_decay=0.05)
def batch(d):
    ix = torch.randint(0, len(d)-T-1, (B,)); return torch.stack([d[i:i+T] for i in ix]), torch.stack([d[i+1:i+T+1] for i in ix])
def lr(s): return 3e-3*min(1,(s+1)/30)*(0.1+0.9*0.5*(1+math.cos(math.pi*s/steps)))
log=[]; t0=time.time()
for s in range(steps):
    for g in opt.param_groups: g["lr"]=lr(s)
    x,y = batch(tr); m.train(); tot,l = m(x,y); opt.zero_grad(); tot.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    for mod in m.modules():
        if isinstance(mod,MoE): mod.update_bias()
    if s%25==0 or s==steps-1:
        m.eval()
        with torch.no_grad(): vx,vy=batch(va); vl=m(vx,vy)[1].item()
        loads=[mod.load for mod in m.modules() if isinstance(mod,MoE)]
        imb=max((ld.max()*len(ld)).item() for ld in loads)
        log.append(dict(step=s,train=l.item(),val=vl,bpb=vl/math.log(2),max_load_ratio=imb,t=time.time()-t0)); print(log[-1],flush=True)
json.dump(log,open("train_log.json","w"),indent=1)
torch.save(m.state_dict(),"tiny.pt")
m.eval(); out=m.generate(torch.tensor([list(b"Once upon a time")]),200,temp=0.7)
txt=bytes(out[0].tolist()).decode("utf8","replace"); print("SAMPLE:",txt); open("sample.txt","w").write(txt)
