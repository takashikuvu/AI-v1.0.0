import torch, sys
from config import *; from model import *
torch.manual_seed(0)
# 1) chunk == recurrent
B,H,T,Dk,Dv=2,3,70,16,16
q=l2n(torch.randn(B,H,T,Dk));k=l2n(torch.randn(B,H,T,Dk));v=torch.randn(B,H,T,Dv)
g=-torch.rand(B,H,T)*0.5;beta=torch.rand(B,H,T)
o1,S1=delta_rule_recurrent(q,k,v,g,beta);o2,S2=delta_rule_chunk(q,k,v,g,beta,chunk=16)
print("delta chunk vs recurrent max err:",(o1-o2).abs().max().item(),(S1-S2).abs().max().item()); assert (o1-o2).abs().max()<1e-4
# 2) causality
m=Kiseki(kiseki_tiny()).eval(); ids=torch.randint(0,256,(1,48)); a=m(ids); ids2=ids.clone(); ids2[0,30:]=torch.randint(0,256,(18,)); b=m(ids2)
print("causality err (pos<30):",(a[:,:30]-b[:,:30]).abs().max().item()); assert (a[:,:30]-b[:,:30]).abs().max()<1e-4
# 3) cached incremental decode == full forward
cache=m.new_cache(); l=m(ids[:,:40],cache=cache); outs=[l]
for t in range(40,48): outs.append(m(ids[:,t:t+1],cache=cache))
inc=torch.cat(outs,1); print("cache decode err:",(inc-a).abs().max().item()); assert (inc-a).abs().max()<2e-3
# 4) param counts
for n,c in [("tiny",kiseki_tiny()),("9B",kiseki_9b())]: print(n,{k:f"{v/1e9:.3f}B" for k,v in count_params(c).items()})
print("ALL OK")
