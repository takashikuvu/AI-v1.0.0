"""Summarise harvested corpora: counts, star distribution, and keyword evidence per architectural component."""
import json, re, collections
P=json.load(open("papers.json")); R=json.load(open("repos.json"))
out=[]
out.append(f"papers harvested (unique arXiv ids, title+abstract): {len(P)}")
out.append(f"repos harvested (unique full_name, metadata): {len(R)}")
b=collections.Counter()
for r in R:
    s=r["stargazers_count"]; b[">=10k" if s>=10000 else "2k-10k" if s>=2000 else "500-2k" if s>=500 else "100-500" if s>=100 else "20-100" if s>=20 else "5-20" if s>=5 else "1-4" if s>=1 else "0"]+=1
out.append("repo star buckets: "+json.dumps(dict(b)))
yrs=collections.Counter(p["published"][:4] for p in P); out.append("paper years: "+json.dumps(dict(sorted(yrs.items()))))
K={"gated delta / delta rule":r"delta rule|deltanet|delta net","mamba/SSM":r"mamba|state[- ]space","linear attention":r"linear attention","hybrid attn+linear":r"hybrid","MoE":r"mixture[- ]of[- ]experts|\bmoe\b","shared/fine-grained experts":r"fine-grained expert|shared expert","aux-loss-free balancing":r"auxiliary[- ]loss[- ]free|loss-free","MLA / KV compression":r"latent attention|kv cache compress|kv-cache compress","sparse attention":r"sparse attention","multi-token prediction":r"multi-token prediction","looped/recurrent depth":r"looped|recurrent[- ]depth|recursion","test-time training/memory":r"test-time training|titans|neural memory","muon":r"muon","RLVR/GRPO":r"grpo|verifiable reward|rlvr","distillation":r"distill","synthetic data":r"synthetic","FP8/low-precision":r"fp8|low[- ]precision|bitnet|1-bit|ternary","speculative decoding":r"speculative","long CoT / test-time compute":r"test-time compute|long chain|chain-of-thought","diffusion LM":r"diffusion language|diffusion lm|masked diffusion"}
txt=[(p["title"]+" "+p["abstract"]).lower() for p in P]
out.append("keyword -> #papers mentioning (title/abstract):")
for k,rx in K.items(): out.append(f"  {k}: {sum(1 for t in txt if re.search(rx,t))}")
rtxt=[((r["description"] or "")+" "+" ".join(r.get("topics") or [])+" "+r["full_name"]).lower() for r in R]
out.append("keyword -> #repos mentioning (name/desc/topics):")
for k,rx in K.items(): out.append(f"  {k}: {sum(1 for t in rtxt if re.search(rx,t))}")
top=sorted(R,key=lambda r:-r["stargazers_count"])[:25]
out.append("top repos: "+", ".join(f"{r['full_name']}({r['stargazers_count']})" for r in top))
open("SURVEY_STATS.txt","w").write("\n".join(out)); print("\n".join(out))
