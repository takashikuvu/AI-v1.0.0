"""Harvest GitHub repos across star ranges (1 star .. 10k+) for LLM-architecture topics; fetch README excerpts."""
import json, time, urllib.request, urllib.parse, re, os, base64
tok=None
for l in open(os.path.expanduser("~/.git-credentials")):
    m=re.match(r"https://[^:]*:([^@]+)@github.com",l.strip())
    if m: tok=m.group(1)
H={"Accept":"application/vnd.github+json","User-Agent":"survey"}
if tok: H["Authorization"]="Bearer "+tok
def get(url):
    for t in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=H),timeout=40) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403,429): time.sleep(30)
            elif e.code==422: return None
            else: time.sleep(3)
        except Exception: time.sleep(3)
topics=["mamba state space model","linear attention transformer","mixture of experts llm","rwkv","xlstm","long context llm","kv cache compression","llm pretraining framework","small language model","llm quantization","speculative decoding","reasoning grpo reinforcement learning llm","sparse attention","looped transformer","test-time training","muon optimizer","llm distillation","model merging llm","diffusion language model","hybrid ssm attention","gated delta net","multi-token prediction","tokenizer bpe llm","llm data curation pretraining","latent attention mla","bitnet 1-bit llm","memory layer language model","byte level language model","nanogpt","llm evaluation harness"]
ranges=["stars:>10000","stars:2000..10000","stars:500..1999","stars:100..499","stars:20..99","stars:5..19","stars:1..4"]
repos={}
if os.path.exists("repos.json"): repos={r["full_name"]:r for r in json.load(open("repos.json"))}
for t in topics:
    for rg in ranges:
        q=urllib.parse.quote(f"{t} {rg}")
        d=get(f"https://api.github.com/search/repositories?q={q}&sort=stars&per_page=30")
        time.sleep(2.2)
        if not d: continue
        for it in d.get("items",[]):
            if it["full_name"] not in repos:
                repos[it["full_name"]]={k:it.get(k) for k in("full_name","description","stargazers_count","language","pushed_at","created_at","html_url","topics")}|{"query":t}
    print(t,len(repos),flush=True)
    json.dump(list(repos.values()),open("repos.json","w"))
print("TOTAL",len(repos))
