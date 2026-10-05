"""Harvest paper metadata (title/abstract/date/id) from arXiv API across many topical queries."""
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, sys, os
NS={'a':'http://www.w3.org/2005/Atom'}
QUERIES = {
 "ssm": ["state space model language", "mamba selective state space", "linear attention", "gated delta rule", "linear recurrent unit", "RWKV", "xLSTM", "test-time training layers", "titans memory neural long-term", "hybrid attention ssm"],
 "attn": ["multi-head latent attention", "grouped query attention", "sliding window attention", "sparse attention long context", "native sparse attention", "KV cache compression", "attention sink", "differential transformer", "rotary position embedding extension", "length extrapolation positional"],
 "moe": ["mixture of experts language model", "fine-grained expert routing", "expert choice routing", "auxiliary-loss-free load balancing", "upcycling dense to MoE", "mixture of depths", "mixture of recursions", "parameter efficient experts", "memory layers product key", "sparse upcycling"],
 "train": ["scaling laws language models", "data pruning quality filtering pretraining", "synthetic data pretraining textbook", "curriculum learning pretraining", "learning rate schedule warmup stable decay", "muon optimizer", "multi-token prediction", "FP8 training", "small language model training", "knowledge distillation language model"],
 "post": ["reinforcement learning verifiable rewards reasoning", "GRPO group relative policy optimization", "chain-of-thought distillation", "test-time compute scaling", "process reward model", "DPO preference optimization", "self-improvement language model", "long chain of thought", "RLHF reward model", "tool use agents language model"],
 "eff": ["quantization LLM 4-bit", "BitNet 1-bit LLM", "speculative decoding", "pruning large language model", "model merging", "parameter sharing layers", "looped transformer recurrent depth", "latent reasoning continuous thought", "diffusion language model", "byte-level language model"],
 "misc": ["retrieval augmented language model", "tokenizer vocabulary", "embedding dimension scaling", "normalization layer transformer", "residual stream hyper-connections", "activation function SwiGLU", "long context benchmark", "LLM evaluation benchmark contamination", "small models reasoning phi", "on-device language model"],
}
def fetch(q, start, n=100):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":"all:"+q if ' ' not in q else 'all:"'+q+'"',"start":start,"max_results":n,"sortBy":"relevance"})
    for t in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"research-survey/1.0"}),timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            time.sleep(5*(t+1))
    return None
papers={}
if os.path.exists("papers.json"): papers={p["id"]:p for p in json.load(open("papers.json"))}
for cat,qs in QUERIES.items():
    for q in qs:
        for start in (0,100):
            root=fetch(q,start)
            if root is None: continue
            for e in root.findall('a:entry',NS):
                i=e.find('a:id',NS).text.split('/abs/')[-1]
                if i in papers: papers[i]["tags"].append(cat) if cat not in papers[i]["tags"] else None; continue
                papers[i]={"id":i,"title":" ".join(e.find('a:title',NS).text.split()),"abstract":" ".join(e.find('a:summary',NS).text.split()),"published":e.find('a:published',NS).text[:10],"query":q,"tags":[cat]}
            time.sleep(3.2)
        print(cat,q,len(papers),flush=True)
        json.dump(list(papers.values()),open("papers.json","w"))
print("TOTAL",len(papers))
