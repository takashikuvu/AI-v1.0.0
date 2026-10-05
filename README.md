# Kiseki (奇跡) — 9B-class hybrid linear-attention / MoE architecture

This repo holds (1) an architecture design, (2) a working PyTorch reference implementation, (3) a literature/repo
harvest, and (4) a tiny-scale training run. **It does not contain a trained 9B model, and it makes no claim of
Claude-level performance.** See "What is and isn't established".

## Architecture (`kiseki/`)
| Component | Choice | Basis (from the harvested corpus / known work) |
|---|---|---|
| Token mixer | 3 x Gated DeltaNet : 1 x global GQA attention (24 of 32 layers linear) | Gated DeltaNet, Qwen3-Next-style hybrids |
| Attention layers | GQA (20q/4kv), QK-norm, partial RoPE, output gate | gated attention, QK-norm papers |
| FFN | 2 dense SwiGLU layers, then 64 fine-grained routed experts (top-6) + 1 shared expert, sigmoid router | DeepSeek-V3 / Qwen3-MoE line |
| Load balancing | auxiliary-loss-free bias update | DeepSeek-V3 |
| Objective | next-token + multi-token prediction (depth 1) | DeepSeek-V3 MTP |
| Size | `kiseki_9b()`: **9.37B total (9.08B w/o MTP), 2.23B active/token** | `python -c "from model import count_params"` |

Implementation is verified by `kiseki/test_model.py`: chunkwise-parallel delta rule == sequential recurrence
(err 4e-7), causality (err 0), KV/state-cached incremental decoding == full forward (err 1e-6).
The 9B config is only instantiated on the `meta` device (counting parameters); the sandbox has ~1 GB RAM, 2 CPU cores, no GPU.

## Survey (`survey/`)
`harvest_papers.py` / `harvest_repos.py` pulled **3,135 arXiv papers** (title + abstract, via arXiv API, 70 topical queries)
and **2,275 GitHub repos** (name/description/topics/stars, via search API; 11 repos >=10k stars down to 701 with 1-4 stars).
`analyze.py` -> `SURVEY_STATS.txt` gives keyword counts only.
**Honest caveat:** I did not read these 3,135 papers or open 2,275 repos' code. Only metadata/abstracts were collected and
keyword-counted; the architecture choices above lean substantially on prior knowledge of the main papers, which this survey
did not independently verify. The keyword counts show these components are heavily studied, not that they work together at 9B.
Also: an earlier version of the config came out at 16B parameters; I resized it to 9.4B.

## Tiny training run (`kiseki/train.py`)
2.6M-param variant (`kiseki_tiny`), byte-level, 6 MB of TinyStories, 600 steps on CPU (~6 min). Validation loss 5.55 -> ~1.25 nats/byte
(~1.8 bits/byte); MoE max-expert load ratio fell from 4.0 to ~1.2 (bias balancing works). Sample output is grammatical-ish
children's-story text (`kiseki/sample.txt`). This only demonstrates that the code trains; it says nothing about 9B quality.

## What is and isn't established
- Established: architecture is coherent, correct on the tests above, trains, and sums to ~9B.
- Not established: that it matches any Claude model. A frontier-level result is determined mainly by data, compute and
  post-training (RL, distillation), not architecture alone. Pretraining a 9B model needs on the order of 10^23 FLOPs
  (thousands of GPU-days) plus trillions of tokens; none of that exists here, so no 9B weights were produced.
  I also have no access to a "Claude 4.8" or to its benchmark scores, so parity can't be measured here.
- Pure-PyTorch delta-rule/MoE loops are slow; real training needs fused kernels (e.g. flash-linear-attention) and expert-parallel MoE.

## Next steps
1. Ablate hybrid ratio / expert count at ~100M-1B scale on a GPU with a real tokenizer and corpus.
2. Pretrain with Muon/AdamW + WSD schedule, FP8, ~10T+ tokens; then long-context extension.
3. Post-train: SFT + distillation from strong teachers + RLVR (GRPO); evaluate on public benchmarks.
