# CS 260C — Compression × Task-Type Eval (Baseline)

End-to-end **uncompressed** evaluation pipeline for our group project on how
Wanda / AWQ compression affects LLM performance across task types. This is
the v0: one model, two tasks, greedy decoding, JSON results. Compression
wrappers plug in later without touching the eval code.

## Layout

```
cs260c/
├── run_baseline.py        # CLI entrypoint
├── requirements.txt
├── src/
│   ├── model.py           # HF causal-LM loader + batched generate
│   ├── gsm8k.py           # 4-shot CoT prompt, last-number extraction, exact match
│   └── humaneval.py       # generate, truncate at stop tokens, subprocess exec w/ timeout
└── results/               # JSON dumps, one per run
```

## Setup

```bash
cd ~/Downloads/cs260c
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default model (`Qwen/Qwen2.5-1.5B-Instruct`) is open, ~3 GB, and runs
on Apple Silicon via MPS in fp16. To use LLaMA-2 you need a Hugging Face
token with access to the gated repo:

```bash
huggingface-cli login   # paste your read token
python run_baseline.py --model meta-llama/Llama-2-7b-hf --task both --limit 50
```

## Smoke test

```bash
python run_baseline.py --task both --limit 5
```

Expected on `Qwen2.5-1.5B-Instruct` (rough, MPS):
- gsm8k @ 5 examples: 3–5 correct
- humaneval @ 5 examples: 2–4 passing
- runtime: ~2–4 min on M-series Mac

## Bigger runs

```bash
# Full GSM8K test set (1319 problems) — plan for ~1–2 h on a single GPU
python run_baseline.py --task gsm8k --limit -1

# Full HumanEval (164 problems)
python run_baseline.py --task humaneval --limit -1
```

## Compute notes

For the **baseline** you do not need a VM — any laptop with ≥16 GB RAM and
MPS or a discrete GPU will run the default 1.5B model. For the real
experiments described in the proposal:

| Model        | Min VRAM (fp16) | Min VRAM (int4) |
|--------------|-----------------|-----------------|
| LLaMA-2-7B   | ~14 GB          | ~5 GB           |
| LLaMA-2-13B  | ~26 GB          | ~9 GB           |
| LLaMA-2-70B  | ~140 GB         | ~40 GB          |

Realistic options: UCLA Hoffman2 (request a GPU node), Lambda Labs A100
hourly, Colab Pro A100, or AWS g5.xlarge / g5.12xlarge.

## What's next (compression wiring)

1. **Wanda** — clone the Wanda repo, run their pruning script on a base
   LLaMA-2 checkpoint to produce a pruned `state_dict`. Wrap loading in
   `src/model.py` so we can do
   `LM("meta-llama/Llama-2-7b-hf", weights_override="...wanda.pt")`.
2. **AWQ** — install `autoawq`; load via
   `AutoAWQForCausalLM.from_quantized(...)`. Add a `--quant awq` flag to
   the CLI that swaps the loader.
3. Re-run `run_baseline.py --tag wanda-2:4` / `--tag awq-int4` so result
   JSONs are directly comparable.
4. Add a third task (TriviaQA or MMLU-flash) to cover "factual recall" per
   the proposal's three-axis hypothesis.

## Output schema

Each run writes `results/<timestamp>_<tag>_<model-slug>.json`:

```json
{
  "tag": "baseline",
  "model_info": {"model": "...", "device": "mps", "dtype": "torch.float16", "n_params": 1543714304},
  "tasks": {
    "gsm8k":     {"accuracy": 0.55, "n": 20, "n_correct": 11, "examples": [...]},
    "humaneval": {"pass@1":   0.30, "n": 20, "n_passed":  6, "examples": [...]}
  },
  "elapsed_s": 137.4
}
```

A future `scripts/compare.py` will read two JSONs and produce a delta
table per task — that's how we'll quantify "compression hurts task type X
more than task type Y" for the final report.
