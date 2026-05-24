# CS 260C — Compression × Task-Type Eval (Baseline)

End-to-end **uncompressed** evaluation pipeline for our group project on how
Wanda / AWQ compression affects LLM performance across task types. This is
the v0: one model, two tasks, greedy decoding, JSON results. Compression
wrappers plug in later without touching the eval code.

## Layout

```
cs260c/
├── run_baseline.py            # CLI: uncompressed eval
├── run_qwen_compression.py    # CLI: compress (Wanda / AWQ) then eval
├── requirements.txt
├── src/
│   ├── model.py               # HF causal-LM loader + batched generate
│   ├── gsm8k.py               # 4-shot CoT prompt, last-number extraction, exact match
│   ├── humaneval.py           # generate, truncate at stop tokens, subprocess exec w/ timeout
│   └── compression/
│       ├── wanda.py           # Wanda pruning (generic HF decoder-only)
│       └── awq_quant.py       # AWQ INT4 quantization via autoawq (CUDA only)
└── results/                   # JSON dumps, one per run
```

## Setup

```bash
cd ~/Downloads/cs260c
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default model (`Qwen/Qwen2.5-1.5B-Instruct`) is open, ~3 GB, and runs
on Apple Silicon via MPS in fp16. To use LLaMA-3.2-3B-Instruct you need a Hugging Face
token with access to the gated repo:

```bash
huggingface-cli login   # paste your read token
python run_baseline.py --model meta-llama/Llama-3.2-3B-Instruct --task both --limit 50
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

## Compression runs

`run_qwen_compression.py` is the compression analogue of `run_baseline.py`.
It loads a model, compresses it in place (Wanda) or quantizes it (AWQ),
then runs the same GSM8K / HumanEval evaluators so the JSON files line up
exactly with the baselines.

### Setup on Colab (GPU runtime)

```bash
pip install -r requirements.txt
# AWQ only; Wanda has no extra deps. CUDA required.
pip install autoawq
```

### Wanda (unstructured 50%)

```bash
python run_qwen_compression.py --method wanda --task both --limit 20
```

### Wanda (2:4 structured, save the pruned model for reuse)

```bash
python run_qwen_compression.py \
    --method wanda --sparsity-type 2:4 \
    --task both --limit 50 \
    --save-compressed ./compressed/qwen-wanda-2-4

# Later, skip the prune step and just re-eval:
python run_qwen_compression.py --method wanda --task humaneval --limit -1 \
    --load-compressed ./compressed/qwen-wanda-2-4
```

### AWQ INT4

```bash
python run_qwen_compression.py --method awq --task both --limit 20 \
    --save-compressed ./compressed/qwen-awq-w4g128
```

### Colab notebook

`notebooks/run_qwen_compression_colab.ipynb` clones the repo, installs
deps, runs the FP16 baseline + Wanda (unstructured + 2:4) + AWQ, and
prints a summary table — all end-to-end on a T4 runtime.

### Notes
- Wanda uses WikiText-2 as calibration (`--nsamples 128 --seqlen 2048` by
  default); the paper's C4 default works too but takes longer to download.
- AWQ requires CUDA (the `autoawq` kernels are CUDA-only). Run AWQ on
  Colab, not on a Mac.
- For LLaMA: same script, just `--model meta-llama/Llama-3.2-3B-Instruct`.
- A future `scripts/compare.py` should read two JSONs (baseline +
  compressed) and produce a per-task delta table.

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
