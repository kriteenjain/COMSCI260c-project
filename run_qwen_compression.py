"""Compress Qwen (Wanda or AWQ) and evaluate on GSM8K / HumanEval.

This is the compression analogue of `run_baseline.py`. The eval pipeline
is identical (same `src.gsm8k` / `src.humaneval` modules, same JSON output
schema) so result files from baseline and compressed runs can be diffed
directly.

Examples (the typical Colab workflow):

    # Wanda, unstructured 50% sparsity, both tasks, 20 examples each
    python run_qwen_compression.py --method wanda --task both --limit 20

    # Wanda 2:4 structured, save the pruned model for reuse
    python run_qwen_compression.py \\
        --method wanda --sparsity-type 2:4 \\
        --task gsm8k --limit 50 \\
        --save-compressed ./compressed/qwen-wanda-2-4

    # AWQ INT4, group size 128 (CUDA-only)
    python run_qwen_compression.py --method awq --task both --limit 20 \\
        --save-compressed ./compressed/qwen-awq-w4g128

    # Re-evaluate a previously compressed model without re-running the
    # search / pruning.
    python run_qwen_compression.py --method awq --task humaneval --limit -1 \\
        --load-compressed ./compressed/qwen-awq-w4g128

LLaMA-2 will work as a drop-in once you have HF access:
    python run_qwen_compression.py --method wanda \\
        --model meta-llama/Llama-2-7b-hf --task both --limit 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from src.model import LM
from src import gsm8k, humaneval


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _build_tag(args) -> str:
    if args.tag:
        return args.tag
    if args.method == "wanda":
        st = args.sparsity_type
        if st == "unstructured":
            return f"wanda-u{int(args.sparsity_ratio * 100)}"
        return f"wanda-{st.replace(':', '-')}"
    if args.method == "awq":
        return f"awq-w{args.w_bit}g{args.q_group_size}"
    return args.method


def _compress_wanda(args) -> tuple[LM, dict]:
    """Load the FP16 model, prune in place with Wanda, return (lm, stats)."""
    from src.compression.wanda import prune_wanda

    if args.load_compressed:
        print(f"[load] pruned model from {args.load_compressed}", flush=True)
        lm = LM(args.load_compressed)
        from src.compression.wanda import check_sparsity
        return lm, {"loaded_from": args.load_compressed, "overall_sparsity": check_sparsity(lm.model)}

    print(f"[load] {args.model}", flush=True)
    lm = LM(args.model)

    stats = prune_wanda(
        lm.model,
        lm.tokenizer,
        sparsity_ratio=args.sparsity_ratio,
        sparsity_type=args.sparsity_type,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        device=lm.device,
    )

    if args.save_compressed:
        os.makedirs(args.save_compressed, exist_ok=True)
        print(f"[save] pruned model -> {args.save_compressed}", flush=True)
        lm.model.save_pretrained(args.save_compressed)
        lm.tokenizer.save_pretrained(args.save_compressed)

    return lm, stats


def _compress_awq(args) -> tuple[LM, dict]:
    """Quantize (or load) with AWQ. Returns (lm, stats)."""
    from src.compression.awq_quant import load_awq, quantize_with_awq

    if args.load_compressed:
        quant_path = args.load_compressed
        print(f"[load] AWQ INT{args.w_bit} model from {quant_path}", flush=True)
    else:
        quant_path = args.save_compressed or f"./compressed/awq-w{args.w_bit}g{args.q_group_size}"
        quantize_with_awq(
            args.model,
            quant_path,
            w_bit=args.w_bit,
            q_group_size=args.q_group_size,
        )

    hf_model, tokenizer = load_awq(quant_path)
    lm = LM(args.model, model=hf_model, tokenizer=tokenizer)
    return lm, {
        "quant_path": quant_path,
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compress a Qwen (or any HF causal-LM) checkpoint with Wanda or AWQ and evaluate it.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id or local path to compress.")
    ap.add_argument("--method", choices=["wanda", "awq"], required=True,
                    help="Compression algorithm.")

    # Eval
    ap.add_argument("--task", choices=["gsm8k", "humaneval", "both"], default="both")
    ap.add_argument("--limit", type=int, default=20,
                    help="Number of examples per task. -1 for full split.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tag", default=None,
                    help="Run label (defaults to a method-specific string).")
    ap.add_argument("--out-dir", default="results")

    # Wanda-only
    ap.add_argument("--sparsity-ratio", type=float, default=0.5,
                    help="(Wanda) Fraction of weights to zero per output row.")
    ap.add_argument("--sparsity-type", default="unstructured",
                    choices=["unstructured", "2:4", "4:8"],
                    help="(Wanda) Sparsity pattern.")
    ap.add_argument("--nsamples", type=int, default=128,
                    help="(Wanda) Number of calibration samples.")
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="(Wanda) Tokens per calibration sample.")
    ap.add_argument("--seed", type=int, default=0,
                    help="(Wanda) Seed for calibration sampling.")

    # AWQ-only
    ap.add_argument("--w-bit", type=int, default=4,
                    help="(AWQ) Weight bit width: 4 (default) or 3.")
    ap.add_argument("--q-group-size", type=int, default=128,
                    help="(AWQ) Quantization group size.")

    # Cache
    ap.add_argument("--save-compressed", default=None,
                    help="Directory to save the compressed model after compression. "
                         "For AWQ this also defaults to ./compressed/awq-... if omitted.")
    ap.add_argument("--load-compressed", default=None,
                    help="Skip compression and load a previously-saved compressed model from this dir.")

    args = ap.parse_args()
    tag = _build_tag(args)

    t0 = time.time()
    if args.method == "wanda":
        lm, comp_stats = _compress_wanda(args)
    elif args.method == "awq":
        lm, comp_stats = _compress_awq(args)
    else:  # argparse guards this
        raise ValueError(args.method)
    compress_elapsed = round(time.time() - t0, 2)

    info = lm.info()
    print(
        f"[load] device={info['device']} dtype={info['dtype']} n_params={info['n_params']:,}",
        flush=True,
    )

    limit = None if args.limit < 0 else args.limit

    results: dict = {
        "tag": tag,
        "method": args.method,
        "model_info": info,
        "limit": limit,
        "batch_size": args.batch_size,
        "compression": {
            **comp_stats,
            "compress_elapsed_s": compress_elapsed,
            "config": {
                "sparsity_ratio": args.sparsity_ratio if args.method == "wanda" else None,
                "sparsity_type": args.sparsity_type if args.method == "wanda" else None,
                "nsamples": args.nsamples if args.method == "wanda" else None,
                "seqlen": args.seqlen if args.method == "wanda" else None,
                "w_bit": args.w_bit if args.method == "awq" else None,
                "q_group_size": args.q_group_size if args.method == "awq" else None,
            },
        },
        "started_at": dt.datetime.utcnow().isoformat() + "Z",
        "tasks": {},
    }

    if args.task in ("gsm8k", "both"):
        r = gsm8k.evaluate(lm, limit=limit, batch_size=args.batch_size)
        print(f"[gsm8k] accuracy={r['accuracy']:.3f}  ({r['n_correct']}/{r['n']})", flush=True)
        results["tasks"]["gsm8k"] = r

    if args.task in ("humaneval", "both"):
        r = humaneval.evaluate(lm, limit=limit, batch_size=args.batch_size)
        print(f"[humaneval] pass@1={r['pass@1']:.3f}  ({r['n_passed']}/{r['n']})", flush=True)
        results["tasks"]["humaneval"] = r

    results["elapsed_s"] = round(time.time() - t0, 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{stamp}_{tag}_{model_slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
