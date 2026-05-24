"""Baseline evaluation entrypoint.

Example:
    python run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --task gsm8k --limit 20
    python run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --task humaneval --limit 10
    python run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --task both --limit 20

Results are written to results/<run-id>.json so we can diff against
compressed runs later.
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

MODEL_PRESETS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default="qwen-1.5b",
        help="Named model preset to run.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Optional HF model id. Overrides --preset when provided.",
    )
    ap.add_argument(
        "--task",
        choices=["gsm8k", "humaneval", "both"],
        default="both",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples per task. Use a small value for smoke tests; None/-1 for full.",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--tag",
        default="baseline",
        help="Free-form label for the run (e.g. 'baseline', 'wanda-2:4', 'awq-int4').",
    )
    ap.add_argument(
        "--out-dir",
        default="results",
    )
    args = ap.parse_args()
    model_name = args.model or MODEL_PRESETS[args.preset]

    limit = None if args.limit is None or args.limit < 0 else args.limit

    t0 = time.time()
    print(f"[load] {model_name}", flush=True)
    lm = LM(model_name)
    info = lm.info()
    print(f"[load] device={info['device']} dtype={info['dtype']} n_params={info['n_params']:,}", flush=True)

    results: dict = {
        "tag": args.tag,
        "model_info": info,
        "limit": limit,
        "batch_size": args.batch_size,
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
    model_slug = model_name.replace("/", "_")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{stamp}_{args.tag}_{model_slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
