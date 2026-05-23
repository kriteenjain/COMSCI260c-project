"""HumanEval pass@1 evaluation.

We follow the standard recipe:
  * feed the function signature + docstring as prompt
  * greedy decode, truncate completion at the next top-level "def"/"class"
    or triple-backtick / "if __name__"
  * stitch (prompt + completion) and run the test against entry_point in a
    subprocess with a hard timeout

Subprocess isolation is important: HumanEval is small and trusted, but
generated code can `import os; os.system(...)` so we keep it cheap and
sandboxed-by-process. For stronger isolation you'd want Docker.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from .model import LM, GenConfig


STOP_TOKENS = [
    "\nclass ",
    "\ndef ",
    "\n#",
    "\nif __name__",
    "\nprint(",
    "\n```",
]


def truncate_completion(completion: str) -> str:
    """Cut the completion at the earliest stop sequence so we don't paste
    a second function on top of the canonical one."""
    end = len(completion)
    for stop in STOP_TOKENS:
        idx = completion.find(stop)
        if idx != -1 and idx < end:
            end = idx
    return completion[:end]


def _worker(program: str, conn) -> None:
    """Run `program` in this subprocess; report success/failure over pipe."""
    try:
        # Limit imports to reduce blast radius a bit. Not real sandboxing.
        ns: dict = {}
        # Some HumanEval tests rely on these being builtins, which they are.
        exec(program, ns)
        conn.send(("ok", None))
    except BaseException as e:  # noqa: BLE001 -- want to catch SystemExit too
        conn.send(("fail", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_with_timeout(program: str, timeout_s: float = 20.0) -> tuple[bool, str]:
    parent_conn, child_conn = mp.Pipe()
    # On Linux/Colab, "fork" avoids heavy spawn startup overhead and reduces
    # false timeouts. Keep "spawn" on macOS for safer process isolation.
    start_method = "fork" if sys.platform.startswith("linux") else "spawn"
    ctx = mp.get_context(start_method)
    p = ctx.Process(target=_worker, args=(program, child_conn))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(0.5)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    if not parent_conn.poll():
        return False, "no_result"
    status, err = parent_conn.recv()
    return status == "ok", err or ""


@dataclass
class HEItem:
    task_id: str
    completion: str
    passed: bool
    error: str


def evaluate(lm: LM, limit: int | None = None, batch_size: int = 4) -> dict:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    cfg = GenConfig(max_new_tokens=512, temperature=0.0)

    items: List[HEItem] = []
    n_passed = 0

    prompts: List[str] = []
    metas: List[dict] = []

    def flush():
        nonlocal n_passed
        completions = lm.generate(prompts, cfg)
        for meta, comp in zip(metas, completions):
            completion = truncate_completion(comp)
            program = (
                meta["prompt"]
                + completion
                + "\n\n"
                + meta["test"]
                + f"\n\ncheck({meta['entry_point']})\n"
            )
            passed, err = run_with_timeout(program, timeout_s=5.0)
            if passed:
                n_passed += 1
            items.append(
                HEItem(
                    task_id=meta["task_id"],
                    completion=completion,
                    passed=passed,
                    error=err,
                )
            )
        prompts.clear()
        metas.clear()

    for ex in tqdm(ds, desc="humaneval"):
        prompts.append(ex["prompt"])
        metas.append(
            {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
            }
        )
        if len(prompts) >= batch_size:
            flush()
    if prompts:
        flush()

    n = len(items)
    return {
        "task": "humaneval",
        "n": n,
        "n_passed": n_passed,
        "pass@1": n_passed / n if n else 0.0,
        "examples": [
            {
                "task_id": it.task_id,
                "passed": it.passed,
                "error": it.error,
                "completion": it.completion,
            }
            for it in items
        ],
    }
