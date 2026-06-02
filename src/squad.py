"""SQuAD v1.1 reading comprehension evaluation.

SQuAD (Rajpurkar et al., 2016) provides a Wikipedia passage and asks a
simple factual question answerable by a short span from that passage. It is
the easiest task in this suite: the answer is always present in the context
and requires only single-hop extraction, no reasoning chain. High baseline
scores (~70–85% EM) give compression plenty of room to show degradation.

We score with exact match and token F1 against all provided gold answers,
using the same normalisation as musique.py and nq_open.py.

Reference: https://arxiv.org/abs/1606.05250
Dataset:   rajpurkar/squad, validation split (10,570 examples)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from .model import LM, GenConfig


# ---------------------------------------------------------------------------
# Prompt — 0-shot; the passage makes the task self-evident
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "Answer the question using only the provided passage. "
    "Give a short, direct answer — a word or phrase from the passage."
)


def build_prompt(context: str, question: str) -> str:
    lines = [_INSTRUCTION, "", f"Passage: {context}", "", f"Question: {question}", "Answer:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer normalisation (standard SQuAD recipe, shared with other tasks)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in set(pred_toks))
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _score(pred: str, answers: list[str]) -> tuple[bool, float]:
    norm_pred = _normalize(pred)
    exact = any(norm_pred == _normalize(a) for a in answers)
    f1 = max(_token_f1(pred, a) for a in answers)
    return exact, f1


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@dataclass
class SQuADItem:
    question: str
    gold: list[str]
    pred: str
    exact: bool
    f1: float


def evaluate(lm: LM, limit: int | None = None, batch_size: int = 4) -> dict:
    ds = load_dataset("rajpurkar/squad", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # SQuAD answers are short spans — 64 tokens is generous.
    cfg = GenConfig(max_new_tokens=64, temperature=0.0)

    items: List[SQuADItem] = []
    n_correct = 0
    total_f1 = 0.0

    prompts: List[str] = []
    metas: List[dict] = []

    def flush() -> None:
        nonlocal n_correct, total_f1
        completions = lm.generate(prompts, cfg)
        for meta, comp in zip(metas, completions):
            # Take first line, then strip any "Human..."  continuation that
            # Qwen-Instruct appends on the same line when given plain-text prompts
            # (e.g. "Denver Broncos.Human resources department is responsible...").
            pred = comp.split("\n")[0]
            pred = re.split(r"\.?\s*Human\b", pred)[0].strip().rstrip(".")
            exact, f1 = _score(pred, meta["answers"])
            if exact:
                n_correct += 1
            total_f1 += f1
            items.append(SQuADItem(
                question=meta["question"],
                gold=meta["answers"],
                pred=pred,
                exact=exact,
                f1=f1,
            ))
        prompts.clear()
        metas.clear()

    for ex in tqdm(ds, desc="squad"):
        prompts.append(build_prompt(ex["context"], ex["question"]))
        metas.append({
            "question": ex["question"],
            "answers": ex["answers"]["text"],
        })
        if len(prompts) >= batch_size:
            flush()
    if prompts:
        flush()

    n = len(items)
    return {
        "task": "squad",
        "n": n,
        "n_correct": n_correct,
        "exact_match": n_correct / n if n else 0.0,
        "f1": total_f1 / n if n else 0.0,
        "examples": [
            {
                "question": it.question,
                "gold": it.gold,
                "pred": it.pred,
                "exact": it.exact,
                "f1": round(it.f1, 4),
            }
            for it in items
        ],
    }
