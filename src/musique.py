"""MuSiQue multi-hop QA evaluation.

MuSiQue (Trivedi et al., 2022) requires 2–4 reasoning hops across Wikipedia
paragraphs. We use the answerable validation split, include only the labeled
supporting paragraphs in each prompt (supporting-only keeps prompt length
predictable and isolates reasoning ability from context-length effects), and
score with exact match and token-level F1 — the standard metrics from the
paper.

Reference: https://arxiv.org/abs/2108.00573
Dataset:   bdsaglam/musique, config "answerable" (mirrors musique_ans_v1.0)
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
# Prompt
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "Answer the question using only the provided passages. "
    "Give a short, direct answer — a word or brief phrase, not a full sentence."
)


def build_prompt(question: str, supporting: list) -> str:
    lines = [_INSTRUCTION, "", "Passages:"]
    for p in supporting:
        lines.append(f"[{p['title']}] {p['paragraph_text']}")
    lines += ["", f"Question: {question}", "Answer:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer normalisation (standard SQuAD / NQ recipe)
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


def _score(pred: str, answer: str, aliases: list[str]) -> tuple[bool, float]:
    """Return (exact_match, best_f1) against the answer and all its aliases."""
    all_answers = [answer] + (aliases or [])
    norm_pred = _normalize(pred)
    exact = any(norm_pred == _normalize(a) for a in all_answers)
    f1 = max(_token_f1(pred, a) for a in all_answers)
    return exact, f1


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@dataclass
class MuSiQueItem:
    question: str
    gold: str
    pred: str
    exact: bool
    f1: float


def evaluate(lm: LM, limit: int | None = None, batch_size: int = 4) -> dict:
    # "answerable" config maps to musique_ans_v1.0 — only answerable examples, no filter needed.
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Short answers: 64 new tokens is generous for a word or phrase.
    cfg = GenConfig(max_new_tokens=64, temperature=0.0)

    items: List[MuSiQueItem] = []
    n_correct = 0
    total_f1 = 0.0

    prompts: List[str] = []
    metas: List[dict] = []

    def flush() -> None:
        nonlocal n_correct, total_f1
        completions = lm.generate(prompts, cfg)
        for meta, comp in zip(metas, completions):
            # First line only — the model sometimes continues after the answer.
            pred = comp.split("\n")[0].strip()
            exact, f1 = _score(pred, meta["answer"], meta["aliases"])
            if exact:
                n_correct += 1
            total_f1 += f1
            items.append(MuSiQueItem(
                question=meta["question"],
                gold=meta["answer"],
                pred=pred,
                exact=exact,
                f1=f1,
            ))
        prompts.clear()
        metas.clear()

    for ex in tqdm(ds, desc="musique"):
        supporting = [p for p in ex["paragraphs"] if p["is_supporting"]]
        prompts.append(build_prompt(ex["question"], supporting))
        metas.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "aliases": ex.get("answer_aliases") or [],
        })
        if len(prompts) >= batch_size:
            flush()
    if prompts:
        flush()

    n = len(items)
    return {
        "task": "musique",
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
