"""NQ Open evaluation.

NQ Open (Kwiatkowski et al., 2019 / Lee et al., 2019) strips the Wikipedia
document context from Natural Questions and frames it as open-domain QA —
the model must answer from parametric knowledge alone. This makes it a clean
contrast to MuSiQue, which supplies supporting paragraphs.

We use a 3-shot prompt with short factual exemplars, greedy decode, and
score with exact match (primary) and token F1 (secondary) against the list
of gold answers.

Reference: https://arxiv.org/abs/1906.00300
Dataset:   google-research-datasets/nq_open, validation split (3,610 examples)
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
# Prompt — 3-shot keeps context short; NQ answers are single entities/phrases
# ---------------------------------------------------------------------------

FEWSHOT = [
    {"q": "who sang i will always love you in the bodyguard", "a": "Whitney Houston"},
    {"q": "what is the largest planet in the solar system",   "a": "Jupiter"},
    {"q": "who wrote the novel hamlet",                       "a": "William Shakespeare"},
]


def build_prompt(question: str) -> str:
    lines = ["Answer each question with a short phrase or name.", ""]
    for ex in FEWSHOT:
        lines.append(f"Q: {ex['q']}")
        lines.append(f"A: {ex['a']}")
        lines.append("")
    lines.append(f"Q: {question}")
    lines.append("A:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer normalisation (same recipe as musique.py / SQuAD standard)
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
class NQItem:
    question: str
    gold: list[str]
    pred: str
    exact: bool
    f1: float


def evaluate(lm: LM, limit: int | None = None, batch_size: int = 4) -> dict:
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # NQ answers are short (names, dates, places) — 32 tokens is generous.
    cfg = GenConfig(max_new_tokens=32, temperature=0.0)

    items: List[NQItem] = []
    n_correct = 0
    total_f1 = 0.0

    prompts: List[str] = []
    metas: List[dict] = []

    def flush() -> None:
        nonlocal n_correct, total_f1
        completions = lm.generate(prompts, cfg)
        for meta, comp in zip(metas, completions):
            pred = comp.split("\n")[0].strip()
            exact, f1 = _score(pred, meta["answers"])
            if exact:
                n_correct += 1
            total_f1 += f1
            items.append(NQItem(
                question=meta["question"],
                gold=meta["answers"],
                pred=pred,
                exact=exact,
                f1=f1,
            ))
        prompts.clear()
        metas.clear()

    for ex in tqdm(ds, desc="nq_open"):
        prompts.append(build_prompt(ex["question"]))
        metas.append({"question": ex["question"], "answers": ex["answer"]})
        if len(prompts) >= batch_size:
            flush()
    if prompts:
        flush()

    n = len(items)
    return {
        "task": "nq_open",
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
