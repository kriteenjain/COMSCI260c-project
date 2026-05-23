"""GSM8K evaluation.

Standard recipe used in most LLM eval papers:
  * 4-shot chain-of-thought prompt
  * greedy decode
  * extract the last number from the completion, compare to gold

Gold answers in GSM8K end with "#### <number>"; we strip commas and match
the numeric value to the model's extracted number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from .model import LM, GenConfig


# Classic 4-shot CoT exemplars (from the original GSM8K / chain-of-thought
# prompting papers). Keep them small so they fit in any context window.
FEWSHOT = [
    {
        "q": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "a": "Natalia sold 48 clips in April. In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    },
    {
        "q": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "a": "She earns 12/60 = $0.2 per minute. 50 minutes of work means she earned 0.2 * 50 = $10.\n#### 10",
    },
    {
        "q": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "a": "Betty has 100 / 2 = $50. Her grandparents gave her 15 * 2 = $30. In total she has 50 + 15 + 30 = $95. She still needs 100 - 95 = $5.\n#### 5",
    },
    {
        "q": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "a": "Today she read 12 * 2 = 24 pages. Total read so far: 12 + 24 = 36. Remaining: 120 - 36 = 84. Half of remaining: 84 / 2 = 42.\n#### 42",
    },
]


def build_prompt(question: str) -> str:
    parts = []
    for ex in FEWSHOT:
        parts.append(f"Question: {ex['q']}\nAnswer: {ex['a']}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


# Match either "#### <num>" or the last signed number with optional commas/decimals.
_HASH_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    m = _HASH_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def gold_answer(answer_field: str) -> str:
    # GSM8K gold always contains "#### <num>" at the end.
    m = _HASH_RE.search(answer_field)
    if not m:
        raise ValueError(f"Malformed GSM8K gold answer: {answer_field!r}")
    return m.group(1).replace(",", "")


def _norm(x: str) -> str:
    x = x.strip().rstrip(".")
    # Treat 5 and 5.0 as equal.
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return x


@dataclass
class GSM8KItem:
    question: str
    gold: str
    completion: str
    pred: str | None
    correct: bool


def evaluate(lm: LM, limit: int | None = None, batch_size: int = 4) -> dict:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    cfg = GenConfig(max_new_tokens=256, temperature=0.0)

    items: List[GSM8KItem] = []
    n_correct = 0

    prompts: List[str] = []
    metas: List[dict] = []

    def flush():
        nonlocal n_correct
        completions = lm.generate(prompts, cfg)
        for meta, comp in zip(metas, completions):
            # Stop at next "Question:" if the model keeps going.
            comp_trimmed = comp.split("Question:")[0].strip()
            pred = extract_answer(comp_trimmed)
            correct = pred is not None and _norm(pred) == _norm(meta["gold"])
            if correct:
                n_correct += 1
            items.append(
                GSM8KItem(
                    question=meta["question"],
                    gold=meta["gold"],
                    completion=comp_trimmed,
                    pred=pred,
                    correct=correct,
                )
            )
        prompts.clear()
        metas.clear()

    for ex in tqdm(ds, desc="gsm8k"):
        prompts.append(build_prompt(ex["question"]))
        metas.append({"question": ex["question"], "gold": gold_answer(ex["answer"])})
        if len(prompts) >= batch_size:
            flush()
    if prompts:
        flush()

    n = len(items)
    return {
        "task": "gsm8k",
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "examples": [
            {
                "question": it.question,
                "gold": it.gold,
                "pred": it.pred,
                "correct": it.correct,
                "completion": it.completion,
            }
            for it in items
        ],
    }
