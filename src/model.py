"""Minimal HF causal-LM wrapper used by every task.

Kept intentionally small: load once, generate many. Compression methods
(Wanda / AWQ) will eventually plug in by mutating `self.model` after load
or by swapping the from_pretrained call, so the eval scripts don't need
to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    # bf16 on CUDA (Ampere+) gives speed/accuracy win; fp16 on MPS; fp32 on CPU.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0


class LM:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        model=None,
        tokenizer=None,
    ):
        """Either load a fresh HF model by name, OR wrap a pre-loaded
        `(model, tokenizer)` pair. The latter path is used by the compression
        runners (Wanda / AWQ) which mutate or replace the model before eval.
        """
        self.model_name = model_name

        if model is not None and tokenizer is not None:
            # Caller already loaded (and possibly compressed) the model.
            self.model = model
            self.tokenizer = tokenizer
            try:
                p = next(self.model.parameters())
                self.device = device or p.device.type
                self.dtype = dtype or p.dtype
            except StopIteration:
                self.device = device or _pick_device()
                self.dtype = dtype or _pick_dtype(self.device)
        else:
            self.device = device or _pick_device()
            self.dtype = dtype or _pick_dtype(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

        # Decoder-only models should left-pad for correct batched generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Most causal LMs ship without pad token; reuse EOS for batched generation.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts: List[str], cfg: GenConfig) -> List[str]:
        """Generate completions for a batch of prompts. Returns only the
        newly generated text (prompt stripped)."""
        if not prompts:
            return []

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        do_sample = cfg.temperature > 0
        out = self.model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else 1.0,
            top_p=cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Strip prompt tokens; decode only the continuation.
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        completions: List[str] = []
        for i, full_ids in enumerate(out):
            new_ids = full_ids[enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            completions.append(text)
            _ = prompt_lens  # kept for future debug logging
        return completions

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "n_params": n_params,
        }
