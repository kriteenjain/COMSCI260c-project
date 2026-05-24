"""Compression backends for the CS260C baseline pipeline.

Each module exposes one entrypoint that mutates / replaces an HF causal-LM
in place so the rest of the evaluation code (src.gsm8k, src.humaneval)
does not need to change.
"""
