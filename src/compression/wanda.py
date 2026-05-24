"""Wanda pruning, generic for any HF decoder-only LM that exposes
`model.model.layers` (LLaMA, Qwen2/2.5, Mistral, etc.).

Reference: Sun, Liu, Bair, Kolter. "A Simple and Effective Pruning Approach
for Large Language Models." arXiv:2306.11695. Code adapted from
https://github.com/locuslab/wanda (which is hard-coded for LLaMA / OPT).

The Wanda importance metric for an output row of a Linear is
        S_ij = |W_ij| * ||X_:j||_2
where ||X_:j||_2 is the L2 norm across the calibration batch of the j-th
input feature. We prune the lowest-scoring entries per output row, which
gives 50% (or N:M) unstructured sparsity without retraining.

We do the pruning layer-by-layer in the order the network executes so that
each layer's activation stats reflect the pruned weights of all upstream
layers (this is what the paper calls "sequential" Wanda).
"""

from __future__ import annotations

import random
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_linears(module: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """Recursively collect every nn.Linear inside `module`. Returns a dict
    keyed by the dotted attribute path so we can log per-layer stats."""
    found: Dict[str, nn.Linear] = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found[full] = child
        else:
            found.update(find_linears(child, full))
    return found


class ActStats:
    """Streaming accumulator for per-input-feature squared L2 norm.

    We need sqrt(sum_x ||x_:j||^2 / N) for the Wanda score. Keeping a
    running mean lets us process calibration samples one at a time without
    blowing up memory."""

    def __init__(self, linear: nn.Linear):
        self.n_in = linear.weight.shape[1]
        self.dev = linear.weight.device
        self.scaler_row = torch.zeros(self.n_in, device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        # `inp` is whatever was passed as the first positional arg to the
        # Linear, typically [batch, seq, in_features] for a decoder block.
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if inp.dim() == 3:
            tokens = inp.shape[0] * inp.shape[1]
            x = inp.reshape(-1, inp.shape[-1])
        else:
            tokens = inp.shape[0]
            x = inp
        x = x.to(self.dev, dtype=torch.float32)
        # Online mean: keep a running average of ||X_:j||^2 over all tokens
        # seen so far. Matches the implementation in locuslab/wanda.
        new_n = self.nsamples + tokens
        self.scaler_row.mul_(self.nsamples / new_n)
        self.scaler_row.add_(x.pow(2).sum(dim=0) / new_n)
        self.nsamples = new_n


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------


def get_wikitext2_calibration(tokenizer, nsamples: int = 128, seqlen: int = 2048, seed: int = 0):
    """Return a list of `nsamples` token-id tensors of shape [1, seqlen].

    WikiText-2 is ~10 MB so it downloads quickly on Colab and is plenty for
    Wanda's activation statistics (the paper uses the C4 train split but the
    method is not very sensitive to the calibration corpus)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    if ids.shape[0] < seqlen + 1:
        raise RuntimeError(
            f"WikiText-2 has only {ids.shape[0]} tokens after tokenization, "
            f"need at least {seqlen + 1}. Try a smaller --seqlen."
        )
    rng = random.Random(seed)
    samples = []
    for _ in range(nsamples):
        i = rng.randint(0, ids.shape[0] - seqlen - 1)
        samples.append(ids[i : i + seqlen].unsqueeze(0))
    return samples


# ---------------------------------------------------------------------------
# Sequential pruning
# ---------------------------------------------------------------------------


def _capture_layer0_inputs(model, samples, device):
    """Run the embedding + pre-block stack on each calibration sample,
    grabbing the hidden states (and kwargs) that would be passed into
    `model.model.layers[0]`. Returns (inps, kwargs_for_layer)."""
    layers = model.model.layers
    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size
    nsamples = len(samples)
    seqlen = samples[0].shape[1]

    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=device)
    cache: Dict[str, object] = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden_states, **kwargs):
            inps[cache["i"]] = hidden_states  # type: ignore[index]
            cache["i"] = cache["i"] + 1  # type: ignore[operator]
            # Keep one snapshot of the auxiliary kwargs (attention mask,
            # position_ids, position_embeddings, ...). They only depend on
            # seqlen, which is constant across samples, so reusing them for
            # every sample is safe.
            cache["kwargs"] = {k: v for k, v in kwargs.items() if k != "past_key_value"}
            raise _StopForward()

    layers[0] = Catcher(layers[0])
    for s in samples:
        try:
            model(s.to(device))
        except _StopForward:
            pass
    layers[0] = layers[0].layer  # type: ignore[assignment]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inps, cache["kwargs"] or {}


class _StopForward(Exception):
    """Sentinel raised by the Catcher to short-circuit the model forward
    once we've grabbed the inputs to layer 0."""


def _compute_mask(W_metric: torch.Tensor, sparsity_ratio: float, prune_n: int, prune_m: int) -> torch.Tensor:
    """Return a boolean mask of weights to ZERO. Per-output-row pruning."""
    if prune_n > 0:
        # N:M structured: for every consecutive M weights in a row, zero
        # out the N smallest.
        mask = torch.zeros_like(W_metric, dtype=torch.bool)
        cols = W_metric.shape[1]
        for col in range(0, cols, prune_m):
            block = W_metric[:, col : col + prune_m]
            _, idx = torch.topk(block.float(), prune_n, dim=1, largest=False)
            mask.scatter_(1, col + idx, True)
        return mask

    # Unstructured: zero the `sparsity_ratio` fraction smallest entries
    # per output row.
    k = int(W_metric.shape[1] * sparsity_ratio)
    mask = torch.zeros_like(W_metric, dtype=torch.bool)
    if k > 0:
        _, idx = torch.topk(W_metric, k, dim=1, largest=False)
        mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def prune_wanda(
    model,
    tokenizer,
    *,
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    """In-place Wanda prune of `model`. Returns a small stats dict.

    Args:
        model: an HF causal LM. Must have `model.model.layers`.
        tokenizer: the matching tokenizer (used for calibration data).
        sparsity_ratio: fraction of weights to zero per output row when
            sparsity_type='unstructured'.
        sparsity_type: 'unstructured', '2:4', or '4:8'.
        nsamples: number of calibration samples.
        seqlen: tokens per calibration sample.
        seed: RNG seed for sampling calibration windows.
        device: where to run pruning. Defaults to the model's current device.
    """
    if device is None:
        device = next(model.parameters()).device.type

    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        try:
            prune_n, prune_m = (int(x) for x in sparsity_type.split(":"))
        except ValueError as e:
            raise ValueError(
                f"--sparsity-type must be 'unstructured' or 'N:M' (got {sparsity_type!r})"
            ) from e
        # When the user picks N:M we ignore --sparsity-ratio (it's implied
        # to be N/M) but still print it for the record.
        sparsity_ratio = prune_n / prune_m

    use_cache_orig = model.config.use_cache
    model.config.use_cache = False

    print(f"[wanda] loading WikiText-2 calibration ({nsamples} x {seqlen} tokens)", flush=True)
    samples = get_wikitext2_calibration(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed)

    print("[wanda] capturing inputs to layer 0", flush=True)
    inps, layer_kwargs = _capture_layer0_inputs(model, samples, device)
    outs = torch.zeros_like(inps)

    layers = model.model.layers
    per_layer_stats = []

    for i in tqdm(range(len(layers)), desc="[wanda] pruning"):
        layer = layers[i].to(device)
        subset = find_linears(layer)

        # Skip empty layers (shouldn't happen for transformer blocks but be safe).
        if not subset:
            continue

        # Hook every linear in this block to accumulate activation norms.
        stats: Dict[str, ActStats] = {name: ActStats(m) for name, m in subset.items()}
        handles = []
        for name, m in subset.items():
            def make_hook(n):  # noqa: ANN001 -- closure over local
                def fn(_mod, _inp, _out):
                    stats[n].add_batch(_inp[0].data)
                return fn
            handles.append(m.register_forward_hook(make_hook(name)))

        # First forward: feed each calibration sample through this layer,
        # using the current (still-dense) weights, to populate `stats`.
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for h in handles:
            h.remove()

        # Compute Wanda score and apply mask per-linear.
        layer_total = 0
        layer_zeros = 0
        for name, m in subset.items():
            W = m.weight.data
            scaler = stats[name].scaler_row.sqrt().to(W.device)
            W_metric = W.abs() * scaler.reshape(1, -1)
            mask = _compute_mask(W_metric, sparsity_ratio, prune_n, prune_m)
            W[mask] = 0
            layer_total += W.numel()
            layer_zeros += int(mask.sum().item())

        per_layer_stats.append(
            {
                "layer": i,
                "sparsity": layer_zeros / layer_total if layer_total else 0.0,
                "modules": list(subset.keys()),
            }
        )

        # Second forward (now with pruned weights) to produce the inputs
        # for layer i+1. This is the "sequential" part of Wanda.
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer
        inps, outs = outs, inps

    model.config.use_cache = use_cache_orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall = check_sparsity(model)
    print(f"[wanda] done. overall sparsity = {overall:.4f}", flush=True)
    return {
        "overall_sparsity": overall,
        "sparsity_ratio": sparsity_ratio,
        "sparsity_type": sparsity_type,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "per_layer": per_layer_stats,
    }


@torch.no_grad()
def check_sparsity(model) -> float:
    """Fraction of pruned (==0) weights across all decoder-block Linears.
    Ignores embeddings and the LM head so the number is directly comparable
    to the Wanda paper's reported sparsity."""
    total = 0
    zeros = 0
    for block in model.model.layers:
        for _, m in find_linears(block).items():
            total += m.weight.numel()
            zeros += int((m.weight == 0).sum().item())
    return zeros / total if total else 0.0
