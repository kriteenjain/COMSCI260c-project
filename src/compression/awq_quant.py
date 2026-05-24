"""AWQ (Activation-aware Weight Quantization) wrapper.

We use the `autoawq` package (pip-installable, CUDA-only) rather than the
original mit-han-lab/llm-awq repo because:
  * autoawq is the maintained third-party implementation the AWQ authors
    recommend for HF integration (linked from llm-awq README, [2023/09]).
  * It has first-class support for Qwen2 / Qwen2.5 architectures.
  * It avoids cloning the upstream repo and building CUDA kernels by hand
    on Colab; `pip install autoawq` ships pre-built wheels.

Reference: Lin et al. "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration." MLSys 2024 (Best Paper).
"""

from __future__ import annotations

import gc
import os
from typing import Tuple


def quantize_with_awq(
    model_path: str,
    quant_path: str,
    *,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
) -> str:
    """Run AWQ search + pseudo-quantize-then-pack on `model_path`, save the
    resulting INT4 weights to `quant_path`. Returns `quant_path`.

    Default config (w_bit=4, q_group_size=128, version='GEMM') matches the
    AWQ paper's main result for LLaMA / Qwen and is what HF's AutoAWQ docs
    recommend for a first run."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as e:
        raise RuntimeError(
            "autoawq is not installed. On Colab run:\n"
            "    pip install autoawq\n"
            "AWQ requires a CUDA GPU."
        ) from e
    from transformers import AutoTokenizer

    print(f"[awq] loading FP16 model from {model_path}", flush=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        trust_remote_code=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version,
    }
    print(f"[awq] running AWQ search + quantize, config={quant_config}", flush=True)
    model.quantize(tokenizer, quant_config=quant_config)

    os.makedirs(quant_path, exist_ok=True)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print(f"[awq] saved INT{w_bit} model to {quant_path}", flush=True)

    # Free the FP16 weights before evaluation reloads as INT4.
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return quant_path


def load_awq(quant_path: str, fuse_layers: bool = True) -> Tuple[object, object]:
    """Load an AWQ-quantized model previously saved by `quantize_with_awq`.

    Returns `(hf_model, tokenizer)` where `hf_model` is the underlying HF
    transformer (Qwen2ForCausalLM / LlamaForCausalLM / ...) with its
    `nn.Linear` layers replaced by AWQ `WQLinear_*` modules. The model is
    drop-in compatible with `model.generate(...)` and can be wrapped by
    `LM(..., model=hf_model, tokenizer=tokenizer)`.
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as e:
        raise RuntimeError(
            "autoawq is not installed. On Colab run: pip install autoawq"
        ) from e
    from transformers import AutoTokenizer

    awq_obj = AutoAWQForCausalLM.from_quantized(
        quant_path,
        fuse_layers=fuse_layers,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
    return awq_obj.model, tokenizer
