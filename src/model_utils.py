from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_cfg: Dict[str, Any]) -> AutoTokenizer:
    """
    Load a Hugging Face tokenizer and make sure padding is available
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["hf_id"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Load a Hugging Face causal language model
    """
    hf_id = model_cfg["hf_id"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }

    if device.type == "cuda":
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    model.to(device)

    if hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = True

    return model


def load_model_and_tokenizer(
    model_cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model and tokenizer as a pair
    """
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, device=device, dtype=dtype)

    if tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def batch_texts(texts: list[str], batch_size: int) -> Iterable[list[str]]:
    """
    Yield batches of texts
    """
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]


def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a batch and move tensors to the selected device
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {key: value.to(device) for key, value in encoded.items()}


def forward_hidden_states(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """
    Run a forward pass and return all hidden states
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "This model did not return hidden_states. Try another checkpoint or inspect its HF interface"
        )

    return hidden_states


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of model parameters
    """
    return sum(parameter.numel() for parameter in model.parameters())