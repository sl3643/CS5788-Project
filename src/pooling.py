from __future__ import annotations

import torch


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token representations using the attention mask

    Args:
        hidden_state: Tensor with shape [batch_size, seq_len, hidden_dim]
        attention_mask: Tensor with shape [batch_size, seq_len]

    Returns:
        Tensor with shape [batch_size, hidden_dim]
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    masked_hidden = hidden_state * mask
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return masked_hidden.sum(dim=1) / lengths


def last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool by selecting the last non-padding token for each example

    Args:
        hidden_state: Tensor with shape [batch_size, seq_len, hidden_dim]
        attention_mask: Tensor with shape [batch_size, seq_len]

    Returns:
        Tensor with shape [batch_size, hidden_dim]
    """
    batch_size = hidden_state.shape[0]
    last_indices = attention_mask.sum(dim=1).long() - 1
    last_indices = last_indices.clamp(min=0)
    batch_indices = torch.arange(batch_size, device=hidden_state.device)
    return hidden_state[batch_indices, last_indices]


def pool_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Dispatch to the requested pooling method
    """
    if pooling == "mean":
        return mean_pool(hidden_state, attention_mask)

    if pooling == "last_token":
        return last_token_pool(hidden_state, attention_mask)

    raise ValueError(f"Unsupported pooling method: {pooling}")


def pool_all_layers(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Pool all layers returned by a Hugging Face model

    Args:
        hidden_states: Tuple of tensors, one per layer, each with shape
            [batch_size, seq_len, hidden_dim]
        attention_mask: Tensor with shape [batch_size, seq_len]
        pooling: Pooling method name

    Returns:
        Tensor with shape [num_layers, batch_size, hidden_dim]
    """
    pooled_layers = []
    for layer_hidden in hidden_states:
        pooled = pool_hidden_state(layer_hidden, attention_mask, pooling)
        pooled_layers.append(pooled)

    return torch.stack(pooled_layers, dim=0)