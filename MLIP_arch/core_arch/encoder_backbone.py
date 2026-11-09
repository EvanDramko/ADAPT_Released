"""Reusable Transformer encoder backbone for embedded structure tokens."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from MLIP_arch.activations import build_activation


AttentionBuilder = Callable[[int, int, float], nn.Module] 


def apply_valid_mask(tokens: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero invalid/padded token positions while preserving tensor shape."""
    if valid_mask is None:
        return tokens
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean tensor.")
    if valid_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"valid_mask shape must match token batch/sequence shape {tuple(tokens.shape[:2])}, "
            f"got {tuple(valid_mask.shape)}."
        )
    return torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class TransformerEncoderBlock(nn.Module):
    """One pre-norm style self-attention/feed-forward encoder block."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float,
        attention_builder: AttentionBuilder,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = attention_builder(d_model, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = build_activation(activation_name)
        self.dropout_ff = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokens with self-attention and a position-wise feed-forward layer."""
        key_padding_mask = None if valid_mask is None else valid_mask

        attn_input = self.norm1(tokens)
        attn_output = self.self_attn(
            attn_input,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            input_mask_convention="sdpa",
        )
        tokens = tokens + self.dropout1(attn_output)
        tokens = apply_valid_mask(tokens, valid_mask)

        ff_input = self.norm2(tokens)
        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(ff_input))))
        tokens = tokens + self.dropout2(ff_output)
        return apply_valid_mask(tokens, valid_mask)


class EncoderBackbone(nn.Module):
    """Stack of Transformer encoder blocks for already-embedded tokens.

    This module does not know about atom coordinates, atomic numbers, charge,
    or prediction heads. It only maps embedded tokens to encoded tokens.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        num_layers: int,
        layer_repetitions: int = 1,
        num_heads: int,
        dropout_rate: float,
        attention_builder: AttentionBuilder,
        activation_name: str,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")
        if layer_repetitions < 1:
            raise ValueError(
                f"layer_repetitions must be at least 1, got {layer_repetitions}."
            )

        self.layer_repetitions = layer_repetitions

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_builder=attention_builder,
                    activation_name=activation_name,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoded tokens with invalid positions zeroed out."""
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape (batch, seq_len, d_model), got {tuple(tokens.shape)}."
            )

        output = apply_valid_mask(tokens, valid_mask)
        for layer in self.layers:
            for _ in range(self.layer_repetitions):
                output = layer(output, attn_mask=attn_mask, valid_mask=valid_mask)

        output = self.norm(output)
        return apply_valid_mask(output, valid_mask) # need zeroing in case the norm re-introduces non-zero values. 
