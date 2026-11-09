"""Single-value decoder energy head."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from MLIP_arch.activations import build_activation
from MLIP_arch.core_arch.encoder_backbone import EncoderBackbone


AttentionBuilder = Callable[[int, int, float], nn.Module]


class ScalarDecoderEnergyHead(nn.Module):
    """Cross-attend one learned query to encoded tokens and predict one energy."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float,
        activation_name: str,
        num_head_layers: int,
        attention_builder: AttentionBuilder,
    ) -> None:
        super().__init__()
        if num_head_layers < 0:
            raise ValueError(f"num_head_layers must be non-negative, got {num_head_layers}.")
        self.head_encoder = None
        if num_head_layers > 0:
            self.head_encoder = EncoderBackbone(
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_head_layers,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_builder=attention_builder,
                activation_name=activation_name,
            )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = build_activation(activation_name)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        atom_count: int | None = None,
    ) -> torch.Tensor:
        """Return one scalar energy per structure with shape ``(batch,)``.

        ``tokens`` may include atom and structure-level tokens. ``atom_count``
        is accepted for API consistency with atomwise heads.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (batch, seq_len, d_model), got {tuple(tokens.shape)}.")

        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"valid_mask must have shape {tuple(tokens.shape[:2])}, got {tuple(valid_mask.shape)}."
                )
            key_padding_mask = ~valid_mask

        if self.head_encoder is not None:
            tokens = self.head_encoder(tokens, valid_mask=valid_mask)

        batch_size = tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.cross_attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = self.norm1(query + self.dropout1(attn_output))
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout2(ff_output))
        return self.out(x).squeeze(-1).squeeze(-1)
