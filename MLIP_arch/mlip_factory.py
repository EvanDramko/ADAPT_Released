"""Factory for assembling the component-based MLIP architecture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from configs.arch_config import ArchitectureConfig, VALID_HEAD_TYPES
from MLIP_arch.core_arch.attentions.registry import ATTENTION_REGISTRY
from MLIP_arch.embeddings import StructureEmbedding
from MLIP_arch.core_arch.encoder_backbone import EncoderBackbone
from MLIP_arch.model import MLIPModel
from MLIP_arch.output_heads import build_output_head


AttentionBuilder = Callable[[int, int, float], nn.Module]


def build_attention_builder(attention_type: str) -> AttentionBuilder:
    """Return an attention-builder callable from a registered attention name."""
    cls = ATTENTION_REGISTRY.get(attention_type)
    if cls is None:
        raise ValueError(
            f"Unknown attention '{attention_type}'. Available attentions: {sorted(ATTENTION_REGISTRY)}."
        )
    return cls


def build_mlip_model(config: ArchitectureConfig) -> MLIPModel:
    """Build one configured component-based MLIP model."""
    encoder_cfg = config.encoder
    embedding_cfg = config.embedding
    attention_builder = build_attention_builder(encoder_cfg.attention)

    structure_embedding = StructureEmbedding(
        coordinate_embedding=embedding_cfg.coordinate_embedding,
        species_embedding=embedding_cfg.species_embedding,
        charge_embedding=embedding_cfg.charge_embedding,
        spin_multiplicity_embedding=embedding_cfg.spin_multiplicity_embedding,
        d_model=encoder_cfg.d_model,
        hidden_dim=encoder_cfg.d_ff,
        activation_name=encoder_cfg.activation,
        dropout_rate=encoder_cfg.dropout_rate,
        max_atomic_number=embedding_cfg.max_atomic_number,
    )

    encoder_backbone = EncoderBackbone(
        d_model=encoder_cfg.d_model,
        d_ff=encoder_cfg.d_ff,
        num_layers=encoder_cfg.num_layers,
        layer_repetitions=encoder_cfg.layer_repetitions,
        num_heads=encoder_cfg.num_heads,
        dropout_rate=encoder_cfg.dropout_rate,
        attention_builder=attention_builder,
        activation_name=encoder_cfg.activation,
    )

    output_heads = nn.ModuleDict(
        {
            VALID_HEAD_TYPES[head_type]: _build_head_from_config(
                VALID_HEAD_TYPES[head_type],
                head_type,
                d_model=encoder_cfg.d_model,
                d_ff=encoder_cfg.d_ff,
                num_heads=encoder_cfg.num_heads,
                num_head_layers=encoder_cfg.head_num_layers,
                dropout_rate=encoder_cfg.dropout_rate,
                activation_name=encoder_cfg.activation,
                attention_builder=attention_builder,
            )
            for head_type in config.heads
        }
    )

    return MLIPModel(
        structure_embedding=structure_embedding,
        encoder_backbone=encoder_backbone,
        output_heads=output_heads,
    )


def _build_head_from_config(
    output: str,
    head_type: str,
    *,
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_head_layers: int,
    dropout_rate: float,
    activation_name: str,
    attention_builder: AttentionBuilder,
) -> nn.Module:
    kwargs: dict[str, Any] = {
        "d_model": d_model,
        "d_ff": d_ff,
        "activation_name": activation_name,
        "num_head_layers": num_head_layers,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "attention_builder": attention_builder,
    }
    if output == "energy" and head_type == "scalar_decoder_energy":
        kwargs.update(
            {
                "num_heads": num_heads,
                "dropout_rate": dropout_rate,
            }
        )
    return build_output_head(
        output,
        head_type,
        **kwargs,
    )
