"""Canonical attention registry."""

from __future__ import annotations

from torch import nn

from MLIP_arch.core_arch.attentions.allresid_attention import (
    MultiHeadedAttention as AllResidAttention,
)
from MLIP_arch.core_arch.attentions.mha_attention import (
    MultiHeadedAttention as StandardMultiHeadedAttention,
)
from MLIP_arch.core_arch.attentions.gqa_attention import (
    MultiHeadedAttention as GQAAttention,
)
from MLIP_arch.core_arch.attentions.mlv_attention import (
    MultiHeadedAttention as MLVAttention,
)
from MLIP_arch.core_arch.attentions.mlv_allresid_attention import (
    MultiHeadedAttention as MLVAllResidAttention,
)
from MLIP_arch.core_arch.attentions.nlv_attention import (
    MultiHeadedAttention as NLVAttention,
)
from MLIP_arch.core_arch.attentions.qek_attention import (
    SymmetricAttention as QEKAttention,
)


ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "standard_mha": StandardMultiHeadedAttention,
    "gqa_attention": GQAAttention,
    "nlv_attention": NLVAttention,
    "mlv_attention": MLVAttention,
    "mlv_allresid_attention": MLVAllResidAttention,
    "allresid_attention": AllResidAttention,
    "qek_attention": QEKAttention,
}

VALID_ATTENTION_TYPES = frozenset(ATTENTION_REGISTRY)
