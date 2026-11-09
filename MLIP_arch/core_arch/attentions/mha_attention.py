import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mask_utility


class MultiHeadedAttention(nn.Module):
    """
    Symmetric multi-head attention optimized for FlashAttention and matched to Glorot (Xavier) initialization scales.
    """

    def _reset_parameters(self) -> None:
        """Explicitly initialize weights using the standard Transformer initialization scheme."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)


    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = d_model // num_heads

        # Define projection layers (bias=False for Q to remain structurally uniform)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Apply Glorot (Xavier) Uniform Initialization to match nn.MultiheadAttention
        self._reset_parameters()


    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a ``(B, N, D)`` tensor into multi-head form ``(B, H, N, head_dim)``."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        x: torch.Tensor,
        input_mask_convention: str,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply symmetric FlashAttention with strict hardware execution path safety."""
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError("Input shape mismatch.")

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x) # Shape: (B, seq_len, 3 * d_model)

        # Chunk into Q and V. Shape of each: (B, seq_len, d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        # Padding-only SDPA masks use valid-token polarity: True means real token.
        if attn_mask is None and input_mask_convention == "sdpa":
            mask = mask_utility.build_valid_token_sdpa_mask(
                valid_tokens=key_padding_mask,
                batch_size=batch_size,
                n_seq_len=seq_len,
                device=x.device,
            )
        else:
            raise RuntimeError("Tried to use the full 4D attention mask!")

        if x.device.type == "cuda":
            backends = [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
        elif x.device.type == "cpu":
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            raise RuntimeError(
                f"Device type {x.device.type!r} is not supported by this attention module."
            )

        with torch.nn.attention.sdpa_kernel(backends):
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )

        # Context reconstruction
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
