""" Provides masking conversion and unification for use in the torch SDPA-based attentions in this directory. """

import torch
import warnings


def sanity_check_mask_convention(mask: torch.Tensor, expected_convention: str) -> None:
    """
    Check mask polarity and basic padded-token layout.
    
    SDPA convention (True=allow): Should generally have a high percentage of True values.
    MHA convention (True=block): Should generally have a low percentage of True values.

    For 2D padded-token masks, padding should be right-aligned. Therefore, the
    first entry in each row should be True for SDPA-style masks and False for
    MHA-style masks.
    """
    if mask is None:
        return

    if expected_convention not in {"sdpa", "mha"}:
        raise ValueError("expected_convention must be 'sdpa' or 'mha'.")
    if mask.dtype != torch.bool:
        raise TypeError("Mask sanity checks require a boolean tensor.")
    if mask.ndim < 2:
        raise ValueError(f"Mask must have at least two dimensions, got {tuple(mask.shape)}.")

    # Calculate the percentage of the mask that evaluates to True
    percent_true = (mask == True).float().mean().item() * 100

    print(f"[Sanity Check] Mask is {percent_true:.2f}% True.")

    if expected_convention == "sdpa" and percent_true < 20.0:
        warnings.warn("WARNING: You specified 'sdpa' (True=allow), but less than 10 percent of the input is allowed.")
        
    elif expected_convention == "mha" and percent_true > 80.0:
        warnings.warn("WARNING: You specified 'mha' (True=block), but over 90 percent of the input is blocked.")

    if mask.ndim == 2:
        first_entries = mask[:, 0]
        if expected_convention == "sdpa":
            bad_first_entries = ~first_entries
            expected_value = True
        else:
            bad_first_entries = first_entries
            expected_value = False

        if bad_first_entries.any().item():
            bad_indices = torch.nonzero(bad_first_entries, as_tuple=False).flatten().tolist()
            raise ValueError(
                "Mask rows must start with the expected unpadded-token polarity. "
                f"expected_convention={expected_convention!r}, "
                f"expected_first_value={expected_value}, "
                f"bad_batch_indices={bad_indices}."
            )
    else:
        warnings.warn(
            "Skipping first-entry mask polarity check for non-2D mask with "
            f"shape {tuple(mask.shape)}.",
            stacklevel=2,
        )


def build_valid_token_sdpa_mask(
    valid_tokens: torch.Tensor | None,
    batch_size: int,
    n_seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build an SDPA-style padding mask from valid-token polarity.

    valid_tokens:
        Shape (B, L), bool, True for real tokens and False for padding.

    returned:
        Shape (B, 1, 1, L), bool, True means the key token may be attended to.
    """
    if valid_tokens is None:
        return None

    if valid_tokens.dtype is not torch.bool:
        raise TypeError("valid_tokens must be bool.")

    if valid_tokens.shape != (batch_size, n_seq_len):
        raise ValueError(
            f"valid_tokens must have shape ({batch_size}, {n_seq_len}), "
            f"got {tuple(valid_tokens.shape)}."
        )

    if valid_tokens.device != device:
        valid_tokens = valid_tokens.to(device)

    return valid_tokens[:, None, None, :]


def build_flash_bool_mask(
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
    n_seq_len: int,
    device: torch.device,
    input_mask_convention: str,
) -> torch.Tensor | None:
    """
    Convert bool masks to SDPA-style masks for scaled_dot_product_attention.

    MHA bool convention:
        True = blocked / ignored

    SDPA bool convention:
        True = allowed / attend

    Returned shape is broadcastable to:
        (batch_size, num_heads, n_seq_len, n_seq_len)
    """
    if input_mask_convention not in {"mha", "sdpa"}:
        raise ValueError("input_mask_convention must be 'mha' or 'sdpa'.")
    elif input_mask_convention == "mha":
        if attn_mask is not None:
            attn_mask = ~attn_mask
        if key_padding_mask is not None:    
            key_padding_mask = ~key_padding_mask


    masks = []

    if attn_mask is not None:
        if attn_mask.dtype is not torch.bool:
            raise TypeError("attn_mask must be bool to preserve SDPA/FlashAttention paths.")

        if attn_mask.device != device:
            attn_mask = attn_mask.to(device)

        if attn_mask.shape[-2:] != (n_seq_len, n_seq_len):
            raise ValueError(
                f"attn_mask trailing dims must be ({n_seq_len}, {n_seq_len}), "
                f"got {tuple(attn_mask.shape)}."
            )            

        if attn_mask.dim() == 2:
            # (L, S) -> (1, 1, L, S)
            attn_mask = attn_mask[None, None, :, :]

        elif attn_mask.dim() == 3:
            if attn_mask.shape[0] == batch_size * num_heads:
                # (B*H, L, S) -> (B, H, L, S)
                attn_mask = attn_mask.view(batch_size, num_heads, n_seq_len, n_seq_len)
            elif attn_mask.shape[0] == batch_size:
                # (B, L, S) -> (B, 1, L, S)
                attn_mask = attn_mask[:, None, :, :]
            else:
                raise ValueError(
                    "3D attn_mask must have leading dim B or B*num_heads; "
                    f"got {attn_mask.shape[0]}, expected {batch_size} or "
                    f"{batch_size * num_heads}."
                )

        elif attn_mask.dim() == 4:
            if attn_mask.shape[:2] not in {
                (batch_size, num_heads),
                (batch_size, 1),
                (1, num_heads),
                (1, 1),
            }:
                raise ValueError(
                    "4D attn_mask must be broadcastable to "
                    f"({batch_size}, {num_heads}, {n_seq_len}, {n_seq_len}), "
                    f"got {tuple(attn_mask.shape)}."
                )
        else:
            raise ValueError("attn_mask must be 2D, 3D, or 4D.")

        masks.append(attn_mask)

    if key_padding_mask is not None:
        if key_padding_mask.dtype is not torch.bool:
            raise TypeError("key_padding_mask must be bool.")

        if key_padding_mask.shape != (batch_size, n_seq_len):
            raise ValueError(
                f"key_padding_mask must have shape ({batch_size}, {n_seq_len}), "
                f"got {tuple(key_padding_mask.shape)}."
            )

        if key_padding_mask.device != device:
            key_padding_mask = key_padding_mask.to(device)

        # (B, S) -> (B, 1, 1, S)
        valid_keys = (key_padding_mask)[:, None, None, :]
        masks.append(valid_keys)

    if not masks:
        return None

    merged = masks[0]
    for mask in masks[1:]:
        merged = merged & mask

    return merged
