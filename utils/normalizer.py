# imports
import torch
from typing import List, Tuple, Dict, Any

RESCALE_FACTOR = 0.3 # constant to spread out the distribution of the data
eps=1e-6

def normalize_batch_padded(X_pad, mask, x_mean, x_std):
    """
    Normalize X except the last feature, only where mask==True.
    Pads remain 0.
    """
    device = X_pad.device
    x_mean = x_mean.to(device, dtype=X_pad.dtype).detach()
    x_std  = x_std.to(device,  dtype=X_pad.dtype).detach()

    Xn = torch.zeros_like(X_pad)                 # keep pads at 0
    m  = mask.unsqueeze(-1)                      # [B, L, 1]

    # Features to normalize (all but last)
    feat = X_pad[..., :-1]                       # [B, L, 12]
    mean = x_mean[:-1].view(1,1,-1)              # [1,1,12]
    std  =  x_std[:-1].view(1,1,-1)              # [1,1,12]

    # Only touch valid tokens
    Xn_feat = (feat - mean) / (RESCALE_FACTOR * (std + eps))
    Xn[..., :-1] = torch.where(m, Xn_feat, torch.zeros_like(Xn_feat))

    # Copy the last column (weighting) through untouched for valid tokens; keep pads at 0
    last = X_pad[..., -1:]
    Xn[..., -1:] = torch.where(m, last, torch.zeros_like(last))
    return Xn

# def unnormalize_batch_paddedX(Xn_pad, mask, x_mean, x_std, RESCALE=1.0, eps=1e-8):
#     """
#     Undo normalization applied in normalize_batch_padded().
#     Restores features back to original scale for valid tokens; pads remain 0.
#     Args mirror normalize_batch_padded and MUST match for reversibility.
#     """
#     device = Xn_pad.device
#     x_mean = x_mean.to(device, dtype=Xn_pad.dtype)
#     x_std  = x_std.to(device,  dtype=Xn_pad.dtype)

#     X = torch.zeros_like(Xn_pad)                  # keep pads at 0
#     m = mask.unsqueeze(-1)                        # [B, L, 1]

#     # Undo normalization on all but last feature
#     feat_n = Xn_pad[..., :-1]                     # [B, L, F-1]
#     mean   = x_mean[:-1].view(1,1,-1)             # [1,1,F-1]
#     std    = x_std[:-1].view(1,1,-1)              # [1,1,F-1]

#     feat = feat_n * (RESCALE * (std + eps)) + mean
#     X[..., :-1] = torch.where(m, feat, torch.zeros_like(feat))

#     # Pass through last column for valid tokens; keep pads at 0
#     last = Xn_pad[..., -1:]
#     X[..., -1:] = torch.where(m, last, torch.zeros_like(last))
#     return X


def unnormalize_batch_padded(Y_hat_norm, mask, y_mean, y_std):
    """
    Unnormalize Y only where mask==True. Pads stay 0.
    """
    device = Y_hat_norm.device
    y_mean = y_mean.to(device, dtype=Y_hat_norm.dtype).view(1,1,-1)
    y_std  = y_std.to(device,  dtype=Y_hat_norm.dtype).view(1,1,-1)
    m = mask.unsqueeze(-1)

    Y_hat = torch.zeros_like(Y_hat_norm)
    Y_hat_valid = Y_hat_norm * (RESCALE_FACTOR * (y_std + eps)) + y_mean
    Y_hat = torch.where(m, Y_hat_valid, torch.zeros_like(Y_hat))
    return Y_hat

def normalize_all(XY, tens, mean, std, eps=1e-6):
    """
    Normalizes all values in the input tensor using the given mean and standard deviation.

    Args:
        XY (int): DEPRICATED - `0` for X, `1` for Y. Used to select the saved mean/std if not provided.
        tens (torch.Tensor): Tensor to normalize. Operation is not performed in-place.
        mean (torch.Tensor, optional): Mean value to use for each feature channel. Defaults to a saved value based on `XY`.
        std (torch.Tensor, optional): Standard deviation to use for each feature channel. Defaults to a saved value based on `XY`.
        eps (float): Small value added for numerical stability in division.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if mean is None or std is None:
        raise Exception("In-utils loading is discontinued.")

    assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
    assert isinstance(mean, torch.Tensor), "mean must be a torch.Tensor"
    assert isinstance(std, torch.Tensor), "std must be a torch.Tensor"

    if((mean.ndim > 1) and (tens.ndim > 1)):
        assert mean.shape[-1] == tens.shape[-1], f"tens and mean must have the same number of feature channels"
        assert std.shape[-1] == tens.shape[-1], f"tens and std must have the same number of feature channels"

    # ensure devices and data types match
    mean = mean.to(device=tens.device, dtype=tens.dtype)
    std  = std.to(device=tens.device, dtype=tens.dtype)

    # broadcast mean, std over leading dimensions
    tensor_to_norm = tens.clone()
    mean = mean.view(*([1] * (tensor_to_norm.ndim - 1)), -1)
    std  = std.view(*([1] * (tensor_to_norm.ndim - 1)), -1)
    # Normalize using saved statistics
    tensor_normed = (tensor_to_norm - mean) / (RESCALE_FACTOR*(std + eps))
    return tensor_normed


def normalize_except_last(XY, tens, mean=None, std=None, eps=1e-6):
    """
    Normalizes all values in the input tensor except the last feature component using the given mean and standard deviation.

    Args:
        XY (int): DEPRICATED - `0` for X, `1` for Y. Used to select the saved mean/std if not provided.
        tens (torch.Tensor): Tensor to normalize. Operation is not performed in-place.
        mean (torch.Tensor, optional): Mean value to use for each feature channel. Defaults to a saved value based on `XY`.
        std (torch.Tensor, optional): Standard deviation to use for each feature channel. Defaults to a saved value based on `XY`.
        eps (float): Small value added for numerical stability in division.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    
    tensor = tens.clone() # do not destroy original tensor with in-place ops
    # Split features
    tensor_to_norm = tensor[..., :-1]
    tensor_weighting_val = tensor[..., -1:]

    tensor_normed = normalize_all(XY, tensor_to_norm, mean[..., :-1], std[..., :-1], eps) # outsource to main normalization code

    # Concatenate back the unchanged feature
    tensor_final = torch.cat([tensor_normed, tensor_weighting_val], dim=-1)
    return tensor_final

def unnormalize_all(XY, tens, mean=None, std=None, eps=1e-6):
    """
    Un-normalizes all values in the input tensor using the given mean and standard deviation.

    Args:
        XY (int): DEPRICATED - `0` for X, `1` for Y. Used to select the saved mean/std if not provided.
        tens (torch.Tensor): Tensor to un-normalize. Operation is not performed in-place.
        mean (torch.Tensor, optional): Mean value to use for each feature channel. Defaults to a saved value based on `XY`.
        std (torch.Tensor, optional): Standard deviation to use for each feature channel. Defaults to a saved value based on `XY`.
        eps (float): Small value added for numerical stability in division.

    Returns:
        torch.Tensor: Un-normalized tensor.
    """

    if mean is None or std is None:
        raise Exception("In-utils loading is discontinued.")
        # if XY == 0:
        #     mean = torch.load('./utils/meanX.pt').to(tens.device)
        #     std = torch.load('./utils/stdX.pt').to(tens.device)
        # else:
        #     mean = torch.load('./utils/forceMeanY.pt').to(tens.device)
        #     std = torch.load('./utils/forceStdY.pt').to(tens.device)

    assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
    assert isinstance(mean, torch.Tensor), "mean must be a torch.Tensor"
    assert isinstance(std, torch.Tensor), "std must be a torch.Tensor"
    assert mean.shape[-1] == tens.shape[-1], f"tens and mean must have the same number of feature channels"
    assert std.shape[-1] == tens.shape[-1], f"tens and std must have the same number of feature channels"

    # ensure devices and data types match
    mean = mean.to(device=tens.device, dtype=tens.dtype)
    std  = std.to(device=tens.device, dtype=tens.dtype)

    # broadcast mean, std over leading dimensions
    tensor_to_unnorm = tens.clone()
    mean = mean.view(*([1] * (tensor_to_unnorm.ndim - 1)), -1)
    std  = std.view(*([1] * (tensor_to_unnorm.ndim - 1)), -1)
    
    # Unnormalize using saved statistics
    tensor_unnormed = (tensor_to_unnorm * (RESCALE_FACTOR* (std + eps))) + mean
    return tensor_unnormed

def unnormalize_except_last(XY, tens, mean=None, std=None, eps=1e-6):
    """
    Un-normalizes all values in the input tensor except the last feature channel using the given mean and standard deviation.

    Args:
        XY (int): DEPRICATED - `0` for X, `1` for Y. Used to select the saved mean/std if not provided.
        tens (torch.Tensor): Tensor to normalize. Operation is not performed in-place.
        mean (torch.Tensor, optional): Mean value to use for each feature channel. Defaults to a saved value based on `XY`.
        std (torch.Tensor, optional): Standard deviation to use for each feature channel. Defaults to a saved value based on `XY`.
        eps (float): Small value added for numerical stability in division.

    Returns:
        torch.Tensor: Un-normalized tensor.
    """
    tensor = tens.clone()  # do not modify original tensor

    # Split features
    tensor_to_unnorm = tensor[..., :-1]
    tensor_weighting_val = tensor[..., -1:]

    # Unnormalize all but last using main unnormalization code
    tensor_unnormed = unnormalize_all(XY, tensor_to_unnorm, mean[..., :-1], std[..., :-1], eps)

    # Concatenate back the unchanged last feature
    tensor_final = torch.cat([tensor_unnormed, tensor_weighting_val], dim=-1)
    return tensor_final


# ================= UTILITIES FOR DEALING WITH RAGGED DATASETS =================
# ---------- ragged helpers ----------
def lengths_of(ragged: List[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([t.shape[0] for t in ragged], dtype=torch.long)

def concat_ragged(ragged: List[torch.Tensor]) -> torch.Tensor:
    if not ragged:
        raise ValueError("Empty ragged list")
    return torch.cat(ragged, dim=0)  # (sum n_i, d)

def split_back(stacked: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
    outs, s = [], 0
    for n in lengths.tolist():
        outs.append(stacked[s:s+n])
        s += n
    return outs

# ---------- fit stats on TRAIN only ----------
@torch.no_grad()
def fit_stats_from_ragged(X_list: List[torch.Tensor], Y_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    X_stack = concat_ragged(X_list)  # (N_x, d_x=13)
    Y_stack = concat_ragged(Y_list)  # (N_y, d_y=3)
    x_mean, x_std = X_stack.mean(0), X_stack.std(0, unbiased=False).clamp_min(1e-8)
    y_mean, y_std = Y_stack.mean(0), Y_stack.std(0, unbiased=False).clamp_min(1e-8)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean_force": y_mean, "y_std_force": y_std}

# ---------- apply your utils.normalizer.normalize_all ----------
@torch.no_grad()
def normalize_ragged_with_stats(
    X_list: List[torch.Tensor],
    Y_list: List[torch.Tensor],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
):
    # keep ragged structure by normalizing a concatenated view then splitting back
    X_len = lengths_of(X_list)
    Y_len = lengths_of(Y_list)

    X_stack = concat_ragged(X_list)  # (n, 13)
    Y_stack = concat_ragged(Y_list)  # (n, 3)

    Xn_stack = normalize_except_last(0, X_stack, mean=x_mean, std=x_std)
    Yn_stack = normalize_all(1, Y_stack, mean=y_mean, std=y_std)

    Xn_list = split_back(Xn_stack, X_len)
    Yn_list = split_back(Yn_stack, Y_len)
    return Xn_list, Yn_list


