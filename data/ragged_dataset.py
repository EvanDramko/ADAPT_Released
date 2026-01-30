# ragged_dataset.py
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import warnings


class RaggedAtomDataset(Dataset):
    """
    Either pass lists directly, or a .pt path created by save_ragged().
    __getitem__ returns a single unpadded pair (X_i, Y_i) with shapes:
      X_i: (n_i, 13), Y_i: (n_i, 3)
    """
    def __init__(
        self,
        X_list: Optional[List[torch.Tensor]] = None,
        Y_list: Optional[List[torch.Tensor]] = None,
        from_path: Optional[str] = None,
        max_len: Optional[int] = 220,
    ):
        if from_path is not None:
            assert X_list is None and Y_list is None, "cannot pass both filepath to dataset and dataset values"
            fileData: Dict[str, Any] = torch.load(from_path, map_location='cpu')
            print(fileData.keys())
            X_list = fileData["X"]
            Y_list = fileData["Y"]

        assert X_list is not None and Y_list is not None and len(X_list) == len(Y_list)
        self.max_len = max_len
        # assert X_list.shape[-2] <= self.max_len # X_list being a ragged list rather than an indexable tensor prevents this efficiently
        self.X_list = X_list
        self.Y_list = Y_list

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_list[idx], self.Y_list[idx]


def collate_pad(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
):
    """
    Pads a batch of variable-length (X_i, Y_i) pairs into:
      X: (B, L, 13)
      Y: (B, L, 3)
      mask: (B, L)  True for real tokens, False for padding
    - If max_len is provided, sequences longer than max_len are TRUNCATED.
    - Otherwise L = max(n_i) in the batch.
    """
    Xs, Ys = zip(*batch)  # tuples of tensors (n_i, vecRep) and (n_i, 3)

    if max_len is not None:
        # check if any sequence is longer than max_len
        truncated_X = any(len(x) > max_len for x in Xs)
        truncated_Y = any(len(y) > max_len for y in Ys)

        if truncated_X or truncated_Y:
            warnings.warn(
                f"Truncating sequences longer than max_len={max_len}. "
                f"Xs truncated: {truncated_X}, Ys truncated: {truncated_Y}"
            )
        # truncate before padding if any sequence is too long
        Xs = tuple(x[:max_len] for x in Xs)
        Ys = tuple(y[:max_len] for y in Ys)

    lengths = torch.tensor([x.shape[0] for x in Xs], dtype=torch.long)  # (B,)
    L = max_len if max_len is not None else int(lengths.max().item())

    # Use pad_sequence: expects list of (seq_len, d) tensors
    # batch_first=True → (B, L, d)
    X = pad_sequence(Xs, batch_first=True, padding_value=pad_value)  # (B, L, vecRep)
    Y = pad_sequence(Ys, batch_first=True, padding_value=pad_value)  # (B, L, 3)

    # If some sequences < L (because max_len was set larger), pad_sequence already padded.
    # If some sequences == L and others shorter, also fine.

    # Build mask: True on real tokens, False on pad
    B = len(Xs)
    mask = torch.zeros((B, L), dtype=torch.bool)
    for i, n_i in enumerate(lengths.tolist()):
        mask[i, :min(n_i, L)] = True

    # If pad_sequence produced longer than L (shouldn't happen unless inputs exceed),
    # clamp to L to be safe:
    X = X[:, :L]
    Y = Y[:, :L]
    mask = mask[:, :L]

    return X, Y, mask
