# imports
import torch

def insert_tensor_as_second_last_channel(tensor_a, tensor_b):
    """
    Inserts a d-dimensional tensor (tensor_b) as the second-to-last channel into a (d+1)-dimensional tensor (tensor_a).
    
    Example:
        - (n, m, c) with (n, m)     → (n, m, c+1)
        - (n, c) with (n,)          → (n, c+1)
        - (n, m, p, q, c) with (n, m, p, q) → (n, m, p, q, c+1)

    Args:
        tensor_a (torch.Tensor): Input tensor of shape (..., c).
        tensor_b (torch.Tensor): Tensor of shape matching tensor_a.shape[:-1].

    Returns:
        torch.Tensor: A tensor with an extra channel inserted at index -2 (second-to-last channel).
    """
    assert isinstance(tensor_a, torch.Tensor), "tensor_a must be a torch.Tensor"
    assert isinstance(tensor_b, torch.Tensor), "tensor_b must be a torch.Tensor"
    assert tensor_a.device == tensor_b.device, "tensor_a and tensor_b must be on the same device"
    assert tensor_a.ndim == tensor_b.ndim + 1, (
        f"tensor_a must have one more dimension than tensor_b: "
        f"got tensor_a.ndim={tensor_a.ndim}, tensor_b.ndim={tensor_b.ndim}"
    )
    assert tensor_a.shape[:-1] == tensor_b.shape, (
        f"tensor_b shape {tensor_b.shape} must match tensor_a shape (except last channel): {tensor_a.shape[:-1]}"
    )

    c = tensor_a.shape[-1]
    insert_index = c - 1

    # Expand tensor_b to match dimensions for concatenation
    tensor_b_expanded = tensor_b.unsqueeze(-1)  # (..., 1)

    # Split tensor_a
    first_part = tensor_a[..., :insert_index]
    second_part = tensor_a[..., insert_index:]

    return torch.cat([first_part, tensor_b_expanded, second_part], dim=-1)


if __name__ == "__main__":
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    baseAddress = "./"
    X_base = torch.load(baseAddress+"<fileName>.pt").clone().detach().to(device)
    X_append = torch.load("<fileName>.pt").clone().detach().float().to(device)

    print("X_train.shape is: ", X_base.shape)

    weightsXTrain = insert_tensor_as_second_last_channel(X_base, X_append)

    print("X_train.shape is now: ", X_base.shape)
    print("Appended weights are: ", weightsXTrain.shape)

    torch.save(weightsXTrain.cpu(), "<fileName>.pt")
    print("Done!")
    