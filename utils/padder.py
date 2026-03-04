# imports
import torch
import warnings

def padToDepth(tens, depth): # number of atoms padding
    if (not isinstance(tens, torch.Tensor)):
        warnings.warn("tens should be a torch.Tensor. Other datatypes may produce unexpected results")
    padDepth = depth - tens.shape[-2]

    if (padDepth < 0):
        raise Exception("Must have non-negative padding size")

    zer2_220 = torch.zeros((*tens.shape[:-2], padDepth, tens.shape[-1])).to(tens.device)
    return torch.cat((tens, zer2_220), dim=-2)


def padToWidth(tens, width): # feature vector padding
    if (not isinstance(tens, torch.Tensor)):
        warnings.warn("tens should be a torch.Tensor. Other datatypes may produce unexpected results")
    padWidth = width - tens.shape[-1]

    if (padWidth < 0):
        raise Exception("Must have non-negative padding size")

    zer2_13 = torch.zeros((*tens.shape[:-2], tens.shape[-2], padWidth)).to(tens.device)
    return torch.cat((tens, zer2_13), dim=-1)


def padTensor(vvd, d=220, w=13):
    """
    Pads vector-valued data to the specified dimensions.

    Args:
        vvd (Tensor): Vector-valued data of dimension at least 2.
        d (int): "Depth" dimension (dim=-2), maximum size.
        w (int): "Width" dimension (dim=-1), maximum size.

    Returns:
        Tensor: The padded vector-valued data.
    """
    return(padToWidth(padToDepth(vvd, d), w))
