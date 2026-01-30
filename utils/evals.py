# This contains all the ``scoring'' code for the results. It has both loss functions and eval metrics. 

# imports
import torch
import torch.nn.functional as F
import numpy as np
# from . import cifSaver, delta_Q
import warnings

@torch.no_grad()
def dq_scores(candidate_weighted, actual, mask, safety_eps: float = 1e-8): 
    """
    Returns (B,) tensor with 1/(DQ+eps). No grad by design.
    """
    warnings.warn("Depricated: A newer version of this allowing gradient tracking and efficient computations is available in dq_tens.py")
    candidate = candidate_weighted[..., :-1]
    B = candidate.shape[0]
    out = candidate.new_full((B,), -1.0)
    for i in range(B):
        pred = candidate[i][mask[i]]
        actl = actual[i][mask[i]]
        assert pred.shape == actl.shape, f"pred {pred.shape} vs actual {actl.shape}"
        pred_struct = cifSaver.getStruct(pred.detach().cpu())
        actl_struct = cifSaver.getStruct(actl.detach().cpu())
        dq, _ = delta_Q.interpolate_Q(pred_struct, actl_struct)
        out[i] = 1.0 / (dq + safety_eps)
    assert torch.all(out != -1), "DQ computation failed for some samples."
    return out


def weightedMSE(pred, y, lossPattern):
    """
    MSE loss where each component is weighted by the corresponding element of lossPattern

    Args:
        pred (torch.Tensor): prediction tensor of shape (b0, ..., bi, n, k)
        y (torch.Tensor): actual value tensor of shape (b0, ..., bi, n, k) 
        lossPattern (torch.Tensor): importance weighting tensor of shape (b0, ..., bi, n, 1)

    Returns:
        float: weighted MSE loss
    """
    assert pred.shape == y.shape, f"predicted and actual vector must have the same dimensions"

    distances = torch.norm(pred-y, p=2, dim=-1)
    values = distances * lossPattern
    return torch.sum(values)


def angMagCompare(pred, y): 
    # ---- ANGULAR COMPONENT ----
    # predMag = torch.norm(pred, p="fro", dim=2)
    # yMag = torch.norm(y, p="fro", dim=2)
    # dotProd = torch.einsum('bij,bij->bi', pred, y)
    # magsMult = 1/((predMag*yMag) + 1e-15)
    # ang = 1 - torch.mul(dotProd, magsMult) 
    ang = F.cosine_similarity(pred, y, dim=-1, eps=1e-8)
    eps = 1e-6
    ang = torch.clamp(ang, -1.0 + eps, 1.0 - eps)
    ang = torch.arccos(ang) * (180/np.pi)
    avgAng = ang.mean()


    # ---- MAGNITUDE CALCULATIONS ----
    predMag = torch.norm(pred, p="fro", dim=2)
    yMag = torch.norm(y, p="fro", dim=2)
    mag = torch.abs(predMag - yMag) 
    avgMag = mag.mean()
    return (avgAng, avgMag)


def angVsMagTest(dataloader, model, device, numSelected=None):
    num_batches = len(dataloader)
    model.eval()
    test_loss = [0, 0]
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            Xdata = X[:, :, :-1]
            pattern = X[:, :, -1]  # pull the loss weighting pattern out of the model

            # Compute prediction error
            key_padding_mask = (Xdata[:, :, 3:] == 0).all(dim=-1)
            pred = model(Xdata, src_key_padding_mask=key_padding_mask)
            # ---- calculate loss ----
            if (numSelected == None):
                loss = angMagCompare(pred, y)
            else:
                norms = y.norm(p=2, dim=1)   # shape: (n,)
                topCount = 30
                top_vals, top_idx = norms.topk(topCount, largest=True)
                loss = angMagCompare(torch.reshape(y[top_idx], (1, topCount, 3)), torch.reshape(pred[top_idx], (1, topCount, 3)))
            # ------------------------

            test_loss[0], test_loss[1] = test_loss[0] + loss[0], test_loss[1] + loss[1]

    angle, magnitude = test_loss[0]/num_batches, test_loss[1]/num_batches
    print("Angle is: ", angle)
    print("Magnitude is: ", magnitude)


def vecAlignLoss5p(pred, y, lossPattern=None, weightingFactor = 0.5): 
    # balancing factor: smallest number of leading zeros.
        # Used to ensure numerical stability in angular calculations
        # Used to bring the magnitude component to equal order as the angular component
    # lengths = pred.shape
    # balFactor = torch.nan_to_num(pred.abs().log10().floor(), neginf=1)
    # rescaleExps = -1*torch.amax(balFactor, dim=2).view(lengths[0], lengths[1], 1)
    # rescaleVals = torch.pow(10, rescaleExps)
    # pred = pred*rescaleVals
    # y = y*rescaleVals

    # ---- ANGULAR COMPONENT ----
    predMag = torch.norm(pred, p="fro", dim=2)
    yMag = torch.norm(y, p="fro", dim=2)
    dotProd = torch.einsum('bij,bij->bi', pred, y)
    magsMult = 1/((predMag*yMag) + 1e-15)
    ang = 1 - torch.mul(dotProd, magsMult) 

    # ---- MAGNITUDE CALCULATIONS ----
    mag = torch.abs(predMag - yMag) 

    # ---- COMBINE COMPONENTS ----
    a = ang + weightingFactor*mag # a is a temp variable, the name has no meaning
    a = a * lossPattern
    b = torch.sum(a)
    c = F.mse_loss(pred, y)
    return c + b


def defectValScore(X, y , importance):
    defectError = 0
    defectCount = 0
    ring3error = 0
    ring3count = 0

    distances = torch.norm((X-y), p=2, dim=-1)/(1e-15+torch.norm(y, p=2))
    ring1elems = torch.where(importance > 1.1, distances, 0)
    defectError += torch.sum(ring1elems).item()
    defectCount += torch.count_nonzero(ring1elems).item()
    # set these importance elements to the flag value of -1 so they are ignored
    importance = torch.where(importance > 1, -1, importance)

    # for every remaining vector, check if it is in ring2
    ring3elems = torch.where(importance > 0, distances, 0)
    ring3error += torch.sum(ring3elems).item()
    ring3count += torch.count_nonzero(ring3elems).item()
    
    return [defectError, defectCount, ring3error, ring3count]


def ringsValScore(X, y, importance): # note that y is a matrix of size 217x3, but we use lowercase y to match convention
    # define importance cutoffs for the "rings" of importance
    ring1boundary = 25
    ring2boundary = 5
    ring3boundary = 0
    
    # define running tallies for error and count in each ring
    # use running tallies in case we need in-routine batching becuase of memory constraints
    ring1error = 0
    ring2error = 0
    ring3error = 0

    ring1count = 0
    ring2count = 0
    ring3count = 0

    # calculate L2 distance between each vector in X, y
    distances = torch.norm((X-y), p=2, dim=-1)/(1e-15+torch.norm(y, p=2))

    # for every vector, check if it is in ring1
    ring1elems = torch.where(importance > ring1boundary, distances, 0)
    ring1error += torch.sum(ring1elems).item()
    ring1count += torch.count_nonzero(ring1elems).item()
    # set these importance elements to the flag value of -1 so they are ignored
    importance = torch.where(importance > ring1boundary, -1, importance)

    # for every remaining vector, check if it is in ring2
    ring2elems = torch.where(importance > ring2boundary, distances, 0)
    ring2error += torch.sum(ring2elems).item()
    ring2count += torch.count_nonzero(ring2elems).item()
    # set these elements to the flag value of -1 so they are ignored
    importance = torch.where(importance > ring2boundary, -1, importance)

    # for every remaining vector, check if it is in ring2
    ring3elems = torch.where(importance > ring3boundary, distances, 0)
    ring3error += torch.sum(ring3elems).item()
    ring3count += torch.count_nonzero(ring3elems).item()
    # set these elements to the flag value of -1 so they are ignored
    # importance = torch.where(importance > ring3boundary, -1, importance) --> unnecessary for the last ring which includes everything
    return [ring1count, ring1error, ring2count, ring2error, ring3count, ring3error]


def defectVsSiMAE(X, y, mask):
    """
    Args: 
    X: predicted forces in shape nx3
    y: actual forces in shape nx3
    mask: mask of length n showing where the non-Si atoms are located

    Return:
    (avg MAE of non-Si, avg MAE of Si)
    """
    distances = torch.abs(X-y)
    defLocs = distances[mask] # shape (m,1), where m = mask.sum()
    defVals = defLocs.mean() # compute the average (a scalar)

    siLocs = distances[torch.logical_not(mask)] # shape (m,1), where m = mask.sum()
    siVals = siLocs.mean() # compute the average (a scalar)

    return (defVals, siVals)


def maskOutElement(X, row=3, col=14):
    """
    Creates a mask showing the location of all except a certain type of atom. Defaults to masking away Si. 
    Args:
    x: input tensor of shape (B, n, 12) or (B, n, 13)
    row: row value of the element
    col: column value of the element

    Returns: A bool mask of shape (B, n, 1) that is False wherever a given type of atom occurs. 
    """
    mask = (X[..., 3] != col) | (X[..., 4] != row) # mask can be applied directly to data as X[mask]
    return mask


def MSEsoftDot(pred, y, pattern=None, alpha=1):
    """
    Args:
    pred: tensor giving model predictions
    y: tensor giving actual values
    pattern: NOT IMPLEMENTED YET

    Return:
    y * softmax(alpha*y)
    """
    if(pattern!=None):
        raise Exception("MSEsoftDot does not allow for a loss pattern weighting. Uncommenting is necessary.")
    
    distances = torch.norm(pred-y, p=2, dim=-1) # * pattern # this allows for the inclusion of a loss pattern in the "distances" calculation
    softScore = torch.nn.functional.softmax(alpha * distances, dim=-1)
    lossVec = distances*softScore
    return torch.sum(lossVec)


def topk_mse(pred, targ, k_ratio: float = 0.1): # From Tasos
    """
    Average of the largest `k_ratio` fraction of squared L2 errors.

    Parameters
    ----------
    pred, targ : (B, N, 3)  predicted / true forces
    k_ratio    : float in (0,1], e.g. 0.1 keeps top-10 % atoms

    Returns
    -------
    scalar tensor
    """
    with torch.no_grad():
        B, N, _ = pred.shape
        k = max(1, int(k_ratio * N))

    err = torch.norm(pred - targ, dim=-1) ** 2      # (B,N)
    topk, _ = torch.topk(err, k=k, dim=-1)          # largest k per sample
    return topk.mean()    


if __name__ == "__main__":
    # testing code to demonstrate functions
    x = torch.tensor([[[0, 1, 0], [1, 31, 3]], [[1, 2, 4], [1, 2, 3]]]).float()
    y = torch.tensor([[[0, 0, 23], [1, 1, 3]], [[1, 2, 4], [1, 2, 3]]]).float()
    res = topk_mse(x, y)
    print(res)
