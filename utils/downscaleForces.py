# imports
import torch
from . import get_atomic_descriptor
# import getAtomicWeights # use instead of "from . import getAtomicWeights" only if trying to debug running as main
import math
import warnings


def force2Angstroms(rawForces, structure, maxA=0.2):
    """
    Converts predicted forces to Angstroms and downscales them if necessary.

    Args:
        rawForces (Tensor or ndarray): Raw, unedited forces (e.g., from MLFF/MLIP or DFT).
        structure (Structure): The atomic structure from which forces were computed.
        maxA (float): Maximum allowed displacement in Angstroms for any atom.

    Returns:
        Tensor or ndarray: Linearly downscaled forces such that movement is no more than `maxA` Angstroms.
    """
    # ----- This is the code that just makes the max force 0.2A -----
    # row_norms = rawForces.norm(dim=1)                # Compute norm of each (1, 3) row
    # max_norm = row_norms.max()               # Maximum norm across rows
    # scale = min(1.0, maxA / max_norm.item())  # Scaling factor
    # return rawForces * scale                         # Return rescaled tensor

    if (not isinstance(rawForces, torch.Tensor)):
        warnings.warn("rawForces should be a torch.Tensor. Other datatypes may produce unexpected results")

    if (not isinstance(structure, torch.Tensor)):
        warnings.warn("structure should be a torch.Tensor. Other datatypes may produce unexpected results")

    assert rawForces.shape[:-1] == structure.shape[:-1], f"rawForces and structure must match in shape except the last dimension"

    mass = get_atomic_descriptor.get_atomic_weights(structure).to(rawForces.device)
    forceNorm = torch.norm(rawForces, dim=-1, keepdim=False)
    dispVec = forceNorm * (1/mass)
    # maxDisp = dispVec.max().item() # worse performance than using total displacement
    totalDisp = dispVec.norm().item()
    if (totalDisp <= maxA):
        scale = 1
    else:
        scale = (maxA/totalDisp)
    
    ind = 0
    scaleDi2 = (scale*dispVec[ind])**2
    s = (scaleDi2)/forceNorm[ind]
    s = math.sqrt(s)

    # max_val, max_ind = torch.max(dispVec, dim=0) # change torch.max to torch.min to see atom with smallest displacement. 
    # scaleDi2 = (scale*max_val)**2
    # s = (scaleDi2)/forceNorm[max_ind]
    # s = math.sqrt(s)
    return s*rawForces


def force2Angstroms_alternate(rawForces, structure, maxA=0.2):
    warnings.warn("This version produces less stable results if you are using the saved weights from the repo.")
    mass = get_atomic_descriptor.get_atomic_weights(structure).to(rawForces.device)
    #mass = torch.ones_like(mass).to(rawForces.device) # THIS MAKES THE PREDICTION HIGHLY UNSTABLE, THOUGH IT CAN SOMETIMES YIELD SLIGHTLY BETTER RESULTS
    forceNorm = torch.norm(rawForces, dim=-1, keepdim=False)
    dispVec = forceNorm * (1/mass)
    maxDisp = dispVec.max().item()
    # maxDisp = dispVec.norm().item() # we found that taking the norm over all of the atoms worked better. 
    if (maxDisp <= 0.2):
        scale = 1
    else:
        scale = (maxA/maxDisp)
    
    return scale*rawForces


def force2Angstroms_differentiable(rawForces, structure, maxA=0.2): 
    raise Exception("This is not the right version. Recreate for the new downscaler!")
    mass = addAtomicWeight.get_atomic_weights(structure).to(rawForces.device)  # (N_atoms,)
    forceNorm = torch.norm(rawForces, dim=-1)  # (N_atoms,)
    
    # Avoid division by zero (add eps)
    dispVec = forceNorm / (mass)  # (N_atoms,)
    
    # Differentiable max displacement (L2 norm of all displacements)
    maxDisp = torch.norm(dispVec)  # scalar tensor

    # Scale factor: maxA / maxDisp, clipped at 1 if already within limit
    scale = torch.clamp(maxA / (maxDisp), max=1.0)

    # Calculate per-atom scale (if needed)
    scaleDi2 = (scale * dispVec[0])**2
    s = torch.sqrt(scaleDi2 / (forceNorm[0] + 1e-10))
    
    return s * rawForces


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
    X_test = torch.load(baseAddress+"<fileName>.pt").clone().detach().to(device)

    # ---- create example force predictions ----
    rawForce = -5*torch.ones(217, 3)
    rawForce[3:5, :2] = 6.3
    rawForce[1, 2] = 7.9
    # --------
    structure = X_test[1]
    print("structure shape is: ", structure.shape)
    a = force2Angstroms(rawForce, structure)
    b = force2Angstroms_differentiable(rawForce, structure)
    print(a[:5])
    print()
    print(b[:5])
