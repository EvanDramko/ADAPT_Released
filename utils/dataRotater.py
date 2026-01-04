# imports
import torch
import numpy as np
import random

def rotateData_about_center(data, rotate):
    """
    data:  (B, n, D) tensor, first 3 channels are xyz
    rotate: (3, 3) rotation matrix
    """
    steps = data.clone().detach()

    # Extract coordinates
    coords = steps[:, :, :3]                     # (B, n, 3)

    # Center of geometry (equal masses)
    center = coords.mean(dim=1, keepdim=True)    # (B, 1, 3)

    # Shift to center, rotate, shift back
    centered = coords - center                   # (B, n, 3)
    rotated  = torch.einsum('ij,...j->...i', rotate, centered)
    steps[:, :, :3] = rotated + center

    return steps


def rotateData_about_origin(data, rotate):
    # create a deep copy of data
    steps = data.clone().detach()

    # apply the rotation matrix to coordinates in place
    steps[:, :, :3] = torch.einsum('ij,...j->...i', rotate, steps[:, :, :3])

    return steps

def rotateX(degrees):
    angle = np.radians(degrees)
    cos = np.cos(angle).item()
    sin = np.sin(angle).item()
    r = torch.tensor([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]).float()
    return r

def rotateZ(degrees):
    angle = np.radians(degrees)
    cos = np.cos(angle).item()
    sin = np.sin(angle).item()
    r = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]).float()
    return r

def rotateY(degrees):
    angle = np.radians(degrees)
    cos = np.cos(angle).item()
    sin = np.sin(angle).item()
    r = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]).float()
    return r

def randomRotate(*tensors: torch.Tensor):
    assert len(tensors) > 0, "Pass at least one tensor"
    assert all(isinstance(t, torch.Tensor) for t in tensors), "All inputs must be tensors"

    # One shared random rotation
    # rigidMorph = rotateX(random.random()) @ rotateY(random.random()) @ rotateZ(random.random())
    max_angle = 360
    rigidMorph = (
    rotateX(random.uniform(0, max_angle)) @
    rotateY(random.uniform(0, max_angle)) @
    rotateZ(random.uniform(0, max_angle))
)
    rigidMorph = rigidMorph.to(tensors[0].device)

    # Apply SAME rigidMorph to each tensor
    return tuple(rotateData_about_center(t, rigidMorph) for t in tensors)


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

    baseAddress = "/"
    X_train = torch.load(baseAddress+"<fileName>.pt").clone().detach().to(device)
    Y_train = torch.load(baseAddress+"<fileName>..pt").clone().detach().to(device)
    X_test = torch.load(baseAddress+"<fileName>..pt").clone().detach().to(device)
    Y_test = torch.load(baseAddress+"<fileName>..pt").clone().detach().to(device)

    # specify the rotations of interest
    rigidMorph = rotateX(12) @ rotateY(321) @ rotateZ(143)
    print(rigidMorph)

    if(rigidMorph.shape != (3, 3)):
        raise Exception("Rigid morph shape is: ", rigidMorph.shape)
    
    # rotate the data
    largeX_train = torch.cat((X_train, rotateData_about_(X_train, rigidMorph.to(device))))
    largeY_train = torch.cat((Y_train, rotateData_about_(Y_train, rigidMorph.to(device))))
    largeX_test = torch.cat((X_test, rotateData_about_(X_test, rigidMorph.to(device))))
    largeY_test = torch.cat((Y_test, rotateData_about_(Y_test, rigidMorph.to(device))))
    torch.save(largeX_train.cpu(), baseAddress+"xTrainRotated.pt")
    torch.save(largeY_train.cpu(), baseAddress+"yTrainRotated.pt")
    torch.save(largeX_test.cpu(), baseAddress+"xTestRotated.pt")
    torch.save(largeY_test.cpu(), baseAddress+"yTestRotated.pt")
    print("shape of rotated data is: ", largeX_train.shape)
    print("Done!")
    