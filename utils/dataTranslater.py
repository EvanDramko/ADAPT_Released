# imports
import torch
import random
import warnings
import inspect

BOX_EDGE = 16.406184

def translateX(tens, distance):
    # Check the calling function is "randomTranslate"
    caller = inspect.stack()[2].function
    if caller != 'randomTranslate':
        warnings.warn("Be careful calling this directly. Protections are incorporated in randomTranslate only")

    assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
    tensor = tens.clone().detach()
    tensor[..., 0] = (tensor[..., 0] + distance) # % BOX_EDGE
    return tensor

def translateY(tens, distance):
    # Check the calling function is "randomTranslate"
    caller = inspect.stack()[2].function
    if caller != 'randomTranslate':
        warnings.warn("Be careful calling this directly. Protections are incorporated in randomTranslate only")
    
    assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
    tensor = tens.clone().detach()
    tensor[..., 1] = (tensor[..., 1] + distance) # % BOX_EDGE
    return tensor

def translateZ(tens, distance):
    # Check the calling function is "randomTranslate"
    caller = inspect.stack()[2].function
    if caller != 'randomTranslate':
        warnings.warn("Be careful calling this directly. Protections are incorporated in randomTranslate only")
    
    assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
    tensor = tens.clone().detach()
    tensor[..., 2] = (tensor[..., 2] + distance) # % BOX_EDGE
    return tensor

def randomTranslate(data):
    # Set box boundaries. Be careful in non-square "boxes" the order in which you translate data
    WIDTH = 10
    shift_x = random.uniform(0, WIDTH) 
    shift_y = random.uniform(0, WIDTH)
    shift_z = random.uniform(0, WIDTH)

    def apply_translation(tens):
        assert isinstance(tens, torch.Tensor), "tens must be a torch.Tensor"
        tens = translateX(tens, shift_x)
        tens = translateY(tens, shift_y)
        tens = translateZ(tens, shift_z)
        return tens

    if isinstance(data, tuple):
        return tuple(apply_translation(tens) for tens in data)
    else:
        return apply_translation(data)


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
    X_train = torch.load(baseAddress+"<fileName>.pt").clone().detach().to(device)

    X = X_train[1]
    print(X.shape)
    print(X)
    print("=========")
    a = randomTranslate(X)
    print(a.shape)
    print(a)
    print(X)

    print("Done!")
    