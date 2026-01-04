# imports
import torch
import utils.padder
import utils.normalizer
from configs import energy_model_hyperparams, force_model_hyperparam
import warnings

def getStructEnergy(atomStruct, model, maxStructSize=energy_model_hyperparams.DataConfig.max_atoms, atomFeatureLength=energy_model_hyperparams.DataConfig.atom_vec_length):
    """
    Finds the predicted energy of the given structure

    Args:
        atomStruct (torch.Tensor): atomic structure (crystal/molecule) as torch tensor: (B,n,v)
        model (torch model): saved nrg prediction model
        maxStructsize (int): maximum number of atoms in the structure
        atomFeatureLength (int): length of the vector representation of each atom

    Return:
        float: predicted energy of the structure
    
    """
    assert atomStruct.shape[-1] == atomFeatureLength, f"The atomic description vector must have {atomFeatureLength} components."
    assert atomStruct.shape[-2] <= maxStructSize, f"You must have no more than maxStructSize atoms."
    assert atomStruct.ndims == 3, f"You must have three dimensional input (batch, n atoms, v feature descriptors). You have {atomStruct.ndims} input dimensions."
    if(model.device == "cpu"):
        warnings.warn("CPU runtimes will likely be lengthy. Typically, gpu accelerated runtimes are recommended.")

    
    # pad the data to appropriate depth (number of atoms)
    structure = utils.padder.padToDepth(atomStruct, maxStructSize).unsqueeze(0).to(model.device) # the maxStructSize is enforced by the model archcitecture. We add a dummy batch dimension

    stats = torch.load(energy_model_hyperparams.ModelPaths.stats_path, map_location=model.device)

    # normalize the data
    meanX = stats["meanX"].clone().detach().to(model.device)
    stdX  = stats["stdX"].clone().detach().to(model.device)
    normedStruct = utils.normalizer.normalize_all(0, structure, meanX, stdX)

    # get energy prediction
    predictedEnergy = model(normedStruct)

    # un-normalize the prediction
    meanY = stats["meanY"].clone().detach().to(model.device)
    stdY  = stats["stdY"].clone().detach().to(model.device)
    normedStruct = utils.normalizer.unnormalize_all(1, predictedEnergy, meanY, stdY)

    return normedStruct.item()


def getOneStepForces(rawStructure, model, maxStructSize=energy_model_hyperparams.DataConfig.max_atoms, atomFeatureLength=energy_model_hyperparams.DataConfig.atom_vec_length):
    """
    Finds the predicted energy of the given structure

    Args:
        atomStruct (torch.Tensor): atomic structure (crystal/molecule) as torch tensor: (B,n,v)
        model (torch model): saved nrg prediction model
        maxStructsize (int): maximum number of atoms in the structure
        atomFeatureLength (int): length of the vector representation of each atom

    Return:
        torch.Tensor: forces as (B,n,3) tensor
    
    """
    assert rawStructure.shape[-1] == atomFeatureLength, f"The atomic description vector must have {atomFeatureLength} components."
    assert rawStructure.shape[-2] <= maxStructSize, f"You must have no more than maxStructSize atoms."
    assert rawStructure.ndims == 3, f"You must have three dimensional input (batch, n atoms, v feature descriptors). You have {rawStructure.ndims} input dimensions."
    if(model.device == "cpu"):
        warnings.warn("CPU runtimes will likely be lengthy. Typically, gpu accelerated runtimes are recommended.")

    # pad the data to appropriate depth (number of atoms)
    structure = rawStructure.to(torch.float32).to(model.device)

    stats = torch.load(force_model_hyperparam.ModelPaths.stats_path, map_location=model.device)

    # normalize the data
    meanX = stats["meanX"].clone().detach().to(model.device)
    stdX  = stats["stdX"].clone().detach().to(model.device)
    normedStruct = utils.normalizer.normalize_all(0, structure, meanX, stdX)

    # get energy prediction
    forcesPred = model(normedStruct)

    # un-normalize the prediction
    meanY = stats["meanY"].clone().detach().to(model.device)
    stdY  = stats["stdY"].clone().detach().to(model.device)
    unnormStruct = utils.normalizer.unnormalize_all(1, forcesPred, meanY, stdY)

    return unnormStruct

def loop2Convergence(getForces, structure, step=0.05, annealFactor=0.99, momemWeight=0.00, threshold=0.001, maxSteps = 75, saveFrames = None): 
    """
    Takes a given structure and a force/gradient predictor, then performs atom-level gradient descent until convergence. 

    Args: 
        getForces: fxn in the form getForces(struct) -> forcePerAtom (likely DFT or MLFF)
        structure: unstable structure to relax as torch tensor (1, n, 12)
        step: PES-level step size
        annealFactor: step size decay factor
        momenWeight: momentum weighting term
        threshold: L1-max angstrom distance based convergence threshold
        maxSteps: maximum number of allowed steps. Use of -1 indicates no limit
        saveFrames (Optional[int]): save a cif file of the structure every saveFrames steps

    Returns: 
        stable prediction as torch tensor (1, n, 12)
    """
    raise Exception("Depricated usage! New relaxation method coming soon!")

    # define nested function for updating step
    #=========================================================== Stuctural Update ============================================================
    def stepStructDirect(atomLocations, displacement, stepSize, momWeight=None, momemTerm=None):
        # scale forces by actual step size
        stuctUpdate = displacement * stepSize
        
        if (momemTerm != None):
            if(momWeight == None):
                raise Exception("If using momentum, you must have a momentum weight!")
            
            regWeight = 1 - momWeight
            stuctUpdate = (stuctUpdate*regWeight) + (momemTerm*momWeight)
        
        # update atom locations based on scaled forces
        newLoc = atomLocations + stuctUpdate
        # delta = torch.norm(scaledForces, p=2)
        delta = torch.max(stuctUpdate)
        return (newLoc, delta)
    #========================================================= End Structural Update =========================================================
    delta = 1+threshold # number larger than the threshold
    countSteps = 0
    oldDisp = None
    # ------ In the Loop ------
    while(delta > threshold):
        if(countSteps == maxSteps):
            # exceeded allowed count of steps. 
            print("steps taken: ", countSteps)
            return structure
        
        countSteps += 1
        # if (countSteps % 10 == 0): # periodically save structure snapshot (change to desired frequency)
        #     makeCif(structure.cpu(), "saveCifsOverwrite/step"+str(countSteps))

        # split locations from features
        oldLocations = structure[:, :3].clone()
        
        # generate forces
        if (atomicWeight == False):
            rawForceVecs = getForces(utils.normalizer.normalize_all(0, structure, mean=torch.load('utils/meanX.pt', map_location=structure.device), std=torch.load('utils/stdX.pt', map_location=structure.device)))
        else:
            tempStuct = structure.clone()
            atomicWeights = utils.getAtomicWeights.get_atomic_weights(tempStuct)
            tempStuct = utils.appendToFeatureList.insert_tensor_as_second_last_channel(tempStuct, atomicWeights)
            rawForceVecs = getForces(utils.normalizer.normalize_all(0, tempStuct))


        dispA = utils.downscaleForces.force2Angstroms(rawForceVecs, structure)

        # update locations
        if (oldDisp == None):
            structure[:, :3], delta = stepStructDirect(oldLocations, dispA, stepSize=step)
        else:
            structure[:, :3], delta = stepStructDirect(oldLocations, dispA, stepSize=step, momWeight=momemWeight, momemTerm=oldDisp)

        # anneal step size
        step *= annealFactor

        # update locations of variables
        oldDisp = dispA.clone()
        # loop again! :)
    
    print("steps taken: ", countSteps)
    return structure
