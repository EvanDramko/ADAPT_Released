# imports
import torch
import utils.padder
import utils.normalizer
from configs import energy_model_hyperparams, force_model_hyperparam
from MLFF_arch import energyMLFF, encoderBase
import warnings

def _require(cond: bool, msg: str, exc=ValueError):
    if not cond:
        raise exc(msg)

class Runner:
    def __init__(self, use_force = True, use_energy = True, hasWeighting = force_model_hyperparam.DataConfig.has_weighting):
        self.device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps:0") if torch.backends.mps.is_available() else torch.device("cpu"))

        # load models
        if use_force:
            config = force_model_hyperparam.ModelConfig() 
            _require(force_model_hyperparam.ModelPaths.pretrainPath != None, "Must have a saved model to use at inference time for the force predictor")
            self.force_model = encoderBase.TransformerEncoder(d_model=config.d_model, d_ff=config.d_ff, num_layers=config.num_layers, d_out=config.d_out, dropout_rate=config.dropout_rate, num_heads=config.num_heads).to(self.device)
            self.force_model.load_state_dict(torch.load(force_model_hyperparam.ModelPaths.pretrainPath, map_location=torch.device(self.device)))
            self.force_model.eval()

        if use_energy:
            _require(energy_model_hyperparams.ModelPaths.pretrainPath != None, "Must have a saved model to use at inference time for the energy predictor")
            self.energy_model = energyMLFF.nrgMLP(energy_model_hyperparams.ModelConfig(energy_model_hyperparams.DataConfig()).dimensions_list).to(self.device)
            self.energy_model.load_state_dict(torch.load(energy_model_hyperparams.ModelPaths.pretrainPath, map_location=self.device))
            self.energy_model.eval()

        # load data shape information
        self.max_atoms = energy_model_hyperparams.DataConfig.max_atoms
        self.atom_feature_length = energy_model_hyperparams.DataConfig.atom_vec_length
        _require(force_model_hyperparam.DataConfig.atom_vec_length == self.atom_feature_length, f"Both the energy and force models should have the same atomic feature descriptors")
        
        # load normalizaton stats
        stats = torch.load(energy_model_hyperparams.ModelPaths.stats_path, map_location=self.device)
        # print(stats)
        if hasWeighting:
            self.meanX = stats["x_mean"][..., :-1]
            self.stdX  = stats["x_std"][..., :-1]
        else:
            self.meanX = stats["x_mean"]
            self.stdX  = stats["x_std"]
        if use_energy:
            self.meanY_energy = stats["y_mean_energy"]
            self.stdY_energy  = stats["y_std_energy"]
        if use_force:
            self.meanY_force = stats["y_mean_force"]
            self.stdY_force  = stats["y_std_force"]
            

    @torch.inference_mode
    def getStructEnergy(self, atomStruct):
        """
        Finds the predicted energy of the given structure using the preloaded model

        Args:
            atomStruct (torch.Tensor): atomic structure (crystal/molecule) as torch tensor: (B,n,v)

        Return:
            float: predicted energy of the structure
        
        """
        maxStructSize = self.max_atoms
        _require(isinstance(atomStruct, torch.Tensor), "raw structure should be a torch tensor")
        _require(atomStruct.shape[-1] == self.atom_feature_length, f"The atomic description vector must have atomFeatureLength={self.atom_feature_length} components.")
        _require(atomStruct.shape[-2] <= maxStructSize, f"You must have no more than maxStructSize={maxStructSize} atoms.")
        _require(atomStruct.ndim == 3, f"You must have three dimensional input (batch, n atoms, v feature descriptors). You have {atomStruct.ndim} input dimensions.")
        _require(self.device == next(self.energy_model.parameters()).device, f"model is not on the same device ({next(self.energy_model.parameters()).device}) as the Runner class ({self.device})", exc=RuntimeError)
        if(self.device == "cpu"):
            warnings.warn("CPU runtimes will likely be lengthy. Typically, gpu accelerated runtimes are recommended.")

        
        # pad the data to appropriate depth (number of atoms)
        structure = utils.padder.padToDepth(atomStruct, maxStructSize).to(self.device) # the maxStructSize is enforced by the model archcitecture. We add a dummy batch dimension

        # normalize the data
        normedStruct = utils.normalizer.normalize_all(0, structure, self.meanX, self.stdX)

        # get energy prediction
        predictedEnergy = self.energy_model(normedStruct)

        # un-normalize the prediction
        normedStruct = utils.normalizer.unnormalize_all(1, predictedEnergy, self.meanY_energy, self.stdY_energy)

        return normedStruct


    @torch.inference_mode
    def getOneStepForces(self, rawStructure):
        """
        Predicts the per atom forces of the given structure(s)

        Args:
            atomStruct (torch.Tensor): atomic structure (crystal/molecule) as torch tensor: (B,n,v)

        Return:
            torch.Tensor: forces as (B,n,3) tensor
        
        """
        _require(isinstance(rawStructure, torch.Tensor), "rawStructure should be a torch tensor")
        _require(rawStructure.shape[-1] == self.atom_feature_length, f"The atomic description vector must have atomFeatureLength={self.atom_feature_length} components.")
        _require(rawStructure.ndim == 3, f"You must have three dimensional input (B batch, n atoms, v feature descriptors). You have {rawStructure.ndim} input dimensions.")
        _require(self.device == next(self.force_model.parameters()).device, f"model is not on the same device ({next(self.force_model.parameters()).device}) as the Runner class ({self.device})", exc=RuntimeError)
        if(self.device == "cpu"):
            warnings.warn("CPU runtimes will likely be lengthy. Typically, gpu accelerated runtimes are recommended.")

        # pad the data to appropriate depth (number of atoms)
        structure = rawStructure.to(torch.float32).to(self.device)

        # normalize the data
        normedStruct = utils.normalizer.normalize_all(0, structure, self.meanX, self.stdX)

        # get energy prediction
        forcesPred = self.force_model(normedStruct)
        # print("Force prediction shape is: ", forcesPred.shape)

        # un-normalize the prediction
        unnormedForces = utils.normalizer.unnormalize_all(1, forcesPred, self.meanY_force, self.stdY_force)

        return unnormedForces



# A NEW AND FAR IMPROVED VERSION OF THIS LOOPING IS BEING ADDED SOON!
# def loop2Convergence(structure, getForces = model, step=0.05, annealFactor=0.99, momemWeight=0.00, threshold=0.001, maxSteps = 75, saveFrames = None): 
#     """
#     Takes a given structure and a force/gradient predictor, then performs atom-level gradient descent until convergence. 

#     Args: 
#         getForces: fxn in the form getForces(struct) -> forcePerAtom (likely DFT or MLFF)
#         structure: unstable structure to relax as torch tensor (1, n, 12)
#         step: PES-level step size
#         annealFactor: step size decay factor
#         momenWeight: momentum weighting term
#         threshold: L1-max angstrom distance based convergence threshold
#         maxSteps: maximum number of allowed steps. Use of -1 indicates no limit
#         saveFrames (Optional[int]): save a cif file of the structure every saveFrames steps

#     Returns: 
#         stable prediction as torch tensor (1, n, 12)
#     """
#     raise Exception("Depricated usage! New relaxation method coming soon!")

#     # define nested function for updating step
#     #=========================================================== Stuctural Update ============================================================
#     def stepStructDirect(atomLocations, displacement, stepSize, momWeight=None, momemTerm=None):
#         # scale forces by actual step size
#         stuctUpdate = displacement * stepSize
        
#         if (momemTerm != None):
#             if(momWeight == None):
#                 raise Exception("If using momentum, you must have a momentum weight!")
            
#             regWeight = 1 - momWeight
#             stuctUpdate = (stuctUpdate*regWeight) + (momemTerm*momWeight)
        
#         # update atom locations based on scaled forces
#         newLoc = atomLocations + stuctUpdate
#         # delta = torch.norm(scaledForces, p=2)
#         delta = torch.max(stuctUpdate)
#         return (newLoc, delta)
#     #========================================================= End Structural Update =========================================================
#     delta = 1+threshold # number larger than the threshold
#     countSteps = 0
#     oldDisp = None
#     # ------ In the Loop ------
#     while(delta > threshold):
#         if(countSteps == maxSteps):
#             # exceeded allowed count of steps. 
#             print("steps taken: ", countSteps)
#             return structure
        
#         countSteps += 1
#         # if (countSteps % 10 == 0): # periodically save structure snapshot (change to desired frequency)
#         #     makeCif(structure.cpu(), "saveCifsOverwrite/step"+str(countSteps))

#         # split locations from features
#         oldLocations = structure[:, :3].clone()
        
#         # generate forces
#         if (atomicWeight == False):
#             rawForceVecs = getForces(utils.normalizer.normalize_all(0, structure, mean=torch.load('utils/meanX.pt', map_location=structure.device), std=torch.load('utils/stdX.pt', map_location=structure.device)))
#         else:
#             tempStuct = structure.clone()
#             atomicWeights = utils.getAtomicWeights.get_atomic_weights(tempStuct)
#             tempStuct = utils.appendToFeatureList.insert_tensor_as_second_last_channel(tempStuct, atomicWeights)
#             rawForceVecs = getForces(utils.normalizer.normalize_all(0, tempStuct))


#         dispA = utils.downscaleForces.force2Angstroms(rawForceVecs, structure)

#         # update locations
#         if (oldDisp == None):
#             structure[:, :3], delta = stepStructDirect(oldLocations, dispA, stepSize=step)
#         else:
#             structure[:, :3], delta = stepStructDirect(oldLocations, dispA, stepSize=step, momWeight=momemWeight, momemTerm=oldDisp)

#         # anneal step size
#         step *= annealFactor

#         # update locations of variables
#         oldDisp = dispA.clone()
#         # loop again! :)
    
#     print("steps taken: ", countSteps)
#     return structure
