import torch
from . import build_atomic_table

def tens_dq_from_defects(candidate, actual):
    # get squared errors
    diff = candidate[..., :3]-actual[..., :3] # only record the difference in coordinates (inclusion of others is fine, but unnecessary!)
    dist_per_atom = torch.sum(diff*diff, dim=-1)

    # build tables and pull atomic weights
    weight_table = build_atomic_table.get_weight_table().to(candidate.device)
    lookup_table = build_atomic_table.get_atomic_number_map().to(candidate.device)
    atomic_nums = lookup_table[candidate[..., 4].long(), candidate[..., 3].long()]
    atom_weights = weight_table[atomic_nums.long()]

    # weight by atomic weights and return weighted displacement
    weight_dist = atom_weights*dist_per_atom
    atom_error_squared = torch.sum(weight_dist, dim=-1)
    atom_error = torch.sqrt(atom_error_squared)
    return torch.sum(atom_error)

def tens_dq_from_z(candidate, actual):
    # get squared errors
    diff = candidate[..., :3]-actual[..., :3] # only record the difference in coordinates (inclusion of others is fine, but unnecessary!)
    dist_per_atom = torch.sum(diff*diff, dim=-1)

    # build tables and pull atomic weights
    weight_table = build_atomic_table.get_weight_table().to(candidate.device)
    atom_weights = weight_table[candidate[..., 3].long()]    

    # weight by atomic weights and return weighted displacement
    weight_dist = atom_weights*dist_per_atom
    atom_error_squared = torch.sum(weight_dist, dim=-1)
    atom_error = torch.sqrt(atom_error_squared)
    return torch.sum(atom_error)