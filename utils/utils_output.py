import torch
from ase import Atom, Atoms
from cdvae.common.data_utils import lattice_params_to_matrix_torch

def output_eval(data_path):
    # for recon, gen, opt
    data = torch.load(data_path, map_location='cpu')
    keys = list(data.keys())
    lengths = data['lengths']
    angles = data['angles']
    num_atoms = data['num_atoms']
    frac_coords = data['frac_coords']
    atom_types = data['atom_types']
    eval_setting = data['eval_setting']
    time = data['time']
    if 'all_frac_coords_stack' in keys:
        all_frac_coords_stack = data['all_frac_coords_stack']
        all_atom_types_stack =data['all_atom_types_stack']
        return lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time
    return lengths, angles, num_atoms, frac_coords, atom_types, eval_setting, time


def get_astruct(atom_type, cart_coord, lattice):
    atoms = Atoms(symbols=atom_type, positions = cart_coord,
            cell = lattice,
            pbc=True) 
    return atoms

def get_astruct_list(data_path, idx):
    # for recon, gen
    outputs = output_eval(data_path)
    lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time = output_eval(data_path)
    lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
    lattice = lattices[idx, :, :]
    sum_idx_bef = num_atoms[0, :idx].sum()
    sum_idx_aft = num_atoms[0, :idx+1].sum()
    steps = all_frac_coords_stack.shape[1]
    astruct_list = []
    for t in range(steps):
        frac = all_frac_coords_stack[0, t, sum_idx_bef:sum_idx_aft, :].to('cpu')
        cart = frac@lattice.T
        atoms = all_atom_types_stack[0, t, sum_idx_bef:sum_idx_aft].to('cpu')
        astruct = Atoms(symbols=atoms, positions = cart, cell = lattice, pbc=True) 
        astruct_list.append(astruct)
    return astruct_list


def get_astruct_all_list(data_path):
    # for recon, gen
    outputs = output_eval(data_path)
    lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time = output_eval(data_path)
    lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
    num = len(lattices)
    astruct_lists = []
    for idx in range(num):
        lattice = lattices[idx, :, :]
        sum_idx_bef = num_atoms[0, :idx].sum()
        sum_idx_aft = num_atoms[0, :idx+1].sum()
        steps = all_frac_coords_stack.shape[1]
        astruct_list = []
        for t in range(steps):
            frac = all_frac_coords_stack[0, t, sum_idx_bef:sum_idx_aft, :].to('cpu')
            cart = frac@lattice.T
            atoms = all_atom_types_stack[0, t, sum_idx_bef:sum_idx_aft].to('cpu')
            astruct = Atoms(symbols=atoms, positions = cart, cell = lattice, pbc=True) 
            astruct_list.append(astruct)
        astruct_lists.append(astruct_list)
    return astruct_lists