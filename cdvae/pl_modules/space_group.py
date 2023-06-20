from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm
import itertools
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from cdvae.common.data_utils import lattice_params_to_matrix_torch

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):  #OK
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

def struct2spgop(batch):
    """
    Get space group operation from the pymatgen.core.Structure
    """
    # print("=====batch start======")
    # print(batch)
    num = len(batch)
    # print("batch:", batch.batch)
    # print(len(batch.batch))
    # print(batch.lengths.shape)
    # print(batch.angles.shape)
    # print(batch.num_atoms.shape)
    # print(batch.atom_types.shape)
    # for i in range(num):
    #     print(batch[i].lengths)
    # print("=====batch ends======")
    num_batch = torch.max(batch.batch)+1
    # print(num_batch)
    lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    # print("lattices: ", lattices.shape)
    sgo =[]
    sgo_batch = []
    for i in range(num_batch):
        is_batch = batch.batch==i
        frac_coord=batch.frac_coords[is_batch,:]
        atom_type = batch.atom_types[is_batch]
        lattice = lattices[i, :, :]
        pstruct = Structure(
            lattice=np.array(lattice.cpu()),
            species=np.array(atom_type.cpu()),
            coords=np.array(frac_coord.cpu()),
            coords_are_cartesian=False
        )
        sga = SpacegroupAnalyzer(pstruct)
        opes = set(sga.get_symmetry_operations())
        nopes = len(opes)
        # print(f"nopes [{i}] {nopes}")
        oprs=torch.stack([Tensor(ope.rotation_matrix) for ope in opes])
        # oprs=torch.cat([Tensor(ope.rotation_matrix) for ope in sga.get_symmetry_operations()], dim=0)
        sgo.append(oprs)
        sgo_batch += [int(i) for _ in range(nopes)]
    # print("sgo_batch0: ", sgo_batch)
    return torch.cat(sgo, dim=0).to(batch.angles), (Tensor(sgo_batch).to(batch.angles).type(torch.int64))
   

class Embed_SPGOP(torch.nn.Module):
    """
    Embed the space group operation as the vector to incorporate it into GNN
    """
    def __init__(self, hidden_dim, out_dim, fc_num_layers) -> None:
        super().__init__()
        self.in_dim = 9
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc_num_layers = fc_num_layers
        self.mlp = build_mlp(self.in_dim, self.hidden_dim, self.fc_num_layers, self.out_dim)


    def forward(self, sgo, sgo_batch):
        # print("sgo_batch: ", sgo_batch)
        # print("sgo.shape: ", sgo.shape)
        x = sgo.reshape(-1, self.in_dim)
        esgo = self.mlp(x)
        return scatter(esgo, sgo_batch, dim=0, reduce='mean')

def esgo_repeat_batches(esgo, batch):
    return esgo[batch, :]

def esgo_repeat(esgo, num_atoms):
    return esgo.repeat_interleave(num_atoms, dim=0)



def get_neighbors(frac, r_max):
    """
    Get the neighbor lists in fractional coordinates. 
    """
    # frac = torch.tensor(frac, requires_grad=grad)
    natms = len(frac)
    # get shift indices of the 1st nearest neighbor cells
    shiftids = torch.tensor(list(itertools.product([0,1,-1], repeat=3)))
    shifts = shiftids.repeat_interleave(natms, dim=0).to(frac)
    #fractional cell with the shift
    frac_ex = torch.cat([frac for _ in range(27)], dim=0) + shifts 
    # frac_ex.requires_grad=True
    # dist matrix
    dmatrix = torch.cdist(frac, frac_ex, p=2)
    mask = dmatrix<=r_max
    idx_mask = torch.nonzero(mask)
    idx_src, idx_dst = idx_mask[:,0].reshape(-1), idx_mask[:,1].reshape(-1)
    # use the indices to get the list of vectors
    frac_src = frac_ex[idx_src, :]
    frac_dst = frac_ex[idx_dst, :]
    vectors = frac_dst-frac_src
    return idx_src, idx_dst, vectors


def sgo_loss(frac, opr, r_max): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    _, _, edge_vec0 = get_neighbors(frac0, r_max)
    _, _, edge_vec1 = get_neighbors(frac1, r_max)
    wvec0 = edge_vec0*edge_vec0
    wvec1 = edge_vec1*edge_vec1
    out0 = wvec0.sum(dim=0)
    out1 = wvec1.sum(dim=0)
    diff= out1 - out0
    return diff.norm()

def sgo_cum_loss(frac, oprs, r_max):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss(frac, opr, r_max)
        loss += diff
    return loss/nops


#230620 
def perm_invariant_loss(tensor1, tensor2):
    dists = torch.cdist(tensor1, tensor2)
    min_dists = dists.min(dim=1)[0]
    return min_dists.mean()

def symmetric_perm_invariant_loss(tensor1, tensor2):
    loss1 = perm_invariant_loss(tensor1, tensor2)
    loss2 = perm_invariant_loss(tensor2, tensor1)
    return (loss1 + loss2) / 2

def sgo_loss_perm(frac, opr, r_max): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    _, _, edge_vec0 = get_neighbors(frac0, r_max)
    _, _, edge_vec1 = get_neighbors(frac1, r_max)
    return symmetric_perm_invariant_loss(edge_vec0, edge_vec1)

def sgo_cum_loss_perm(frac, oprs, r_max):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_perm(frac, opr, r_max)
        loss += diff
    return loss/nops

def diffuse_frac(pstruct, sigma=0.1):
    frac = pstruct.frac_coords
    lat = pstruct.lattice.matrix
    spec = pstruct.species
    dist = np.random.normal(loc=0.0, scale=sigma, size=frac.shape)
    frac1 = frac + dist
    struct_out = Structure(
        lattice=lat,
        species=spec,
        coords=frac1,
        coords_are_cartesian=False
    )
    return struct_out