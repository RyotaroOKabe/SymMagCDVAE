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

