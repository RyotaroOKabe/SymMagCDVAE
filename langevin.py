#%%
import torch
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from cdvae.common.data_utils import lattice_params_to_matrix_torch
from cdvae.pl_modules.model import CDVAE
from torch_geometric.data import Data
from torch.utils.data import Dataset
import math as m

from utils.utils_plot import vis_structure
from utils.utils_material import MatSym, MatTrans, distance_sorted, Rx, Ry, Rz, rotate_cart, switch_latvecs
tol = 1e-03
import os, sys
import itertools
from pymatgen.analysis.structure_matcher import StructureMatcher
from torch.autograd.functional import jacobian

from pathlib import Path
from types import SimpleNamespace

from scripts.eval_utils import load_model

#%%
# load data
model_path = ''
tasks = ['gen']
start_from='data'
n_step_each=100
step_lr=1e-4
min_sigma=0
save_traj=True
disable_bar=False

# load model
model_path = Path(model_path)
model, test_loader, cfg = load_model(
    model_path, load_data=('recon' in tasks) or
    ('opt' in tasks and start_from == 'data'))
ld_kwargs = SimpleNamespace(n_step_each=n_step_each,
                            step_lr=step_lr,
                            min_sigma=min_sigma,
                            save_traj=save_traj,
                            disable_bar=disable_bar)


#%%
# defineLangevin class with dynamics indidivually
class CDVAE_SGO(CDVAE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
       
    @torch.no_grad()
    def langevin_dynamics_sgo(self, z, esgo, ld_kwargs, sgo, alpha, gt_num_atoms=None, gt_atom_types=None):   #!! play around
        """
        decode crystral structure from latent embeddings.
        z: latent space
        esgo: embedded space group operations
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats. #OK
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, esgo, gt_num_atoms)  #OK #?esgo
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.    #OK
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.  #OK
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.   
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, esgo, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles) #? esgo 
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                if alpha > 0:
                    frac = cur_frac_coords.copy()
                    sgo_loss = sgo_cum_loss(cur_frac_coords, sgo, r_max=0.8)    #!
                    sgo_loss.backward()
                    dLdx = alpha * frac.grad
                else: 
                    dLdx = 0    #!
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart + dLdx#!
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj: #OK
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict
    
     
    def forward(self, batch, teacher_forcing, training):
        pass




# [1] normal langevin
torch.no_grad()


# [2] langevin with sgo loss



