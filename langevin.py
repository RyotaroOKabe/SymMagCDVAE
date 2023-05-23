#%%
import torch
from torch_geometric.data import Batch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.groups import SpaceGroup
from cdvae.common.data_utils import lattice_params_to_matrix_torch, frac_to_cart_coords, cart_to_frac_coords
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

from scripts.eval_utils import load_model   # maybe need to modify this!
import time
# import argparse   # do not use this!!!

from tqdm import tqdm
from torch.optim import Adam
import hydra
from hydra import initialize_config_dir
from hydra.experimental import compose


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
def load_model_sgo(model_path, load_data=False, testing=True):  #how to load the model with the updated langevin dynamics?
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams_sgo')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model.load_from_checkpoint(ckpt, strict=False)
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
        model.scaler = torch.load(model_path / 'prop_scaler.pt')

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg


def prep(model_path, n_step_each, step_lr, min_sigma, save_traj, disable_bar) :
    model_path = Path(model_path)
    model, test_loader, cfg = load_model_sgo(       #!!
        model_path, load_data=('recon' in tasks) or
        ('opt' in tasks and start_from == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=n_step_each,
                                step_lr=step_lr,
                                min_sigma=min_sigma,
                                save_traj=save_traj,
                                disable_bar=disable_bar)
    if torch.cuda.is_available():
        model.to('cuda')
    return model, test_loader, cfg, ld_kwargs

def recon():
    print('Evaluate model on the reconstruction task.')
    start_time = time.time()
    (frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstructon(
        test_loader, model, ld_kwargs, args.num_evals,
        args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)

    if args.label == '':
        recon_out_name = 'eval_recon.pt'
    else:
        recon_out_name = f'eval_recon_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'all_frac_coords_stack': all_frac_coords_stack,
        'all_atom_types_stack': all_atom_types_stack,
        'time': time.time() - start_time
    }, model_path / recon_out_name)

def gen(model, test_loader, cfg):
    print('Evaluate model on the generation task.')
    start_time = time.time()

    (frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack) = generation(
        model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
        args.batch_size, args.down_sample_traj_step)

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'all_frac_coords_stack': all_frac_coords_stack,
        'all_atom_types_stack': all_atom_types_stack,
        'time': time.time() - start_time
    }, model_path / gen_out_name)


#%%
# defineLangevin class with dynamics indidivually
class CDVAE_SGO(CDVAE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
       
    @torch.no_grad()
    def langevin_dynamics_sgo(self, z, ld_kwargs, sgo, alpha=1, gt_num_atoms=None, gt_atom_types=None):   #!! play around
        """
        decode crystral structure from latent embeddings.
        z: latent space
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
            z, gt_num_atoms)  #OK #?esgo
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
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles) #? esgo 
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


def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics_sgo(
                z, ld_kwargs, gt_num_atoms, gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def generation(model, ld_kwargs, sgo, alpha, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics_sgo(z, ld_kwargs, sgo, alpha)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


# [1] normal langevin
torch.no_grad()


# [2] langevin with sgo loss


#%%
if __name__=='__main__':
    model_path='/home/rokabe/data2/generative/hydra/singlerun/2023-05-18/mp_20_1'
    n_step_each=20 
    step_lr=1e-4
    min_sigma=0 
    save_traj=False
    disable_bar=False
    num_batches_to_sample=4
    num_samples_per_z=5
    alpha=1
    # Define the space group number
    spacegroup_number = 62

    # Create a SpaceGroup object from the space group number
    spacegroup = SpaceGroup.from_int_number(spacegroup_number)

    # Get the space group operations
    operations = spacegroup.symmetry_ops

    sgo = [ope.rotation_matrix for ope in operations]
    model, test_loader, cfg, ld_kwargs = prep(model_path, n_step_each, step_lr, min_sigma, save_traj, disable_bar)
    generation(model, ld_kwargs, sgo, alpha, num_batches_to_sample, num_samples_per_z, batch_size=512, down_sample_traj_step=1)


# %%
