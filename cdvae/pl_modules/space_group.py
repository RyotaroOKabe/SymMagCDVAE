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


class SGO_Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def sgo_loss(self, frac, opr):
        pass
    
    def sgo_cum_loss(self, fracs, natms, oprss, noprs):
        """
        Cumulative loss from space group operations.
        """
        loss = 0    #torch.zeros(1).to(fracs)
        # loss.requires_grad=True
        natms = natms.flatten()
        noprs = noprs.flatten()
        nfracs = len(natms)
        for idx in range(nfracs):
            idx_f_bef = natms[:idx].sum()
            idx_f_aft = natms[:idx+1].sum()
            idx_o_bef = noprs[:idx].sum()
            idx_o_aft = noprs[:idx+1].sum()
            # print('idx, idx_f_from, idx_f_to: ', (idx, idx_f_bef, idx_f_aft))
            # print('idx, idx_o_from, idx_o_to: ', (idx, idx_o_bef, idx_o_aft))
            frac = fracs[idx_f_bef:idx_f_aft, :]
            oprs = oprss[idx_o_bef:idx_o_aft, :]
            # print('oprs, fracs: ', oprs.shape, frac.shape)
            nops = len(oprs)
            for opr in oprs:
                diff = self.sgo_loss(frac, opr)
                loss += diff/(nops*nfracs)
        return loss

    def forward(self, fracs, natms, oprss, noprs):
        return self.sgo_cum_loss(fracs, natms, oprss, noprs)

class SGO_Loss_Prod(SGO_Loss):
    def __init__(self, r_max, power=0) -> None:
        super().__init__()
        self.r_max=r_max
        self.power=power
        
    def sgo_loss(self, frac, opr): # can be vectorized for multiple space group opoerations?
        """
        Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
        """
        frac0 = frac.clone()#.detach()
        frac0.requires_grad_()
        frac1 = frac.clone()
        frac1.requires_grad_()
        frac1 = frac1@opr.T%1
        natm = len(frac0)
        # print('frac0, frac1, opr: ', frac0.shape, frac1.shape, opr.shape)
        _, _, edge_vec0 = get_neighbors(frac0, self.r_max/pow(natm, self.power))
        _, _, edge_vec1 = get_neighbors(frac1, self.r_max/pow(natm, self.power))
        wvec0 = edge_vec0*edge_vec0
        wvec1 = edge_vec1*edge_vec1
        out0 = wvec0.sum(dim=0)
        out1 = wvec1.sum(dim=0)
        diff= out1 - out0
        return diff.norm()


#230620 
class SGO_Loss_Perm(SGO_Loss):
    def __init__(self, r_max, power=0, use_min_edges=False, num_lens=1, threshold=1e-3) -> None:
        super().__init__()
        self.r_max = r_max
        self.use_min_edges=use_min_edges
        self.num_lens = num_lens
        self.threshold = threshold
        self.power=power
        
    def perm_invariant_loss(self, tensor1, tensor2):
        print('tensor1, tensor2: ', tensor1.shape, tensor2.shape)   #!
        dists = torch.cdist(tensor1, tensor2)
        min_dists = dists.min(dim=1)[0]
        return min_dists.mean()

    def symmetric_perm_invariant_loss(self, tensor1, tensor2):
        loss1 = self.perm_invariant_loss(tensor1, tensor2)
        loss2 = self.perm_invariant_loss(tensor2, tensor1)
        return (loss1 + loss2) / 2

    def sgo_loss(self, frac, opr): # can be vectorized for multiple space group opoerations?
        """
        Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
        """
        frac0 = frac.clone()#.detach()
        frac0.requires_grad_()
        frac1 = frac.clone()
        frac1.requires_grad_()
        frac1 = frac1@opr.T%1
        natm = len(frac0)
        print('frac0, frac1: ', frac0.shape, frac1.shape)   #!
        _, _, edge_vec0 = get_neighbors(frac0, self.r_max/pow(natm, self.power))
        _, _, edge_vec1 = get_neighbors(frac1, self.r_max/pow(natm, self.power))
        if self.use_min_edges:
            edge_len0 = edge_vec0.norm(dim=-1)
            edge_len1 = edge_vec1.norm(dim=-1)
            mask0 = edge_len0 > self.threshold
            mask1 = edge_len1 > self.threshold
            edge_len_f0 = edge_len0[mask0]
            edge_len_f1 = edge_len1[mask1]
            if self.num_lens>1:
                target_lens0 = torch.sort(torch.unique(edge_len_f0)).values[:self.num_lens]
                target_lens1 = torch.sort(torch.unique(edge_len_f1)).values[:self.num_lens]
                indices0, indices1 = [], []
                for target_len in target_lens0:
                    # print(target_len.shape)
                    # print(torch.where(torch.eq(edge_len_f0, target_len.reshape(1,-1))))
                    target_indices = torch.where(torch.eq(edge_len_f0, target_len.reshape(1,-1)))[-1]
                    indices0.append(target_indices)
                for target_len in target_lens1:
                    # print(target_len)
                    target_indices = torch.where(torch.eq(edge_len_f1, target_len.reshape(1,-1)))[-1]
                    indices1.append(target_indices)
                indices0, indices1 = torch.cat(indices0), torch.cat(indices1)
                # print('indices0: ', indices0)
                # print('indices1: ', indices1)
                idx_edge_min0 = torch.tensor(indices0)
                idx_edge_min1 = torch.tensor(indices1)
            else:
                idx_edge_min0 = torch.nonzero(edge_len_f0 == edge_len_f0.min()).flatten()
                idx_edge_min1 = torch.nonzero(edge_len_f1 == edge_len_f1.min()).flatten()
            edge_vec0 = edge_vec0[idx_edge_min0]
            edge_vec1 = edge_vec1[idx_edge_min1]
        return self.symmetric_perm_invariant_loss(edge_vec0, edge_vec1)


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

# 230714
class SGO_Loss_Assign(SGO_Loss):
    def __init__(self, r_max) -> None:
        super().__init__()
        self.r_max = r_max
        
    def perm_invariant_loss_assign(self, tensor1, tensor2):
        dists = torch.cdist(tensor1, tensor2)
        row_indices, col_indices = self.linear_sum_assignment_torch(dists)
        return torch.sum(dists[row_indices, col_indices])
        # min_dists = dists.min(dim=1)[0]
        # return min_dists.mean()

    def symmetric_perm_invariant_loss_assign(self, tensor1, tensor2):
        loss1 = self.perm_invariant_loss_assign(tensor1, tensor2)
        loss2 = self.perm_invariant_loss_assign(tensor2, tensor1)
        return (loss1 + loss2) / 2


    def sgo_loss_perm_assign(self, frac, opr): # can be vectorized for multiple space group opoerations?
        """
        Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
        """
        frac0 = frac.clone()#.detach()
        frac0.requires_grad_()
        frac1 = frac.clone()
        frac1.requires_grad_()
        frac1 = frac1@opr.T%1
        _, _, edge_vec0 = get_neighbors(frac0, self.r_max)
        _, _, edge_vec1 = get_neighbors(frac1, self.r_max)
        return self.symmetric_perm_invariant_loss_assign(edge_vec0, edge_vec1)

    def sgo_cum_loss_perm_assign(self, frac, oprs):
        """
        Cumulative loss from space group operations.
        """
        loss = torch.zeros(1).to(frac)
        # loss.requires_grad=True
        nops = len(oprs)
        for opr in oprs:
            diff = self.sgo_loss_perm_assign(frac, opr)
            loss += diff
        return loss/nops

    def linear_sum_assignment_torch(cost_matrix):
        """
        Solve the linear sum assignment problem using the Hungarian algorithm for a distance matrix represented by a torch.tensor.

        Args:
            cost_matrix (torch.tensor): The distance matrix representing the costs or profits of assigning agents to tasks.

        Returns:
            tuple: A tuple containing two torch.tensor arrays representing the row indices and column indices of the optimal assignments.
        """
        # Convert the cost matrix to a numpy array
        cost_matrix_np = cost_matrix.detach().numpy()

        # Import the linear_sum_assignment function from scipy.optimize
        from scipy.optimize import linear_sum_assignment

        # Solve the linear sum assignment problem using linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix_np)

        # Convert the row indices and column indices to torch.tensor
        row_indices_torch = torch.from_numpy(row_indices)
        col_indices_torch = torch.from_numpy(col_indices)

        return row_indices_torch, col_indices_torch


# OT
import ot

# class SGO_Loss_OT(torch.nn.Module):
#     def __init__(self, r_max) -> None:
#         super().__init__()
def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached


def sgo_loss_ot1(frac, opr, r_max=None, use_min_edges=False, num_lens=1, threshold=1e-3): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    ot_mat = compute_sq_dist_mat(frac0, frac1)
    return compute_ot_emd(ot_mat, device='cpu')[0]

def sgo_cum_loss_ot1(frac, oprs, r_max):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_ot1(frac, opr, r_max)
        loss += diff
    return loss/nops

def sgo_loss_ot2(frac, opr, r_max): # can be vectorized for multiple space group opoerations?
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
    ot_mat = compute_sq_dist_mat(edge_vec0, edge_vec1)
    return compute_ot_emd(ot_mat, device='cpu')[0]

def sgo_cum_loss_ot2(frac, oprs, r_max):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_ot2(frac, opr, r_max)
        loss += diff
    return loss/nops



def sgo_loss_gw(frac, opr): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    p, q = ot.unif(len(frac0)), ot.unif(len(frac1))
    p, q = torch.tensor(p, requires_grad=False), torch.tensor(q, requires_grad=False)
    C0 = torch.cdist(frac0, frac0, p=2)
    C1 = torch.cdist(frac1, frac1, p=2)
    gw, log = ot.gromov.gromov_wasserstein2(
        C0, C1, p, q, 'square_loss', verbose=False, log=True)
    return gw


def sgo_cum_loss_gw(frac, oprs):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_gw(frac, opr)
        loss += diff
    return loss/nops


