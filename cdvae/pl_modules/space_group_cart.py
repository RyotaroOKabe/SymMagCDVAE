from typing import Any, Dict
import numpy as np
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
from cdvae.pl_modules.space_group import get_neighbors, symmetric_perm_invariant_loss

# get_neighbor → frac2cart for fractional edge vector. 
def sgo_loss_cart(frac, opr, r_max, lattice): # can be vectorized for multiple space group opoerations?
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
    edge_vec_cart0 = torch.einsum('bi, ij->bj', edge_vec0, lattice)
    edge_vec_cart1 = torch.einsum('bi, ij->bj', edge_vec1, lattice)
    wvec0 = edge_vec_cart0*edge_vec_cart0
    wvec1 = edge_vec_cart1*edge_vec_cart1
    out0 = wvec0.sum(dim=0)
    out1 = wvec1.sum(dim=0)
    diff= out1 - out0
    return diff.norm()

def sgo_cum_loss_cart(frac, oprs, r_max, lattice):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_cart(frac, opr, r_max, lattice)
        loss += diff
    return loss/nops

def sgo_loss_perm_cart(frac, opr, r_max, lattice, use_min_edges=False, num_lens=1, threshold=1e-3): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    _, _, edge_vec0 = get_neighbors(frac0, r_max)
    _, _, edge_vec1 = get_neighbors(frac1, r_max)
    edge_vec_cart0 = torch.einsum('bi, ij->bj', edge_vec0, lattice)
    edge_vec_cart1 = torch.einsum('bi, ij->bj', edge_vec1, lattice)
    if use_min_edges:
        edge_len0 = edge_vec_cart0.norm(dim=-1)
        edge_len1 = edge_vec_cart1.norm(dim=-1)
        mask0 = edge_len0 > threshold
        mask1 = edge_len1 > threshold
        edge_len_f0 = edge_len0[mask0]
        edge_len_f1 = edge_len1[mask1]
        if num_lens>1:
            target_lens0 = torch.sort(torch.unique(edge_len_f0)).values[:num_lens]
            target_lens1 = torch.sort(torch.unique(edge_len_f1)).values[:num_lens]
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
        edge_vec_cart0 = edge_vec_cart0[idx_edge_min0]
        edge_vec_cart1 = edge_vec_cart1[idx_edge_min1]
    return symmetric_perm_invariant_loss(edge_vec_cart0, edge_vec_cart1)

def sgo_cum_loss_perm_cart(frac, oprs, r_max, lattice):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_perm_cart(frac, opr, r_max, lattice)
        loss += diff
    return loss/nops

# get_neighbor for cart coordinate
def get_neighbors_cart(frac, r_max, lattice):
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
    cart = torch.einsum('bi, ij->bj', frac, lattice)
    cart_ex = torch.einsum('bi, ij->bj', frac_ex, lattice)
    r_max_c = r_max * torch.min(torch.norm(lattice, dim=-1))
    # frac_ex.requires_grad=True
    # dist matrix
    dmatrix = torch.cdist(cart, cart_ex, p=2)
    mask = dmatrix<=r_max_c
    idx_mask = torch.nonzero(mask)
    idx_src, idx_dst = idx_mask[:,0].reshape(-1), idx_mask[:,1].reshape(-1)
    # use the indices to get the list of vectors
    cart_src = cart_ex[idx_src, :]
    cart_dst = cart_ex[idx_dst, :]
    vectors = cart_dst-cart_src
    return idx_src, idx_dst, vectors

# frac2cart→ get_neighbor  → we need another function of get_neighbor for cart coordinates
def sgo_loss_cart2(frac, opr, r_max, lattice): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    # cart0 = torch.einsum('bi, ij->bj', frac0, lattice)
    # cart1 = torch.einsum('bi, ij->bj', frac1, lattice)
    _, _, edge_vec_cart0 = get_neighbors_cart(frac0, r_max, lattice)
    _, _, edge_vec_cart1 = get_neighbors_cart(frac1, r_max, lattice)
    wvec0 = edge_vec_cart0*edge_vec_cart0
    wvec1 = edge_vec_cart1*edge_vec_cart1
    out0 = wvec0.sum(dim=0)
    out1 = wvec1.sum(dim=0)
    diff= out1 - out0
    return diff.norm()

def sgo_cum_loss_cart2(frac, oprs, r_max, lattice):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_cart2(frac, opr, r_max, lattice)
        loss += diff
    return loss/nops


def sgo_loss_perm_cart2(frac, opr, r_max, lattice, use_min_edges=False, num_lens=1, threshold=1e-3): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    _, _, edge_vec_cart0 = get_neighbors_cart(frac0, r_max, lattice)
    _, _, edge_vec_cart1 = get_neighbors_cart(frac1, r_max, lattice)
    if use_min_edges:
        edge_len0 = edge_vec_cart0.norm(dim=-1)
        edge_len1 = edge_vec_cart1.norm(dim=-1)
        mask0 = edge_len0 > threshold
        mask1 = edge_len1 > threshold
        edge_len_f0 = edge_len0[mask0]
        edge_len_f1 = edge_len1[mask1]
        if num_lens>1:
            target_lens0 = torch.sort(torch.unique(edge_len_f0)).values[:num_lens]
            target_lens1 = torch.sort(torch.unique(edge_len_f1)).values[:num_lens]
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
        edge_vec_cart0 = edge_vec_cart0[idx_edge_min0]
        edge_vec_cart1 = edge_vec_cart1[idx_edge_min1]
    return symmetric_perm_invariant_loss(edge_vec_cart0, edge_vec_cart1)

def sgo_cum_loss_perm_cart2(frac, oprs, r_max, lattice):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_perm_cart2(frac, opr, r_max, lattice)
        loss += diff
    return loss/nops



# gromov_wasserstein2 with cartesian coordinates
def sgo_loss_gw_cart(frac, opr, lattice): # can be vectorized for multiple space group opoerations?
    """
    Space group loss: The larger this loss is, the more the structure is apart from the given space group. 
    """
    frac0 = frac.clone()#.detach()
    frac0.requires_grad_()
    frac1 = frac.clone()
    frac1.requires_grad_()
    frac1 = frac1@opr.T%1
    cart0 = torch.einsum('bi, ij->bj', frac0, lattice)
    cart1 = torch.einsum('bi, ij->bj', frac1, lattice)
    p, q = ot.unif(len(cart0)), ot.unif(len(cart1))
    p, q = torch.tensor(p, requires_grad=False), torch.tensor(q, requires_grad=False)
    C0 = torch.cdist(cart0, cart0, p=2)
    C1 = torch.cdist(cart1,cart1, p=2)
    gw, log = ot.gromov.gromov_wasserstein2(
        C0, C1, p, q, 'square_loss', verbose=False, log=True)
    return gw


def sgo_cum_loss_gw_cart(frac, oprs, lattice):
    """
    Cumulative loss from space group operations.
    """
    loss = torch.zeros(1).to(frac)
    # loss.requires_grad=True
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss_gw_cart(frac, opr, lattice)
        loss += diff
    return loss/nops


