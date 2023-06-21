#%%
'''
https://www.notion.so/230416-space-group-operation-loss-c6e8cbc2e93f45f69982fe5a3ccec4f1
'''
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

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))

#%%
# cosn
cosn = mpdata['mp-20536']
silicon = mpdata['mp-149']
frac = torch.tensor(cosn.frac_coords)
lat = torch.tensor(cosn.lattice.matrix)
spec = cosn.species
natms = len(frac)
smatcher = StructureMatcher()

#%%
# get shift indices of the 1st nearest neighbor cells
shiftids = torch.tensor(list(itertools.product([0,1,-1], repeat=3)))
shifts = shiftids.repeat_interleave(natms, dim=0)


#%%

#fractional cell with the shift
frac_ex = torch.cat([frac for _ in range(27)], dim=0) + shifts 


#%%
# dist matrix
dmatrix = torch.cdist(frac, frac_ex, p=2)


#%%
# mask matrix, get nonzero indices
r_max = 0.5
mask = dmatrix<=r_max
idx_mask = torch.nonzero(mask)
idx_src, idx_dst = idx_mask[:,0].reshape(-1), idx_mask[:,1].reshape(-1)

#%% 
# use the indices to get the list of vectors
frac_src = frac_ex[idx_src, :]
frac_dst = frac_ex[idx_dst, :]
vectors = frac_dst-frac_src
vmask = torch.norm(vectors, dim=-1)>tol

#%%
# functionalize it!

def get_neighbors(frac, r_max):
    # frac = torch.tensor(frac, requires_grad=grad)
    natms = len(frac)
    # get shift indices of the 1st nearest neighbor cells
    shiftids = torch.tensor(list(itertools.product([0,1,-1], repeat=3)))
    shifts = shiftids.repeat_interleave(natms, dim=0)
    #fractional cell with the shift
    frac_ex = torch.cat([frac for _ in range(27)], dim=0) + shifts 
    # dist matrix
    dmatrix = torch.cdist(frac, frac_ex, p=2)
    # print(dmatrix)
    # dmatrix.backward()
    # print('dmatrix: ', dmatrix.grad)
    # mask matrix, get nonzero indices
    mask = dmatrix<=r_max
    idx_mask = torch.nonzero(mask)
    idx_src, idx_dst = idx_mask[:,0].reshape(-1), idx_mask[:,1].reshape(-1)
    # use the indices to get the list of vectors
    frac_src = frac_ex[idx_src, :]
    frac_dst = frac_ex[idx_dst, :]
    vectors = frac_dst-frac_src
    return idx_src, idx_dst, vectors


def sgo_loss(frac, opr, r_max): # can be vectorized for multiple space group opoerations?
    # frac = frac.clone().detach().requires_grad_(grad)
    # opr = opr.clone().detach().requires_grad_(grad)
    frac0 = frac
    frac1 = frac@opr.T%1
    _, _, edge_vec0 = get_neighbors(frac0, r_max)
    _, _, edge_vec1 = get_neighbors(frac1, r_max)
    # assert len(edge_vec0)==len(edge_vec1)
    # W = torch.rand(3, 10).to(edge_vec0)
    # wvec0 = edge_vec0@W
    # wvec1 = edge_vec1@W
    # out0 = torch.sum(wvec0, dim=0)
    # out1 = torch.sum(wvec1, dim=0)
    wvec0 = edge_vec0*edge_vec0
    wvec1 = edge_vec1*edge_vec1
    out0 = wvec0.sum(dim=0)
    out1 = wvec1.sum(dim=0)
    diff= out1 - out0
    return diff.norm()

def sgo_cum_loss(frac, oprs, r_max):
    loss = torch.zeros(1)
    nops = len(oprs)
    for opr in oprs:
        diff = sgo_loss(frac, opr, r_max)
        loss += diff
    return loss/nops

def J_sgo_cum_loss(frac, oprs, r_max):
    """
    Jacobian with the autograd
    """
    def sgo_func(frac):
        return sgo_cum_loss(frac, oprs, r_max)
    J = jacobian(sgo_func, frac)
    return J

def J_sgo_cum_num(frac, oprs, r_max, h=1e-03):
    """
    Jacobian with numerical diff
    """
    out = torch.zeros_like(frac)
    n = out.shape[0]
    for i in range(n):
        for j in range(3):
            fr1, fr2 = frac, frac
            sub = torch.zeros(frac.shape)
            sub[i, j] = 1
            fr1 = fr1 - h*sub
            fr2 = fr2 + h*sub
            # print('fr1 aft: ', fr1)
            # print('fr2 aft: ', fr2)
            loss1 = sgo_cum_loss(fr1, oprs, r_max)
            loss2 = sgo_cum_loss(fr2, oprs, r_max)
            dLdx = 0.5*(loss2-loss1)/h
            # print('[loss1, loss2, dLdx]: ', [loss1, loss2, dLdx])
            out[i, j] = dLdx
    return out

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

#%%
# test the loss function
pstruct = cosn
frac = torch.tensor(pstruct.frac_coords)
frac.requires_grad = True
mt = MatTrans(pstruct)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]

r_max=0.9
sgo = torch.tensor(oprs[0])
frac1 = frac@sgo.T%1
vecs0 = get_neighbors(frac, r_max)[-1]
vecs1 = get_neighbors(frac1, r_max)[-1]
loss1 = sgo_loss(frac, sgo, r_max)
loss1.backward()
print(loss1)
print(loss1.grad)
print(frac.grad)

#%%
pstruct = cosn
r_max=0.8
grad=True
frac = torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
frac.requires_grad = grad
mt = MatTrans(pstruct)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
sgo = [torch.tensor(opr, requires_grad=False) for opr in oprs]
loss2 = sgo_cum_loss(frac, sgo, r_max)
print(loss2)
loss2.backward()
print(loss2.grad)
print(frac.grad)
J = J_sgo_cum_loss(frac, sgo, r_max)
print('J (auto)', J)
Jnum = J_sgo_cum_num(frac, sgo, r_max, h=1e-06)
print('J (num)', Jnum)
if torch.allclose(J, frac.grad, atol=tol):
    print()

#%%
# with grad
# test with the diffused structures
logvars = np.linspace(-3, 0, num=31) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys = []
pstruct = cosn  #mpdata['mp-1003']#kstruct
mt = MatTrans(pstruct)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
sgo = [torch.tensor(opr, requires_grad=False) for opr in oprs]
natms = len(mt.pstruct.sites)
nops = len(opes)
r_max = 0.7
h = 1e-07
tol = 1e-03
grad=True
for i, l in enumerate(logvars):
    sigma = 10**l
    dstruct = diffuse_frac(pstruct, sigma=sigma)
    # vis_structure(dstruct, title=f"$log(\sigma)$={round(l, 7)}")
    # plt.show()
    # plt.close()
    frac = torch.tensor(dstruct.frac_coords)
    frac.requires_grad = grad
    loss = sgo_cum_loss(frac, sgo, r_max)
    ys.append(loss)
    sgo = [torch.tensor(opr, requires_grad=False) for opr in oprs]
    def sgo1(frac):
        return sgo_cum_loss(frac, sgo, r_max)
    J = jacobian(sgo1, frac)
    Jnum = J_sgo_cum_num(frac, sgo, r_max, h=1e-06)
    print(f'-----[{i}]------')
    if torch.allclose(J, Jnum, atol=tol):
        print('J~Jnum: ', J)
    else:
        print('J', J)
        print('Jnum', Jnum)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ys = [y.detach().numpy() for y in ys]
ax.plot(logvars, ys)
ax.set_ylabel(f"Mismatch term by space group operation")
ax.set_xlabel(f"$log(\sigma)$")
ax.set_title(pstruct.formula)
# ax.set_yscale('log')
fig.patch.set_facecolor('white')

#%%
# # test with the diffused structures
# logvars = np.linspace(-3, 0, num=31) #range(10, -5, -1)
# xs = [10**l for l in logvars]
# ys = []
# pstruct = cosn #mpdata['mp-1000']#kstruct
# mt = MatTrans(pstruct)
# opes = list(set(mt.spgops))
# oprs = [op.rotation_matrix for op in opes]
# opts = [op.translation_vector for op in opes]
# natms = len(mt.pstruct.sites)
# nops = len(opes)
# r_max = 0.7
# for l in logvars:
#     sigma = 10**l
#     dstruct = diffuse_frac(pstruct, sigma=sigma)
#     vis_structure(dstruct, title=f"$log(\sigma)$={round(l, 7)}")
#     plt.show()
#     plt.close()
#     frac = torch.tensor(dstruct.frac_coords)
#     loss = sgo_cum_loss(frac, oprs, r_max)
#     ys.append(loss)

# fig, ax = plt.subplots(1,1,figsize=(8,8))
# ax.plot(logvars, ys)
# ax.set_ylabel(f"Mismatch term by space group operation")
# ax.set_xlabel(f"$log(\sigma)$")
# ax.set_title(pstruct.formula)
# # ax.set_yscale('log')
# fig.patch.set_facecolor('white')

#%%
# 230620 loss with permutation invarinace

def perm_invariant_loss(tensor1, tensor2):
    dists = torch.cdist(tensor1, tensor2)
    min_dists = dists.min(dim=1)[0]
    return min_dists.mean()

def symmetric_perm_invariant_loss(tensor1, tensor2):
    loss1 = perm_invariant_loss(tensor1, tensor2)
    loss2 = perm_invariant_loss(tensor2, tensor1)
    return (loss1 + loss2) / 2

#%%