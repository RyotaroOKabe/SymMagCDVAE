#%%
'''
https://www.notion.so/230416-space-group-operation-loss-c6e8cbc2e93f45f69982fe5a3ccec4f1
'''
import torch
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle as pkl

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from cdvae.common.data_utils import lattice_params_to_matrix_torch
from cdvae.pl_modules.space_group import *
from torch_geometric.data import Data
from torch.utils.data import Dataset
import math as m

from utils.utils_plot import vis_structure
from utils.utils_material import MatSym, MatTrans, distance_sorted, Rx, Ry, Rz, rotate_cart, switch_latvecs
from utils.plot_3d import create_animation
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

#%%
# test the single loss (w/o cum)
pstruct = cosn
frac = torch.tensor(pstruct.frac_coords)
frac.requires_grad = True
mt = MatTrans(pstruct)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]

r_max=0.8
sgo = torch.tensor(oprs[0])
loss_prod = sgo_loss(frac, sgo, r_max)
loss_perm = sgo_loss_perm(frac, sgo, r_max)


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
oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
frac0 = torch.tensor(pstruct.frac_coords)
frac0.requires_grad = grad
loss_prod = sgo_cum_loss(frac0, oprs, r_max)
loss_prod.backward()
print('loss_prod: ', loss_prod)
print('loss_prod.grad: ', loss_prod.grad)
print('frac0.grad: ', frac0.grad)

frac1 = torch.tensor(pstruct.frac_coords)
frac1.requires_grad = grad
loss_perm = sgo_cum_loss_perm(frac1, oprs, r_max)
loss_perm.backward()
print('loss_perm: ', loss_perm)
print('loss_perm.grad: ', loss_perm.grad)
print('frac1.grad: ', frac1.grad)


#%%
# with grad
# test with the diffused structures
logvars = np.linspace(-2, 0, num=51) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys0 = []
ys1 = []
frac0_list = []
grads0_list = []
frac1_list = []
grads1_list = []
pstruct = cosn  #mpdata['mp-1003']#kstruct
frac_coords0 = pstruct.frac_coords  # ooriginal
mt = MatTrans(pstruct)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
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
    frac0, frac1 = torch.tensor(dstruct.frac_coords), torch.tensor(dstruct.frac_coords)
    frac0.requires_grad, frac1.requires_grad = grad, grad
    loss0 = sgo_cum_loss(frac0, oprs, r_max)
    ys0.append(loss0)
    loss0.backward()
    grads0 = frac0.grad
    frac0_list.append(frac0.detach().numpy())
    grads0_list.append(-grads0.detach().numpy())
    loss1 = sgo_cum_loss_perm(frac1, oprs, r_max)
    ys1.append(loss1)
    loss1.backward()
    grads1 = frac1.grad
    frac1_list.append(frac1.detach().numpy())
    grads1_list.append(-grads1.detach().numpy())
    # plot_3d_vectors(frac_coords0, frac0.detach().numpy(), -grads0.detach().numpy())
    # plot_3d_vectors(frac_coords0, frac1.detach().numpy(), -grads1.detach().numpy())
create_animation(frac_coords0, frac0_list, grads0_list, 'figures/cosn0')
create_animation(frac_coords0, frac1_list, grads1_list, 'figures/cosn1')

fig, ax = plt.subplots(1,1,figsize=(8,8))
ys0 = [y.detach().numpy() for y in ys0]
ys1 = [y.detach().numpy() for y in ys1]
ax.plot(logvars, ys0/max(ys0), label='prod')
ax.plot(logvars, ys1/max(ys1), label='perm')
ax.legend()
ax.set_ylabel(f"Mismatch term by space group operation")
ax.set_xlabel(f"$log(\sigma)$")
ax.set_title(pstruct.formula)
# ax.set_yscale('log')
fig.patch.set_facecolor('white')


#%%