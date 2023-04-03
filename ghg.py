#%%
"""
https://www.notion.so/230328-29-make-filters-of-symmops-e31d24bb7ede445aac3ba362e132c151
"""

from pathlib import Path
from typing import List

import torch

from torch_geometric.data import Batch

from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import pickle as pkl
from ase import Atoms

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from itertools import combinations

from scripts.eval_utils import load_model
from pymatgen.core.lattice import Lattice
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
from itertools import permutations
import imageio
import os, sys

#%%

mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))

#%%
pstruct = mpdata['mp-2654']  #['mp-20536']
kfrac = np.array([
    [0,0,0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0]
])#np.random.rand(N,3)
# N = len(hfrac)
hex_latice = np.array([
    [6,0,0],
    [3, 3*np.sqrt(3),0],
    [0,0,4]
])
kspecies = np.array([6, 6, 6])  
kspecies1 = np.array([6, 7, 8])
kstruct = Structure(
    lattice=hex_latice,
    species=kspecies,
    coords=kfrac,
    coords_are_cartesian=False
)

#%%
pstruct = mpdata['mp-20536']
# pstruct = mpdata['mp-1003319']
struct_in = pstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
frac = struct_in.frac_coords


#%%
# def ghg(pstruct, ope, skip_tau=True):
#     # mt = MatTrans(pstruct)
#     lat = np.array(pstruct.lattice.matrix)
#     cart = pstruct.cart_coords
#     frac = pstruct.frac_coords
#     spec = pstruct.species
#     opr = ope.rotation_matrix
#     opt = ope.translation_vector
#     if skip_tau:
#         ghg = opr@



#%%
frac = pstruct.frac_coords
opr = oprs[0]
frac1 = np.einsum('ij, kj-> ki', opr, frac@np.linalg.inv(opr.T))
np.allclose(frac, frac1, atol=tol)



#%%
cosn = mpdata['mp-20536']
struct_in = cosn #mpdata['mp-20536'] # mpdata['mp-2534']   #kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
idx = 0
opr = oprs[idx]
opt = opts[idx]
frac = struct_in.frac_coords
A = frac
B = (frac@opr.T)%1
C = (frac@opr.T  + opt)%1
lenf, d = frac.shape
fmatrix1 = np.zeros((lenf, lenf, d))
fmatrix2 = np.zeros((lenf, lenf, d))
for i in range(lenf):
    for j in range(lenf):
        fmatrix1[i,j,:] = A[i]-B[j]
        fmatrix2[i,j,:] = A[i]-C[j]
fmatrix1

#%%
cosn = mpdata['mp-20536']
#kstruct1: original
kstruct1 = kstruct

# kstruct2: shift kstruct1 by [0.5, 0, 0]
kfrac2 = np.array([
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.0]
])
kspecies = np.array([6,6,6])  
kstruct2 = Structure(
    lattice=hex_latice,
    species=kspecies,
    coords=kfrac2,
    coords_are_cartesian=False
)

# kstruct3: kagome lattice following CoSn
kfrac3 = np.array([
    [0,0.5,0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.0]
])#np.random.rand(N,3)
# N = len(hfrac)
hex_latice3 = np.array([
    [6,0,0],
    [-3, 3*np.sqrt(3),0],
    [0,0,4]
])
kspecies = np.array([6, 6, 6])  
kstruct3 = Structure(
    lattice=hex_latice3,
    species=kspecies,
    coords=kfrac3,
    coords_are_cartesian=False
)


#%%

struct_in = kstruct1 # mpdata['mp-2534']   #kstruct
mt1 = MatTrans(struct_in)
opes1 = list(set(mt1.spgops))
oprs1 = [op.rotation_matrix for op in opes1]
opts1 = [op.translation_vector for op in opes1]


struct_in = kstruct2 # mpdata['mp-2534']   #kstruct
mt2 = MatTrans(struct_in)
opes2 = list(set(mt2.spgops))
oprs2 = [op.rotation_matrix for op in opes2]
opts2 = [op.translation_vector for op in opes2]


struct_in = kstruct3 # mpdata['mp-2534']   #kstruct
mt3 = MatTrans(struct_in)
opes3 = list(set(mt3.spgops))
oprs3 = [op.rotation_matrix for op in opes3]
opts3 = [op.translation_vector for op in opes3]

struct_in = cosn # mpdata['mp-2534']   #kstruct
mtx = MatTrans(struct_in)
opesx = list(set(mtx.spgops))
oprsx = [op.rotation_matrix for op in opesx]
optsx = [op.translation_vector for op in opesx]

#%%

struct_in = cosn
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opr = oprs[0]
frac = struct_in.frac_coords
A = frac
B = (frac@opr.T)%1
lenf, d = frac.shape
fmatrix = np.zeros((lenf, lenf))
for i in range(lenf):
    for j in range(lenf):
        fmatrix[i,j] = np.linalg.norm(A[i]-B[j])
fmatrix
np.isclose(fmatrix, np.zeros_like(fmatrix), atol=1e-2)

#%%
struct_in = kstruct2
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opr = oprs[0]
frac = struct_in.frac_coords
A = frac
B = (frac@opr.T)%1
lenf, d = frac.shape
dmatrix1, dmatrix2 = np.zeros((lenf, lenf)), np.zeros((lenf, lenf))
for i in range(lenf):
    for j in range(lenf):
        dmatrix1[i,j] = np.linalg.norm(A[i]-A[j])
        dmatrix2[i,j] = np.linalg.norm(B[i]-B[j])
        
        
#%% 
def operation_identity(struct_in, opr, tol, perm=True, get_diff=False):
    frac = struct_in.frac_coords
    A = frac
    B = (frac@opr.T)%1
    lenf, d = frac.shape
    dmatrix1, dmatrix2 = np.zeros((lenf, lenf)), np.zeros((lenf, lenf))
    for i in range(lenf):
        for j in range(lenf):
            dmatrix1[i,j] = np.linalg.norm(A[i]-A[j])
            dmatrix2[i,j] = np.linalg.norm(B[i]-B[j])
    assert dmatrix1.shape == dmatrix2.shape
    d = dmatrix1.shape[0]
    dm1 = dmatrix1[np.argsort(dmatrix1[0]), :][:, np.argsort(dmatrix1[0])]
    diffs = []
    if perm:
        for pm in permutations(range(d)):
            dm2 = dmatrix2[pm, :][:, pm]
            # print(f"[{i}] {dm2}")
            if get_diff:
                diffs.append(np.linalg.norm(dm1-dm2))
            else: 
                if np.allclose(dm1, dm2, atol=tol):
                    return True
    else:
        for i in range(d):
            dm2 = dmatrix2[np.argsort(dmatrix2[i]), :][:, np.argsort(dmatrix2[i])]
            # print(f"[{i}] {dm2}")
            if get_diff:
                diffs.append(np.linalg.norm(dm1-dm2))
            else: 
                if np.allclose(dm1, dm2, atol=tol):
                    return True
    if get_diff:
        return np.min(diffs)
    else: 
        return False


def dmatrices_identity(dmatrix1, dmatrix2, tol):
    assert dmatrix1.shape == dmatrix2.shape
    d = dmatrix1.shape[0]
    dm1 = dmatrix1[np.argsort(dmatrix1[0]), :][:, np.argsort(dmatrix1[0])]
    for i in range(d):
        dm2 = dmatrix2[np.argsort(dmatrix2[i]), :][:, np.argsort(dmatrix2[i])]
        # print(f"[{i}] {dm2}")
        if np.allclose(dm1, dm2, atol=tol):
            return True
    return False
#%%

# symmorphic_sg = []    
# find materials that have all zero translation vectors.    
tol = 1e-01     
mpids = sorted(list(mpdata.keys()))   
#%%
mpids_symmo = []
for mpid in mpids: 
    struct_in = mpdata[mpid]
    mt = MatTrans(struct_in)
    opes = list(set(mt.spgops))
    oprs = [op.rotation_matrix for op in opes]
    opts = [op.rotation_matrix for op in opes]
    opts_norm = np.array([np.linalg.norm(opt) for opt in opts])
    if np.all(opts_norm<tol):
        print(f"[{mpid}] symmorphic!")
        mpids_symmo.append(mpid)
    else:
        print(f"({mpid}) exclude")

mp_symmo = {}   
for mpid in mpids_symmo:
    pstruct = mpdata[mpid]
    mp_symmo[mpid]=pstruct

pkl.dump(mp_symmo, open('data/mp_symmo.pkl', 'wb'))
print(len(mp_symmo))


#%%
# use multiple oprs, make it vectors.
def diff_with_opes(struct_in, oprs):
    vector = []
    for opr in oprs:
        diff = operation_identity(struct_in, opr, tol, perm=True, get_diff=True)
        vector.append(diff)
    return np.array(vector)

vector = diff_with_opes(cosn, oprsx)
vector


#%%
# disturb the structures
def disturb_frac(struct_in, sigma=0.1 , random=True, different=True):
    frac = struct_in.frac_coords
    lat = struct_in.lattice.matrix
    spec = struct_in.species
    dist = np.random.normal(loc=0.0, scale=sigma, size=frac.shape)
    frac1 = frac + dist
    struct_out = Structure(
    lattice=lat,
    species=spec,
    coords=frac1,
    coords_are_cartesian=False
    )
    return struct_out

sigma = 0.001
cosn_d = disturb_frac(cosn, sigma=sigma, random=True, different=True)
mtd = MatTrans(cosn_d)
vis_structure(cosn, title=f"original")
vis_structure(cosn_d, title=f"sigma={sigma}")
vector_d = diff_with_opes(cosn_d, oprsx)
vector_d

#%%
# compare values
logvars = range(-1, -30, -1)
xs = [10**l for l in logvars]
ys = []
for l in logvars:
    sigma = 10**l
    dstruct = disturb_frac(cosn, sigma=sigma, random=True, different=True)
    dvec = diff_with_opes(dstruct, oprsx)
    ys.append(np.linalg.norm(dvec))

plt.plot(logvars, ys)

#%%%
# noising process
# https://www.notion.so/230331-noising-process-to-check-the-space-group-filters-12452baac0c348e3b3c14bf23fa1471f#f244b74e130b405aad369408870cd65a

def diffusion(struct_in, betas, visualize=True):
    struct_list = [struct_in]
    for i, b in enumerate(betas):
        lat = struct_in.lattice.matrix
        spec = struct_in.species
        frac = struct_in.frac_coords
        natms = frac.shape[0]
        dist = np.random.normal(loc=0.0, scale=b*0.07, size=frac.shape)
        frac_d = frac + dist
        struct_in = Structure(
        lattice=lat,
        species=spec,
        coords=frac_d,
        coords_are_cartesian=False
        )
        struct_list.append(struct_in)
        if visualize:
            if i%10==0:
                vis_structure(struct_in, supercell=np.diag([3,3,1]), title=f"[{i}] beta = {b}")
    return struct_list


def movie_noising(struct_list, titles=None, savedir=None, supercell=np.diag([3,3,1])):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, struct_in in enumerate(struct_list):
        vis_structure(struct_in, supercell, title=f"{{0:04d}}".format(i), savedir=savedir)
    
    with imageio.get_writer(os.path.join(savedir, 'diffusion.gif'), mode='I') as writer:
        for figurename in sorted(os.listdir(savedir)):
            if figurename.endswith('png'):
                image = imageio.imread(os.path.join(savedir, figurename))
                writer.append_data(image)

#%%
betas = [np.random.uniform(0, 1) for _ in range(300)]
struct_in = cosn.copy()
struct_list = diffusion(struct_in, betas, visualize=False)
savedir = "/home/rokabe/data2/generative/magcdvae/figures/cosn1"
movie_noising(struct_list, titles=None, savedir=savedir, supercell=np.diag([3,3,1]))

