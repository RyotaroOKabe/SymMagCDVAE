#%%
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


#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))

#%%
pstruct = mpdata['mp-1000']
ms = MatSym(pstruct)

#%%
N = 5
rlat = np.random.rand(3,3)*10
rfrac = np.random.rand(N,3)
rspecies = np.array([np.random.randint(30)+1 for _ in range(N)])
rstruct = Structure(
    lattice=rlat,
    species=rspecies,
    coords=rfrac,
    coords_are_cartesian=False
)

hlat = np.array([
    [6,0,0],
    [3, 3*np.sqrt(3),0],
    [0,0,4]
])
hfrac = np.array([
    [0,0,0],
    [0.5, 0.5, 0.5]
])#np.random.rand(N,3)
# N = len(hfrac)
hspecies = np.array([6, 7])   #np.array([6  for _ in range(N)])#np.array([np.random.randint(30)+1 for _ in range(N)])
hstruct = Structure(
    lattice=hlat,
    species=hspecies,
    coords=hfrac,
    coords_are_cartesian=False
)

hfrac0 = np.array([
    [1,0,0]
])
hspecies0 = np.array([6]) 
hstruct0 = Structure(
    lattice=hlat,
    species=hspecies0,
    coords=hfrac0,
    coords_are_cartesian=False
)

pstruct = mpdata['mp-2534']
ms = MatSym(pstruct)
ope = ms.symops[16]
pstructx = pstruct.copy()
pstructx.apply_operation(ope)

lat = pstruct.lattice.matrix
cart = pstruct.cart_coords
frac = pstruct.frac_coords
spec = pstruct.species

pstruct_art = Structure(
    lattice=lat,
    species=spec,
    coords=frac,
    coords_are_cartesian=False
)


kfrac = np.array([
    [0,0,0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0]
])#np.random.rand(N,3)
# N = len(hfrac)
kspecies = np.array([6, 6, 6])  
kspecies1 = np.array([6, 7, 8])
kstruct = Structure(
    lattice=hlat,
    species=kspecies,
    coords=kfrac,
    coords_are_cartesian=False
)
kstruct1 = Structure(
    lattice=hlat,
    species=kspecies1,
    coords=kfrac,
    coords_are_cartesian=False
)

ksqlat = np.array([[3,0,0], [0,3,0],[0,0,4]])
kfrac = np.array([
    [0,0,0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0]
])#np.random.rand(N,3)
# N = len(hfrac)
kspecies = np.array([6, 6, 6])  
ksqstruct = Structure(
    lattice=ksqlat,
    species=kspecies,
    coords=kfrac,
    coords_are_cartesian=False
)

#%%
pstruct = mpdata['mp-20536']  #['mp-20536']  #['mp-2534']
ms = MatSym(kstruct)
opes = list(set(ms.spgops))
n_ops = len(opes)
ope_struct = []
for i in range(n_ops):
    ope = opes[i]
    opr = ope.rotation_matrix
    opt = ope.translation_vector
    
    # original structure info
    lat = ms.pstruct.lattice.matrix
    cart = ms.pstruct.cart_coords
    frac = ms.pstruct.frac_coords
    spec = ms.pstruct.species

    # pstructx: apply operation to cart coords with apply_operation method
    pstructx = ms.pstruct.copy()
    pstructx.apply_operation(ope)   # this operation might not be reliable
    latx = pstructx.lattice.matrix
    cartx = pstructx.cart_coords
    fracx = pstructx.frac_coords

    # pstructy: apply operation to frac coords
    laty = (np.eye(3)@opr.T)@lat
    fracy = frac@opr.T + opt # frac coords with respect to the original structure's lattice vectors
    carty = fracy@lat
    fracyy = carty@np.linalg.inv(laty) # frac coords with respect to the original structure's lattice vectors
    pstructy = Structure(
        lattice=laty,
        species=spec,
        coords=carty,
        coords_are_cartesian=True
    )

    ope_struct.append([ope, pstructy])
    frac_diff = frac@opr.T - frac

    # print([i], ope)
    # print('vec', frac_diff)
    print(pstructy.cart_coords)


#%%
struct_in = kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
n_ops = len(opes)
ope_struct, ope_structx, ope_structy = [], [], []
for i in range(n_ops):
    ope = opes[i]
    mt.transform1(ope)
    mt.transform2(ope, translation=False)
    ope_struct.append([ope, mt.pstruct])
    ope_structx.append([ope, mt.pstruct1])
    ope_structy.append([ope, mt.pstruct2])


struct_in = kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
n_ops = len(opes)
ope_struct, ope_struct1 = [], []
ope_struct2_0, ope_struct2_1, ope_struct2_2, ope_struct2_3 = [], [], [], []
count1 = 0
count2_0, count2_1, count2_2, count2_3 = 0, 0, 0, 0
for i in range(n_ops):
    ope = opes[i]
    mt.transform1(ope)
    ope_struct.append([ope, mt.pstruct])
    c_dists, f_dists = distance_sorted(mt.pstruct)
    ope_struct1.append([ope, mt.pstruct1])
    c_dists1, f_dists1 = distance_sorted(mt.pstruct1)
    mt.transform2(ope, new_lat=True, translation=True)
    ope_struct2_0.append([ope, mt.pstruct2])
    c_dists2_0, f_dists2_0 = distance_sorted(mt.pstruct2)
    mt.transform2(ope, new_lat=True, translation=False)
    ope_struct2_1.append([ope, mt.pstruct2])
    c_dists2_1, f_dists2_1 = distance_sorted(mt.pstruct2)
    mt.transform2(ope, new_lat=False, translation=True)
    ope_struct2_2.append([ope, mt.pstruct2])
    c_dists2_2, f_dists2_2 = distance_sorted(mt.pstruct2)
    mt.transform2(ope, new_lat=False, translation=False)
    ope_struct2_3.append([ope, mt.pstruct2])
    c_dists2_3, f_dists2_3 = distance_sorted(mt.pstruct2)
    print('geometry) pstruct==pstruct1: ',  int(np.allclose(c_dists[:,0], c_dists1[:,0], atol=1e-03)))
    print('geometry) pstruct==pstruct2_0: ',  int(np.allclose(c_dists[:,0], c_dists2_0[:,0], atol=1e-03)))
    print('geometry) pstruct==pstruct2_1: ',  int(np.allclose(c_dists[:,0], c_dists2_1[:,0], atol=1e-03)))
    print('geometry) pstruct==pstruct2_2: ',  int(np.allclose(c_dists[:,0], c_dists2_2[:,0], atol=1e-03)))
    print('geometry) pstruct==pstruct2_3: ',  int(np.allclose(c_dists[:,0], c_dists2_3[:,0], atol=1e-03)))
    count1 += int(np.allclose(c_dists[:,0], c_dists1[:,0], atol=1e-03))
    count2_0 += int(np.allclose(c_dists[:,0], c_dists2_0[:,0], atol=1e-03))
    count2_1 += int(np.allclose(c_dists[:,0], c_dists2_1[:,0], atol=1e-03))
    count2_2 += int(np.allclose(c_dists[:,0], c_dists2_2[:,0], atol=1e-03))
    count2_3 += int(np.allclose(c_dists[:,0], c_dists2_3[:,0], atol=1e-03))
print('count1: ', 100*count1/n_ops)
print('count2_0: ', 100*count2_0/n_ops)
print('count2_1: ', 100*count2_1/n_ops)
print('count2_2: ', 100*count2_2/n_ops)
print('count2_3: ', 100*count2_3/n_ops)

#%%
# use distance matrix instead
struct_in = pstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
n_ops = len(opes)
ope_struct, ope_struct1 = [], []
ope_struct2_0, ope_struct2_1, ope_struct2_2, ope_struct2_3 = [], [], [], []
count1 = 0
count2_0, count2_1, count2_2, count2_3 = 0, 0, 0, 0
for i in range(n_ops):
    ope = opes[i]
    mt.transform1(ope)
    ope_struct.append([ope, mt.pstruct])
    dmatrix = mt.pstruct.distance_matrix
    ope_struct1.append([ope, mt.pstruct1])
    dmatrix1 = mt.pstruct1.distance_matrix
    mt.transform2(ope, new_lat=True, translation=True)
    ope_struct2_0.append([ope, mt.pstruct2])
    dmatrix2_0 = mt.pstruct2.distance_matrix
    mt.transform2(ope, new_lat=True, translation=False)
    ope_struct2_1.append([ope, mt.pstruct2])
    dmatrix2_1 = mt.pstruct2.distance_matrix
    mt.transform2(ope, new_lat=False, translation=True)
    ope_struct2_2.append([ope, mt.pstruct2])
    dmatrix2_2 = mt.pstruct2.distance_matrix
    mt.transform2(ope, new_lat=False, translation=False)
    ope_struct2_3.append([ope, mt.pstruct2])
    dmatrix2_3 = mt.pstruct2.distance_matrix
    # print('geometry) pstruct==pstruct1: ',  int(np.allclose(dmatrix, dmatrix1, atol=1e-03)))
    # print('geometry) pstruct==pstruct2_0: ',  int(np.allclose(dmatrix, dmatrix2_0, atol=1e-03)))
    # print('geometry) pstruct==pstruct2_1: ',  int(np.allclose(dmatrix, dmatrix2_1, atol=1e-03)))
    # print('geometry) pstruct==pstruct2_2: ',  int(np.allclose(dmatrix, dmatrix2_2, atol=1e-03)))
    # print('geometry) pstruct==pstruct2_3: ',  int(np.allclose(dmatrix, dmatrix2_3, atol=1e-03)))
    count1 += int(np.allclose(dmatrix, dmatrix1, atol=1e-03))
    count2_0 += int(np.allclose(dmatrix, dmatrix2_0, atol=1e-03))
    count2_1 += int(np.allclose(dmatrix, dmatrix2_1, atol=1e-03))
    count2_2 += int(np.allclose(dmatrix, dmatrix2_2, atol=1e-03))
    count2_3 += int(np.allclose(dmatrix, dmatrix2_3, atol=1e-03))
print('count1: ', 100*count1/n_ops)
print('count2_0: ', 100*count2_0/n_ops)
print('count2_1: ', 100*count2_1/n_ops)
print('count2_2: ', 100*count2_2/n_ops)
print('count2_3: ', 100*count2_3/n_ops)

#%%
opz = Rz(1*m.pi/2)
laty = np.array(lat@opz.T)
#fracy = frac@opz.T # frac coords with respect to the original structure's lattice vectors
# carty = fracy@lat
# fracyy = carty@np.linalg.inv(laty) # frac coords with respect to the original structure's lattice vectors
carty = np.array(cart@opz.T)
fracy = np.array(carty@np.linalg.inv(lat))
fracyy = np.array(carty@np.linalg.inv(laty))
pstructy_ = Structure(
    lattice=laty,
    species=hspecies0,
    coords=carty,
    coords_are_cartesian=True
)

#%%
def count_valence_electrons(pstruct):
    spec = pstruct.species
    elcount = 0
    for i, elem in enumerate(spec):
        va = elem.valence[1]
        elcount += va
    return elcount

struct = kstruct
rot = Rx(-0.01*m.pi)
tau = np.array([0,0,0])
struct_in = rotate_cart(struct, rot, tau, ope_lat=0)
mt = MatTrans(struct_in)
print(mt.sg)
print('dmatrix is maintained: ', np.allclose(struct.distance_matrix, struct_in.distance_matrix))
opes = list(set(mt.spgops))
opes


struct = kstruct #list(mp_hex.values())[2]    #['mp-10009']
rot = Rx(-0.0*m.pi)
tau = np.array([0,0,0])
struct_in = rotate_cart(struct, rot, tau, ope_lat=0)
mt = MatTrans(struct_in)
print(mt.sg)
print('dmatrix is maintained: ', np.allclose(struct.distance_matrix, struct_in.distance_matrix))
kopes = list(set(mt.spgops))
struct = list(mp_hex.values())[2]    #['mp-10009']
rot = Rx(-0.0*m.pi)
tau = np.array([0,0,0])
struct_in = rotate_cart(struct, rot, tau, ope_lat=0)
mt = MatTrans(struct_in)
print(mt.sg)
print('dmatrix is maintained: ', np.allclose(struct.distance_matrix, struct_in.distance_matrix))
opes = list(set(mt.spgops))
opes1 = [op.rotation_matrix for op in kopes]
opes2 = [op.rotation_matrix for op in opes]
len1, len2 = len(opes1), len(opes2)
matrix = np.zeros((len1, len2))
for i, op1 in enumerate(opes1):
    for j, op2 in enumerate(opes2):
        if np.allclose(abs(op1), abs(op2)):
            print(op1)
            matrix[i, j] = 1
matrix

#%%