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

from scripts.eval_utils import load_model
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from cdvae.common.data_utils import lattice_params_to_matrix_torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from utils.utils_plot import vis_structure


#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))

#%%

class MatSym():
    def __init__(self, pstruct):
        self.pstruct = pstruct
        self.astruct = Atoms(list(map(lambda x: x.symbol, pstruct.species)) , # list of symbols got from pymatgen
                        positions=pstruct.cart_coords.copy(),
                        cell=pstruct.lattice.matrix.copy(), pbc=True) 
        self.sga = SpacegroupAnalyzer(pstruct)
        self.symops=self.sga.get_symmetry_operations()
        self.spgops=self.sga.get_space_group_operations()
        
        self.pr_lat = self.sga.find_primitive()
        self.conv_struct = self.sga.get_conventional_standard_structure()
        self.hall = self.sga.get_hall()
        self.lat_type = self.sga.get_lattice_type()
        
        self.pgops_frac = self.sga.get_point_group_operations(cartesian=False)
        self.pgops_cart = self.sga.get_point_group_operations(cartesian=True)
        
        self.pg_sym = self.sga.get_point_group_symbol()
        self.pr_struct = self.sga.get_primitive_standard_structure()
        
        self.sg = [self.sga.get_space_group_number(), self.sga.get_space_group_symbol()]
        
        # self.pga = PointGroupAnalyzer(pstruct)
    def stransform(self, op):
        """_summary_

        Args:
            op (SymmOp): operation
        """
        self.pstruct.apply_operation(op)
    
    # def valid_symop(self, op):
    #     transformed_structure = op.operate(self.pstruct)
    #     if not transformed_structure.matches(self.pstruct):
    #         print("Symmetry operation is invalid")
    #     else:
    
    # def 


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
import math as m
  
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


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

#%%