#%%
from pathlib import Path
from typing import List
import torch

from torch_geometric.data import Batch

# from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT
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

# from scripts.eval_utils import load_model
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
# from cdvae.common.data_utils import lattice_params_to_matrix_torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import math as m

from utils.utils_plot import vis_structure

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

class MatTrans(MatSym):
    def __init__(self, pstruct):
        super().__init__(pstruct)
        opes = list(set(self.spgops))
        n_ops = len(opes)
        # original structure info
        self.lat = self.pstruct.lattice.matrix
        self.cart = self.pstruct.cart_coords
        self.frac = self.pstruct.frac_coords
        self.spec = self.pstruct.species
    
    def transform1(self,ope):
        opr = ope.rotation_matrix
        opt = ope.translation_vector
        # pstructx: apply operation to cart coords with apply_operation method
        self.pstruct1 = self.pstruct.copy()
        self.pstruct1.apply_operation(ope)   # this operation might not be reliable
        self.lat1 = self.pstruct1.lattice.matrix
        self.cart1 = self.pstruct1.cart_coords
        self.frac1 = self.pstruct1.frac_coords

    def transform2(self,ope, new_lat=False, translation=True):
        opr = ope.rotation_matrix
        opt = ope.translation_vector
        # pstructy: apply operation to frac coords
        self.lat2 = (np.eye(3)@opr.T)@self.lat
        if translation:
            self.frac2 = (self.frac@opr.T + opt) # frac coords with respect to the original structure's lattice vectors
        else: 
            self.frac2 = self.frac@opr.T
        self.cart2 = self.frac2@self.lat
        self.frac2_2 = self.cart2@np.linalg.inv(self.lat2) # frac coords with respect to the original structure's lattice vectors
        if new_lat:
            lat_use = self.lat2
        else: 
            lat_use = self.lat
        self.pstruct2 = Structure(
            lattice=lat_use,
            species=self.spec,
            coords=self.cart2,
            coords_are_cartesian=True
        )

    def transform3(self,ope, new_lat=False, translation=True):
        opr = ope.rotation_matrix
        opt = ope.translation_vector
        # pstructy: apply operation to frac coords
        self.lat3 = (np.eye(3)@opr.T)@self.lat
        if translation:
            self.frac3 = (self.frac+opt)@opr.T # frac coords with respect to the original structure's lattice vectors
        else: 
            self.frac3 = self.frac@opr.T
        self.cart3 = self.frac3@self.lat
        self.frac3_2 = self.cart3@np.linalg.inv(self.lat3) # frac coords with respect to the original structure's lattice vectors
        if new_lat:
            lat_use = self.lat3
        else: 
            lat_use = self.lat
        self.pstruct3 = Structure(
            lattice=lat_use,
            species=self.spec,
            coords=self.cart3,
            coords_are_cartesian=True
        )
        

def distance_sorted(pstruct):
    ccoord = pstruct.cart_coords
    fcoord = pstruct.frac_coords
    natms = len(ccoord)
    num = int(natms*(natms-1)/2)
    c_dists = np.zeros((num, 3))
    f_dists = np.zeros((num, 3))
    for i, (src, dst) in enumerate(combinations(range(natms), 2)):
        c_d = np.linalg.norm(ccoord[dst]-ccoord[src])
        f_d = np.linalg.norm(fcoord[dst]-fcoord[src])
        c_dists[i, 0] = c_d 
        c_dists[i, 1] = src
        c_dists[i, 2] = dst
        f_dists[i, 0] = f_d 
        f_dists[i, 1] = src
        f_dists[i, 2] = dst
    return c_dists[c_dists[:, 0].argsort()], f_dists[f_dists[:, 0].argsort()]


def Rx(theta):
    """_summary_

    Args:
        theta (float): angle (rad)

    Returns:
        _type_: _description_
    """
    return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])   # np.matrix
  
def Ry(theta):
    """_summary_

    Args:
        theta (float): angle (rad)

    Returns:
        _type_: _description_
    """
    return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
    """_summary_

    Args:
        theta (float): angle (rad)

    Returns:
        _type_: _description_
    """
    return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


def rotate_cart(pstruct_in, rot, tau, ope_lat=False):
    """_summary_

    Args:
        pstruct_in (_type_): _description_
        rot (_type_): rotation operation for cart coord
        tau (_type_): translation (cart)
        ope_lat (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    pstruct = pstruct_in.copy()
    lat0 = pstruct.lattice.matrix
    cart0 = pstruct.cart_coords
    frac0 = pstruct.frac_coords
    spec0 = pstruct.species
    if ope_lat:
        lat = np.array(lat0@rot.T)
    else:
        lat = lat0
    cart = np.array(cart0@rot.T) + tau
    out =  Structure(
        lattice=lat,
        species=spec0,
        coords=cart,
        coords_are_cartesian=True
    )
    return out

def switch_latvecs(pstruct_in, abc=True):
    """

    Args:
        pstruct_in (_type_): _description_
        abc (bool, optional): If True, vec{a} becomes vec{b}, vec{b} becomes vec{c}, vec{c} becomes vec{a}. Defaults to True.

    Returns:
        _type_: _description_
    """
    pstruct = pstruct_in.copy()
    lat0 = pstruct.lattice.matrix
    cart0 = pstruct.cart_coords
    frac0 = pstruct.frac_coords
    spec0 = pstruct.species
    rotmat = np.array([[0,1,0],[0,0,1],[1,0,0]])
    if abc: 
        lat = rotmat @ lat0
    else: 
        lat = np.linalg.inv(rotmat) @ lat0
    out =  Structure(
        lattice=lat,
        species=spec0,
        coords=cart0,
        coords_are_cartesian=True
    )
    return out

#%%

