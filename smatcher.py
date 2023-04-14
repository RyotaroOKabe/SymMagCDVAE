#%%
'''
https://www.notion.so/structure-matcher-pymatgen-b4ea9fa8b85b43eba2199ccfc1e40179
'''
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
import os, sys
from itertools import permutations
from pymatgen.analysis.structure_matcher import StructureMatcher

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))

#%%
# cosn
cosn = mpdata['mp-20536']
silicon = mpdata['mp-149']
frac = cosn.frac_coords
lat = cosn.lattice.matrix
spec = cosn.species
natms = len(frac)
smatcher = StructureMatcher()

#%% 
# initial
cosn_copy = cosn.copy()
smatch0 = smatcher.fit(struct1=cosn, struct2=cosn_copy)
smatch_ = smatcher.fit(struct1=cosn, struct2=silicon)
print("[original copy] ", smatch0)
print("[vs silicon] ", smatch_)

#%%
# translation
for _ in range(5):
    shift = np.random.rand(1, 3)
    frac1 = frac + shift
    cosn1 = Structure(
        lattice=lat,
        species=spec,
        coords=frac1,
        coords_are_cartesian=False
        )
    smatch1 = smatcher.fit(struct1=cosn, struct2=cosn1)
    print(f'[translation] {smatch1} with shift: {shift}')



#%%
# for _ in range(5):
for pm in permutations(range(natms)):
    pm = list(pm)
    frac2 = frac[pm, :]
    spec2 = [spec[p] for p in pm]
    cosn2 = Structure(
        lattice=lat,
        species=spec2,
        coords=frac2,
        coords_are_cartesian=False
        )
    smatch2 = smatcher.fit(struct1=cosn, struct2=cosn2)
    print(f'[permutation] {smatch1} with pm: {pm}')

#%%


