#%%
'''
https://www.notion.so/230416-space-group-operation-loss-c6e8cbc2e93f45f69982fe5a3ccec4f1
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