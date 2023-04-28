#%%
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

from pathlib import Path
from types import SimpleNamespace

from scripts.eval_utils import load_model

#%%
# load data
model_path = ''
tasks = ['gen']
start_from='data'
n_step_each=100
step_lr=1e-4
min_sigma=0
save_traj=True
disable_bar=False

# load model
model_path = Path(model_path)
model, test_loader, cfg = load_model(
    model_path, load_data=('recon' in tasks) or
    ('opt' in tasks and start_from == 'data'))
ld_kwargs = SimpleNamespace(n_step_each=n_step_each,
                            step_lr=step_lr,
                            min_sigma=min_sigma,
                            save_traj=save_traj,
                            disable_bar=disable_bar)


# [1] normal langevin
torch.no_grad()


# [2] langevin with sgo loss



