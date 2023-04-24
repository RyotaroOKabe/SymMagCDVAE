#%%
"""
https://www.notion.so/230408-visualization-of-the-generative-process-84753ea722e14a358cf61832902bb127
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from ase import Atom, Atoms
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
import torch
import os
from os.path import join
import imageio
from utils.utils_plot import vis_structure, movie_structs
from utils.utils_output import get_astruct_list, get_astruct_all_list, output_eval
from cdvae.common.data_utils import lattice_params_to_matrix_torch
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
homedir = '/home/rokabe/data2/generative/magcdvae'
hydradir = '/home/rokabe/data2/generative/magcdvae/hydra/singlerun/'
datadir = join(homedir, 'data/mp_20')   #!
file =  join(datadir, 'train.csv')
savedir = join(homedir, 'figures')

print("datadir: ", datadir)

#%%
job = "2023-04-15/mp_20_2"   #!
task = 'recon'
jobdir = join(hydradir, job)
use_path = join(jobdir, f'eval_{task}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)
#%%
#[1] check each structure aaa
idx = 0
astruct_list = get_astruct_list(use_path, idx)

#%%
# structdir = join(savedir, job, str(idx))
# movie_structs(astruct_list, f"struct{idx}", savedir=structdir, supercell=np.diag([1,1,1]))

#%%
#[1] check all structures in the batch
astruct_lists = get_astruct_all_list(use_path)

#%%
for idx in range(num):
    print(f"0000_{idx}")
    structdir = join(savedir, job, task, str(idx))
    print("structdir: ", structdir)
    movie_structs(astruct_lists[idx], f"0000_{idx}", savedir=structdir, supercell=np.diag([1,1,1]))


#%%
