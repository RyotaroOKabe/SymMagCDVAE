#%%
"""
https://www.notion.so/Evaluation-of-output-materials-symmetry-dcfe7ee56436496a993cd612babe7255
evaluate the symmetry of the output materials 
"""
# import modules
"""
https://www.notion.so/231005-symmetry-enforcement-evaluation-9d71492bd7244f2bb682f76e5954bb90?pvs=4#856ab88f19ea49d5944117e86bb42544
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atom, Atoms
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
import torch
import os
from os.path import join
import imageio
from utils.utils_plot import vis_structure, movie_structs
from utils.utils_output import get_astruct_list, get_astruct_all_list, get_astruct_end, output_eval
from utils.utils_material import *
from cdvae.pl_modules.space_group import *
from cdvae.common.data_utils import lattice_params_to_matrix_torch
import random
from dirs import *
# mpdata = pkl.load(open('./data/mp_full.pkl', 'rb'))
# mpids = sorted(list(mpdata.keys()))
# mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
# homedir = '/home/rokabe/data2/generative/symcdvae'
# hydradir = '/home/rokabe/data2/generative/hydra/singlerun/'
datadir = join(homedir, 'data/mp_20')   #!
file =  join(datadir, 'train.csv')
savedir = join(homedir, 'figures')

print("datadir: ", datadir)


#%%
# load data (gen data)
task = 'gen'
label = '_n2a2' 
jobdir = join(hydradir, job_folder2)
use_path = join(jobdir, f'eval_{task}{label}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time_out =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)




#%%
# space group distributions (parameter, tolerance)
idx = 5
astruct_list = get_astruct_list(use_path, idx)
pstruct_list = [AseAtomsAdaptor.get_structure(a) for a in astruct_list]
mts = [MatTrans(p) for p in pstruct_list]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
# (1) plt.plot
plt.plot(sgs)
plt.title(f'[{idx}] {astruct_list[-1].get_chemical_formula()}')
plt.show()
plt.close()
# (2) plt.hist
plt.hist(np.array(sgs), bins=len_)
plt.title(f'space group distributions')
plt.show()
plt.close()



#%% 
# comparison with the ground truth






#%%






#%%






#%%
