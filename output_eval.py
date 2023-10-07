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
from utils.utils_material import MatTrans
from cdvae.common.data_utils import lattice_params_to_matrix_torch
from dirs import *
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
job = "2023-06-10/mp_20_2"   #!
task = 'gen'
jobdir = join(hydradir, job)
use_path = join(jobdir, f'eval_{task}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)
#%%
#[1] check each structure aaa
idx = 5
astruct_list = get_astruct_list(use_path, idx)
pstruct_list = [AseAtomsAdaptor.get_structure(a) for a in astruct_list]
mts = [MatTrans(p) for p in pstruct_list]
plt.plot([mt.sg[0] for mt in mts])
plt.title(f'[{idx}] {astruct_list[-1].get_chemical_formula()}')
plt.show()
plt.close()

#%%
# space group distribution
# check only the final product
# astruct_lists = get_astruct_all_list(use_path)
# astructs = [astruct_list[-1] for astruct_list in astruct_lists] # only the final product
astructs = get_astruct_end(use_path)
pstructs = [AseAtomsAdaptor.get_structure(a) for a in astructs] # only the final product
mts = [MatTrans(p) for p in pstructs]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
plt.hist(np.array(sgs), bins=len_)
plt.title(f'space group distributions')
plt.show()
plt.close()


#%%
# tune the tolerance
# def get_sg(pstruct, tol):
#     if isinstance(pstruct, Atoms):
#         pstruct = AseAtomsAdaptor.get_structure(pstruct)
#     sga = SpacegroupAnalyzer(pstruct)
#     symops=sga.get_symmetry_operations()
#     spgops=sga.get_space_group_operations()
    
#     pr_lat = sga.find_primitive()
#     conv_struct = sga.get_conventional_standard_structure()
#     hall = sga.get_hall()
#     lat_type = sga.get_lattice_type()
#     pgops_frac = sga.get_point_group_operations(cartesian=False)
#     pgops_cart = sga.get_point_group_operations(cartesian=True)
#     pg_sym = sga.get_point_group_symbol()
#     pr_struct = sga.get_primitive_standard_structure()
#     sg = [sga.get_space_group_number(), sga.get_space_group_symbol()]
symprec=0.09 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts = [MatTrans(p, symprec, angle_tolerance) for p in pstructs]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
# plt.hist(np.array(sgs), bins=len_)
plt.plot(np.array(sgs))
plt.title(f'space group distributions (symprec={symprec}, ang tol={angle_tolerance})')
plt.show()
plt.close()

#%%
#  transition of the space group through the optimization process



#%% 
# comparison with the ground truth
# load reconstruction
task1 = 'recon'
use_path1 = join(jobdir, f'eval_{task1}.pt') #!

lengths1, angles1, num_atoms1, frac_coords1, atom_types1, all_frac_coords_stack1, all_atom_types_stack1, eval_setting1, time1 =output_eval(use_path1)
lattices1 = lattice_params_to_matrix_torch(lengths1[0], angles1[0])
num1 = len(lattices1)

idx = 5
astruct_list1 = get_astruct_list(use_path1, idx)
pstruct_list1 = [AseAtomsAdaptor.get_structure(a) for a in astruct_list1]
astructs1 = get_astruct_end(use_path1)
pstructs1 = [AseAtomsAdaptor.get_structure(a) for a in astructs1] # only the final product
symprec=0.09 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts1 = [MatTrans(p, symprec, angle_tolerance) for p in pstruct_list1]
plt.plot([mt.sg[0] for mt in mts1])
plt.title(f'[{idx}] {astruct_list1[-1].get_chemical_formula()}')
plt.show()
plt.close()

symprec=0.09 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts1 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs1]
sgs1 = [mt.sg[0] for mt in mts1]
len1_ = len(sgs1)
# plt.hist(np.array(sgs), bins=len_)
plt.plot(np.array(sgs))
plt.title(f'space group distributions (symprec={symprec}, ang tol={angle_tolerance})')
plt.show()
plt.close()

#%%
# load ground truth (train, test, val data)



# check the length of the data

#%%
