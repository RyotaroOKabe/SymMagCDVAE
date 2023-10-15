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
# load data (gen data) original CDVAE
task = 'gen'
label, sg_target = '' , 1
jobdir = join(hydradir, job_folder2)
use_path = join(jobdir, f'eval_{task}{label}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time_out =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)

astruct_list = get_astruct_end(use_path)
pstruct_list = [AseAtomsAdaptor.get_structure(a) for a in astruct_list]
symprec=0.08 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts = [MatTrans(p, symprec, angle_tolerance) for p in pstruct_list]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
sg_match = [sg==sg_target for sg in sgs]
correct_sum = sum(sg_match)
score = 100*correct_sum/len(sgs)
p1_match = [sg==1 for sg in sgs]
p1_score = 100*sum(p1_match)/len(sgs)
# (1) plt.scatter
plt.scatter(range(len(sgs)), sgs, s=2)
plt.title(f'[{label}] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})\n sg{sg_target} match rate: {score}% | P1: {p1_score}%')
plt.show()
plt.close()
# (2) plt.hist
plt.hist(np.array(sgs), bins=max(sgs))
plt.title(f'[{label}] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})\n sg{sg_target} match rate: {score}% | P1: {p1_score}%')
plt.show()
plt.close()




#%%

# load data (gen data)
task = 'gen'
label, sg_target = '_n2a1' , 2
# label, sg_target = '' , 1
jobdir = join(hydradir, job_folder2)
use_path = join(jobdir, f'eval_{task}{label}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time_out =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)

#%%

astruct_list = get_astruct_end(use_path)
pstruct_list = [AseAtomsAdaptor.get_structure(a) for a in astruct_list]
symprec=0.08 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts = [MatTrans(p, symprec, angle_tolerance) for p in pstruct_list]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
sg_match = [sg==sg_target for sg in sgs]
correct_sum = sum(sg_match)
score = 100*correct_sum/len(sgs)
p1_match = [sg==1 for sg in sgs]
p1_score = 100*sum(p1_match)/len(sgs)
# (1) plt.scatter
plt.scatter(range(len(sgs)), sgs, s=2)
plt.title(f'[{label}] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})\n sg{sg_target} match rate: {score}% | P1: {p1_score}%')
plt.show()
plt.close()
# (2) plt.hist
plt.hist(np.array(sgs), bins=max(sgs))
plt.title(f'[{label}] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})\n sg{sg_target} match rate: {score}% | P1: {p1_score}%')
plt.show()
plt.close()

# percent achievements
# sg_match = [sg==sg_target for sg in sgs]
# correct_sum = sum(sg_match)
# score = 100*correct_sum/len(sgs)
print(f'[sg#: {sg_target}]: {score}%')


#%% 
# comparison with the ground truth
# load ground truth (train, test, val data)
data_dir = join(homedir, 'data/mp_20')
file_name = 'test.csv'
file = data_dir + '/' + file_name
df = pd.read_csv(file)
pstructs0 = [str2pymatgen(crystal_str) for crystal_str in df['cif']]
astructs0 = [pymatgen2ase(p) for p in pstructs0]
lattices0, num_atoms0, frac_coords0, atom_types0 = pymatgen2outs(pstructs0)
print(df.keys())
print(f"{file}: {len(df)}") # same number as pstructs1 (recon)!!!
mts0 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs0]
sgs0 = [mt.sg[0] for mt in mts0]
len0_ = len(sgs0)
# (1) plt.scatter
plt.scatter(range(len(sgs0)), sgs0, s=2)
plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
plt.show()
plt.close()
# (2) plt.hist
plt.hist(np.array(sgs0), bins=max(sgs0))
plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
plt.show()
plt.close()




#%%


#%%
# GT distribution change under different threshold values. 
# symprec=0.08 # default: 0.01
count = 0
angle_tolerance=20.0 # default: 5.0
for symprec in np.linspace(0.01, 0.08, 9):
    mts0 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs0]
    sgs0 = [mt.sg[0] for mt in mts0]
    len0_ = len(sgs0)
    # (1) plt.scatter
    plt.scatter(range(len(sgs0)), sgs0, s=2)
    plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
    plt.show()
    plt.close()
    # (2) plt.hist
    plt.hist(np.array(sgs0), bins=max(sgs0))
    plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
    plt.show()
    plt.close()
    
    count += 1
    print('count: ', count)


symprec=0.08 # default: 0.01
# angle_tolerance=20.0 # default: 5.0
for angle_tolerance in np.linspace(2, 20, 10):
    mts0 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs0]
    sgs0 = [mt.sg[0] for mt in mts0]
    len0_ = len(sgs0)
    # (1) plt.scatter
    plt.scatter(range(len(sgs0)), sgs0, s=2)
    plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
    plt.show()
    plt.close()
    # (2) plt.hist
    plt.hist(np.array(sgs0), bins=max(sgs0))
    plt.title(f'[GT] space groups (symprec: {symprec}, ang_tol: {angle_tolerance})')
    plt.show()
    plt.close()
    
    count += 1
    print('count: ', count)

#%%
