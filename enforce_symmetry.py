#%%
"""
https://www.notion.so/231016-symmetry-enforcement-with-minimum-repetitive-unit-for-initial-structure-score-and-langeevin--579607f204a4467ab5c277725dda7a1c
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
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
mpdata = pkl.load(open('./data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))
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
# diffuse structures retaining symmetry 
sg = 225
mpids_sg = sorted(list(mp_dicts[sg].keys()))
mpid = mpids_sg[-1]
pstruct = mp_dicts[sg][mpid]
print('sg#: ', sg)
print('mpid: ', mpid)
print(pstruct)
vis_structure(pstruct)
plt.show()
plt.close()

# histogram
sg_sites = [len(ps.sites) for ps in mp_dicts[sg].values()]
plt.hist(sg_sites, bins = max(sg_sites))
plt.title(f'[sg {sg}]: {len(sg_sites)} materials')
plt.show()
plt.close()
# Iterate through the list
count_dict = {}
for num in sg_sites:
    # Check if the number is already in the dictionary
    if num in count_dict:
        # If it is, increment the count
        count_dict[num] += 1
    else:
        # If it's not, add it to the dictionary with a count of 1
        count_dict[num] = 1
# Print the counts
for num in sorted(count_dict.keys()):
    count = count_dict[num]
    print(f"{num}: {count}")
    
#%%
# diffuse structures retaining symmetry 
sg = 194
mpids_sg = sorted(list(mp_dicts[sg].keys()))
mpid = mpids_sg[0]
pstruct = mp_dicts[sg][mpid]

frac = pstruct.frac_coords
mt = MatTrans(pstruct)
opes, oprs, opts, nops = mt.spgops,mt.oprs, mt.opts, mt.n_ops
sigma = 0.05
print('sg#: ', sg)
print('mpid: ', mpid)
print('frac (original): \n', frac)
print(pstruct)
vis_structure(pstruct)
# plt.close()


for i, (opr, opt) in enumerate(zip(oprs, opts)):
    print(opr, opt)
    noise0, noise1 = sigma*np.random.randn(*frac.shape), sigma*np.random.randn(*frac.shape)
    frac0 = (frac + noise0)%1
    frac1 = (frac@opr.T + opt)%1
    # print('frac1 (transformed): \n', frac1)
    differences = frac[:, np.newaxis, :] - frac1[np.newaxis, :, :]
    # Square the differences element-wise
    squared_differences = differences**2
    # Sum the squared differences along the last axis to get the squared distances
    squared_distances = np.sum(squared_differences, axis=-1)
    # Take the square root to get the Euclidean distances
    distances = np.sqrt(squared_distances)
    zero_indices = np.where(np.abs(distances) < 1e-03)
    zero_indices_list = list(zip(zero_indices[0], zero_indices[1]))
    print(distances)
    print(i, zero_indices_list)   


#%%
# dictionary to store the same sg, same natm
sg = 194
mpids_sg = sorted(list(mp_dicts[sg].keys()))
sg_natms = {}
for mpid in mpids_sg:
    natm = len(mp_dicts[sg][mpid].sites)
    if natm not in sg_natms.keys():
        sg_natms[natm]=[mpid]
        print(f'new natm {natm}: {mpid}')
    else:
        sg_natms[natm].append(mpid)
        print(f'natm {natm}: append {mpid}')
sg_natms = {key: sg_natms[key] for key in sorted(sg_natms.keys())}

mpid = sg_natms[2][-1]
pstruct = mp_dicts[sg][mpid]
frac = pstruct.frac_coords
mt = MatTrans(pstruct)
opes, oprs, opts, nops = mt.spgops,mt.oprs, mt.opts, mt.n_ops
sigma = 0.05
print('sg#: ', sg)
print('mpid: ', mpid)
print('frac (original): \n', frac)
print(pstruct)
vis_structure(pstruct)

#%% 
# # make it for all sgs
get_dict = False
if get_dict:
    dict_sg_natm ={}
    for sg in sorted(mp_dicts.keys()):
        print(f'[{sg}]')
        mpids_sg = sorted(list(mp_dicts[sg].keys()))
        sg_natms = {}
        for mpid in mpids_sg:
            natm = len(mp_dicts[sg][mpid].sites)
            if natm not in sg_natms.keys():
                sg_natms[natm]=[mpid]
                print(f'new natm {natm}: {mpid}')
            else:
                sg_natms[natm].append(mpid)
                print(f'natm {natm}: append {mpid}')
        sg_natms = {key: sg_natms[key] for key in sorted(sg_natms.keys())}
        dict_sg_natm[sg]=sg_natms
        

    # Save the dictionary as a JSON file
    with open('./data/mp_sg_natm.json', 'w') as json_file:
        json.dump(dict_sg_natm, json_file)

else: 
    with open('./data/mp_sg_natm.json', 'r') as json_file:
        dict_sg_natm = json.load(json_file)

#%%
sg = 191
mpid = dict_sg_natm[sg][2][-1]
pstruct = mp_dicts[sg][mpid]
frac = pstruct.frac_coords
mt = MatTrans(pstruct)
opes, oprs, opts, nops = mt.spgops,mt.oprs, mt.opts, mt.n_ops
sigma = 0.05
print('sg#: ', sg)
print('mpid: ', mpid)
print('frac (original): \n', frac)
print(pstruct)
vis_structure(pstruct)

#%%