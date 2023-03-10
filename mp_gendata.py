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
import pandas as pd

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
import codecs

save_dir = '/data2/rokabe/generative/magcdvae/data/mp_hex'

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))

#%%
def get_cif(file, cif_dir, string=True):
    cif_path = cif_dir + '/' + file
    with codecs.open(cif_path, 'rb', encoding='utf-8', errors='ignore') as ciff:
        cif_read=ciff.readlines()
    if string:
        mcif_read = "".join(mcif_read)
    return mcif_read

# cif > pymatgen 
def cif2pymatgen(file, cif_dir):
    mcif_read = get_cif(file, cif_dir)
    # print(type(mcif_read))
    pstruct=Structure.from_str(mcif_read, "CIF")
    return pstruct

# pymatgen > ase.Atom
def pymatgen2ase(pstruct):
    return Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions = pstruct.cart_coords.copy(),
                    cell = pstruct.lattice.matrix.copy(), 
                    pbc=True)

def pymatgen2cif(pstruct):
    cif_str = pstruct.to(filename="", fmt='cif')
    return cif_str

def split_df(df, ratio=(0.8, 0.1, 0.1)):
    train, valid, test = \
                np.split(df.sample(frac=1, random_state=42), 
                        [int(ratio[0]*len(df)), int(sum(ratio[:2])*len(df))])
    train, valid, test = [d.reset_index(drop=True) for d in [train, valid, test]] 
    return train, valid, test

#%%
mp_hex = {}
for mpid in mpids:
    pstruct = mpdata[mpid]
    sga = SpacegroupAnalyzer(pstruct)
    sgn = sga.get_space_group_number()
    if 143 <=sgn<=194:
        print(mpid, sgn)
        mp_hex[mpid]=pstruct

pkl.dump(mp_hex, open('data/mp_hex.pkl', 'wb'))

#%%
mp_cif ={}
mpids_hex = sorted(list(mp_hex.keys()))
for mpid in mpids_hex:
    pstruct = mp_hex[mpid]
    cif_str = pymatgen2cif(pstruct)
    mp_cif[mpid] =cif_str

#%%
df = pd.DataFrame(mp_cif.items(), columns=['material_id', 'cif'])
df['energy'] = df['material_id'].map(lambda x: 0)
train, valid, test = split_df(df)
#%%
df.to_csv(f'{save_dir}/mp_hex.csv')
train.to_csv(f'{save_dir}/train.csv')
valid.to_csv(f'{save_dir}/valid.csv')
test.to_csv(f'{save_dir}/test.csv')
#%%
# test loading data
dffull = pd.read_csv(f'{save_dir}/mp_hex.csv', index_col=[0])
dftrain = pd.read_csv(f'{save_dir}/train.csv', index_col=[0])
dfvalid = pd.read_csv(f'{save_dir}/val.csv', index_col=[0])
dftest = pd.read_csv(f'{save_dir}/test.csv', index_col=[0])
df_set = [dffull, dftrain, dfvalid, dftest]
print('data size: ', [len(d) for d in df_set])
# print(f"{save_dir}/mp_hex.csv: {len(df1)}")