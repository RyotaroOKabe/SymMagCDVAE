#%%
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from torch_geometric.data import Batch

from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from scripts.eval_utils import load_model
# hydra, omegaconfig

# set parmters 
model_path="/home/rokabe/data2/generative/magcdvae/hydra/singlerun/2023-02-22/mp_20"
n_step_each=100
step_lr=1e-4
min_sigma=0
save_traj=False
disable_bar=False

#%%
# load trained model (se load_model function)
model_path = Path(model_path)

model, test_loader, cfg =  load_model(
            model_path, load_data=True)

#%%
ld_kwargs = SimpleNamespace(n_step_each=n_step_each,
                    step_lr=step_lr,
                    min_sigma=min_sigma,
                    save_traj=save_traj,
                    disable_bar=disable_bar)
#%%
# Get train_loader
hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
datamodule: pl.LightningDataModule = hydra.utils.instantiate(
    cfg.data.datamodule, _recursive_=False
)
datamodule.setup()
test_datasets = datamodule.test_datasets
test_data_list = [test_datasets[i] for i in range(len(test_datasets))]
#%%
batches = []
for i in range(len(test_data_list)):
    print(i)
    batch = Batch.from_data_list(test_data_list[i])
    batches.append(batch)


#%%
idx = 0
# Get the latent variable for a set of crystal structures
latent_variable = model.encode(batches[idx])


#%%
# Perform PCA on the latent variable
n_components = 2
pca = PCA(n_components=n_components)
latent_pca = pca.fit_transform(latent_variable[-1].detach())

# Plot the first two principal components
plt.scatter(latent_pca[:,0], latent_pca[:,1], c=batches[idx].num_atoms, s=1, cmap=plt.cm.get_cmap('Spectral', 20))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.title("[num_atoms] $N_{pca}$="+str(n_components))
plt.show()

#%%
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from cdvae.common.data_utils import lattice_params_to_matrix_torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

class BatchStruct(Dataset):
    def __init__(self, batch):
        super().__init__()
        self.num = batch.y.shape[0]
        #self.struct_list, self.sga_list, self.sga_list = []
        self.data_list = []
        for i in range(self.num):
            fcoords=batch[i].frac_coords
            lengths=batch[i].lengths
            angles=batch[i].angles
            natoms=batch[i].num_atoms
            atypes=batch[i].atom_types
            matrix=lattice_params_to_matrix_torch(lengths, angles)
            lattice = Lattice(matrix)
            struct = Structure(lattice=lattice, species=atypes, coords=fcoords, coords_are_cartesian=False)
            sga = SpacegroupAnalyzer(struct)
            pga = PointGroupAnalyzer(struct)
            data = Data(struct=struct, sga=sga, pga=pga)
            self.data_list.append(data)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index):
        return self.data_list[index]      

#%%
magnetic_atoms = [23, 24, 25, 26, 27]

def count_mag(atypes):
    mcount = 0
    for a in atypes:
        if a in magnetic_atoms:
            mcount += 1
    return mcount

bs = BatchStruct(batch)

#%%
# Perform PCA on the latent variable
n_components = 2
pca = PCA(n_components=n_components)
latent_pca = pca.fit_transform(latent_variable[-1].detach())
lattice_types = {'triclinic':0, 'monoclinic':1, 'orthorhombic':2, 'tetragonal':3, 
                 'rhombohedral':4, 'hexagonal':5, 'cubic':6}

category = []
for j in range(len(bs)):
    category.append(lattice_types[bs[j].sga.get_lattice_type()])
category = np.array(category)

# Plot the first two principal components
plt.scatter(latent_pca[:,0], latent_pca[:,1], 
            c=category, s=1,   
            cmap=plt.cm.get_cmap('Spectral', len(category)))
plt.xlim([-8, 11])
plt.ylim([-7, 7])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.title("[Lattice Type] $N_{pca}$="+str(n_components))
plt.show()

#%%
len_lt = len(lattice_types)
fig, axs = plt.subplots(1,len_lt, figsize=(5*len_lt, 5))
for i in range(len_lt):
    ax = axs[i]
    ax.scatter(latent_pca[:,0][category==i], latent_pca[:,1][category==i], 
                c=category[category==i], s=1,) 
                # cmap=plt.cm.get_cmap('Spectral', len(category[category==0])))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_xlim([-8, 11])
    ax.set_ylim([-7, 7])
    # plt.colorbar()
    ax.set_title(f"Lat: {list(lattice_types.keys())[i]} / $Npca$="+str(n_components))
    # ax.show()
