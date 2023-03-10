#%%
# import modules
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
from scripts.eval_utils import load_model

# set parmters 
model_path="/home/rokabe/data2/generative/magcdvae/hydra/singlerun/2023-02-26/mp_20_short"
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


# model





# trainer
