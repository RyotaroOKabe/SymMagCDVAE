#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

from scripts.eval_utils import load_model

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *


from utils.utils_plot import vis_structure
#%%
from mendeleev import *

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))

#%%
