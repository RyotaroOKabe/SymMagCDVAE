#%%
from e3nn import o3
from e3nn.o3 import Irreps, Irrep
import torch
from torch import Tensor
from pathlib import Path
from typing import List
from torch_geometric.data import Batch

# from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import pickle as pkl
from ase import Atoms

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from itertools import combinations

# from scripts.eval_utils import load_model
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
# from cdvae.common.data_utils import lattice_params_to_matrix_torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import math as m

from utils.utils_plot import vis_structure
from utils.utils_material import MatSym, MatTrans, distance_sorted, Rx, Ry, Rz, rotate_cart, switch_latvecs
tol=1e-05

#%%
irreps = Irreps("1o")
# irreps

# show the transformation matrix corresponding to the inversion
irreps.D_from_angles(alpha=Tensor(0.0), beta=Tensor(0.0), gamma=Tensor(0.0), k=Tensor(1))

#%%
opr = Tensor(
    [[-1., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0., -1.]]
    )

# opr_ = irreps.D_from_matrix(opr)
irreps = Irreps("4x0e + 3x0o + 3x1o + 2x2o")
D = irreps.D_from_matrix(opr)
plt.imshow(D, cmap='bwr', vmin=-1, vmax=1);

#%%

mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))

#%%
pstruct = mpdata['mp-20536']
kfrac = np.array([
    [0,0,0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0]
])#np.random.rand(N,3)
# N = len(hfrac)
hex_latice = np.array([
    [6,0,0],
    [3, 3*np.sqrt(3),0],
    [0,0,4]
])
kspecies = np.array([6, 6, 6])  
kspecies1 = np.array([6, 7, 8])
kstruct = Structure(
    lattice=hex_latice,
    species=kspecies,
    coords=kfrac,
    coords_are_cartesian=False
)

#%%

struct_in = pstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]

#%%
tr = 3 ** -0.5
tw = 2 ** -0.5
irrep2tens = torch.tensor([[    tr,  0,   0,   0,      tr,  0,  0,   0,     tr],
                                [     0,  0,   0,   0,       0, tw,  0, -tw,      0],
                                [     0,  0, -tw,   0,       0,  0, tw,   0,      0],
                                [     0, tw,   0, -tw,       0,  0,  0,   0,      0],
                                [     0,  0,  tw,   0,       0,  0, tw,   0,      0],
                                [     0, tw,   0,  tw,       0,  0,  0,   0,      0],
                                [-tw*tr,  0,   0,   0, 2*tw*tr,  0,  0,   0, -tw*tr],
                                [     0,  0,   0,   0,       0, tw,  0,  tw,      0],
                                [   -tw,  0,   0,   0,       0,  0,  0,   0,     tw]], dtype = torch.complex128)

#%%
# rotation of 1o object
irreps1 = Irreps('1x1o+1x1e')
rot1ca = o3.rand_matrix(1)
rot1ir = irreps1.D_from_matrix(rot1ca)
print('rot1ca: ', rot1ca)
print('rot1ir: ', rot1ir)
rot2ca = o3.rand_matrix(1)
rot2ir = irreps1.D_from_matrix(rot2ca)
rot_prod_ca = rot1ca@rot2ca
rot_prod_ir = irreps1.D_from_matrix(rot_prod_ca)
print("rot_prod correspondence: ", torch.allclose(rot_prod_ir, rot1ir@rot2ir, atol=tol))

#%%
# https://docs.e3nn.org/en/latest/api/nn/models/v2103.html
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork

net = SimpleNetwork(
    irreps_in="3x0e + 2x1o",
    irreps_out="1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0
)

pos = torch.randn(5, 3)
x = net.irreps_in.randn(5, -1)

net({
    'pos': pos,
    'x': x
})

#%%
rot = o3.matrix_x(torch.tensor(2*m.pi / 3.0))
output1 = net({
    'pos': pos @ rot.T,
    'x': x @ net.irreps_in.D_from_matrix(rot).T
})

output2 = net({
    'pos': pos,
    'x': x
}) @ net.irreps_out.D_from_matrix(rot).T

print("output correspondence: ", torch.allclose(output1, output2, atol=tol))

#%%
def rot_corresp(model, pos, rot, tol=tol):
    output1 = net({
        'pos': pos @ rot.T,
        'x': x @ net.irreps_in.D_from_matrix(rot).T
    })

    output2 = net({
        'pos': pos,
        'x': x
    }) @ net.irreps_out.D_from_matrix(rot).T
    print('output1: ', output1)
    print('output2: ', output2)
    return torch.allclose(output1, output2, atol=tol)

#%%
pos = torch.randn(5, 3)
print(rot_corresp(net, pos, rot, tol=tol))
print(rot_corresp(net, pos, Tensor(oprs[0]), tol=tol))
#%%