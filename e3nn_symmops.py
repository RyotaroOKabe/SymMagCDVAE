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
from itertools import permutations
tol=1e-05

#%%
irreps = Irreps("1o")
# irreps

# show the transformation matrix corresponding to the inversion
irreps.D_from_angles(alpha=Tensor([0.0]), beta=Tensor([0.0]), gamma=Tensor([0.0]), k=Tensor([1]))

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
cosn = mpdata['mp-20536']
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

struct_in = cosn
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
def rot_corresp(model, pos, x, rot, tol=tol):
    output1 = model({
        'pos': pos @ rot.T,
        'x': x @ model.irreps_in.D_from_matrix(rot).T
    })

    output2 = model({
        'pos': pos,
        'x': x
    }) @ model.irreps_out.D_from_matrix(rot).T
    print('output1: ', output1)
    print('output2: ', output2)
    return torch.allclose(output1, output2, atol=tol)

#%% 
# test
pos = torch.randn(5, 3)
natms = len(pos)
x = net.irreps_in.randn(natms, -1)
rotx = o3.matrix_x(torch.tensor(1*m.pi / 3.0))
roty = o3.matrix_y(torch.tensor(1*m.pi / 3.0))
rotz = o3.matrix_z(torch.tensor(1*m.pi / 3.0))
rotr = o3.rand_matrix(1).reshape(3,3)
print("[1-1] ", rot_corresp(net, pos, x, rotx, tol=tol))
print("[1-2] ", rot_corresp(net, pos, x, roty, tol=tol))
print("[1-3] ", rot_corresp(net, pos, x, rotz, tol=tol))
print("[2] ", rot_corresp(net, pos, x, rotr, tol=tol))
print("[3] ", rot_corresp(net, pos, x, Tensor(oprs[0]), tol=tol))

#%%
# teststructure's coordinatesruct_in = kstruct
pos = Tensor(struct_in.cart_coords)#[:5]
natms = len(pos)
x = net.irreps_in.randn(natms, -1)
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
rotx = o3.matrix_x(torch.tensor(1*m.pi / 3.0))
roty = o3.matrix_y(torch.tensor(1*m.pi / 3.0))
rotz = o3.matrix_z(torch.tensor(1*m.pi / 3.0))
rotr = o3.rand_matrix(1).reshape(3,3)
print("[1-1] ", rot_corresp(net, pos, x, rotx, tol=tol))
print("[1-2] ", rot_corresp(net, pos, x, roty, tol=tol))
print("[1-3] ", rot_corresp(net, pos, x, rotz, tol=tol))
print("[2] ", rot_corresp(net, pos, x, rotr, tol=tol))
print("[3] ", rot_corresp(net, pos, x, Tensor(oprs[11]), tol=tol))
#%%
net1 = SimpleNetwork(
    irreps_in="1x1o",
    irreps_out="1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0
)
struct_in = pstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
frac = Tensor(struct_in.frac_coords)
opr = Tensor(oprs[0])
x = frac#
# x = net1.irreps_in.randn(len(frac), -1)
output1 = net1({
    'pos': (frac @ opr.T)%1,
    'x': x @ net1.irreps_in.D_from_matrix(opr).T%1
})


output2 = net1({
    'pos': frac,
    'x': x
}) @ net1.irreps_out.D_from_matrix(opr).T
output2 = output2#%1

print("output correspondence: ", torch.allclose(output1, output2, atol=tol))

#%%
net1 = SimpleNetwork(
    irreps_in="1x1o",
    irreps_out="1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0
)

struct_in = cosn
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
frac = Tensor(struct_in.frac_coords)
opr = Tensor(oprs[0])
x = frac#
# x = net1.irreps_in.randn(len(frac), -1)
output1 = net1({
    'pos': (frac @ opr.T)%1,
    'x': (x @ opr.T)%1
})


output2 = net1({
    'pos': frac,
    'x': x
}) @ opr.T #@ net1.irreps_out.D_from_matrix(opr).T

print("output correspondence: ", torch.allclose(output1, output2, atol=tol))

output3 = net1({
    'pos': frac,
    'x': x
}) @ net1.irreps_out.D_from_matrix(opr).T


f_idx = [1, 2, 0, 4, 3, 5]
output4 = net1({
    'pos': frac[f_idx, :],
    'x': x[f_idx, :]
}) @ opr.T #@ net1.irreps_out.D_from_matrix(opr).T




#%%
struct_in = cosn
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
frac = Tensor(struct_in.frac_coords)
opr = Tensor(oprs[0])

#%%

class SimpleTP(torch.nn.Module):
    def __init__(self, 
                 irreps_in1='1x1o',
                 irreps_in2='1x1o',
                 irreps_out='1x1o',
                 ):
        super().__init__()
        
        self.irreps_in1=Irreps(irreps_in1)
        self.irreps_in2=Irreps(irreps_in2)
        # self.irreps_out=Irreps(irreps_out)
    
        self.tp = o3.FullTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in1
        )
        self.irreps_out=self.tp.irreps_out

    def forward(self, frac):
        output = self.tp(frac, frac)
        return torch.sum(output, dim=0)


#%%
model = SimpleTP()
print(model)
tol = 1e-02
out1 = model(frac)
out2 = model((frac@opr.T)%1)

torch.allclose(out1, out2, atol=tol)

#%%

# def edge_corresp(model, data1, data2, opr, tol=tol):
#     output1 = model(
#         input1= data1 @ opr.T,
#         input2=x @ model.irreps_in.D_from_matrix(rot).T
#     })

#     output2 = model({
#         'pos': pos,
#         'x': x
#     }) @ model.irreps_out.D_from_matrix(rot).T
#     print('output1: ', output1)
#     print('output2: ', output2)
#     return torch.allclose(output1, output2, atol=tol)

def operation_identity(struct_in, opr, tol, get_diff=True):
    frac = Tensor(struct_in.frac_coords)
    opr = Tensor(opr)
    A = frac
    B = (frac@opr.T)%1
    assert A.shape == B.shape
    out1 = model(A)
    out2 = model(B)
    if get_diff:
        return out1-out2
    else: 
        return torch.allclose(out1, out2, atol=tol)
#
def disturb_frac(struct_in, sigma=0.1 , random=True, different=True):
    frac = struct_in.frac_coords
    lat = struct_in.lattice.matrix
    spec = struct_in.species
    dist = np.random.normal(loc=0.0, scale=sigma, size=frac.shape)
    frac1 = frac + dist
    struct_out = Structure(
    lattice=lat,
    species=spec,
    coords=frac1,
    coords_are_cartesian=False
    )
    return struct_out

def shift_frac(struct_in, shift=[0, 0.5, 0]):
    frac = struct_in.frac_coords
    lat = struct_in.lattice.matrix
    spec = struct_in.species
    frac1 = frac + np.array(shift)
    frac1 = frac1%1
    struct_out = Structure(
    lattice=lat,
    species=spec,
    coords=frac1,
    coords_are_cartesian=False
    )
    return struct_out

def diff_with_opes(struct_in, oprs):
    vectors = []
    for opr in oprs:
        diff = operation_identity(struct_in, opr, tol, get_diff=True)
        vectors.append(diff)
    return torch.stack(vectors, dim=0)


logvars = range(0, -45, -1)
xs = [10**l for l in logvars]
ys = []
for l in logvars:
    sigma = 10**l
    dstruct = disturb_frac(cosn, sigma=sigma, random=True, different=True)
    dvec = diff_with_opes(dstruct, oprs)
    ys.append(np.linalg.norm(dvec))

plt.plot(logvars, ys)


#%%
# get neigbor
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom, Atoms

struct_in = cosn
astruct = Atoms(list(map(lambda x: x.symbol, struct_in.species)) , # list of symbols got from pymatgen
        positions=struct_in.frac_coords.copy(),
        cell=np.eye(3), pbc=True) 

vis_structure(astruct)
r_max = 0.9
edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = r_max, self_interaction = True)

#%%
#  operation iddentity2
def operation_identity2(struct_in, opr, r_max, tol, get_diff=True):
    frac = Tensor(struct_in.frac_coords)
    opr = Tensor(opr)
    frac0 = frac
    frac1 = frac@opr.T #(frac@opr.T)%1
    assert frac0.shape == frac1.shape
    astruct0 = Atoms(list(map(lambda x: x.symbol, struct_in.species)),
                        positions=frac0, cell=np.eye(3), pbc=True) 
    astruct1 = Atoms(list(map(lambda x: x.symbol, struct_in.species)),
                        positions=frac1, cell=np.eye(3), pbc=True) 
    edge_src0, edge_dst0, edge_shift0, edge_vec0, edge_len0 = neighbor_list("ijSDd", a = astruct0, cutoff = r_max, self_interaction = True)
    edge_src1, edge_dst1, edge_shift1, edge_vec1, edge_len1 = neighbor_list("ijSDd", a = astruct1, cutoff = r_max, self_interaction = True)
    out0 = model(Tensor(edge_vec0))
    out1 = model(Tensor(edge_vec1))
    if get_diff:
        return out1-out0
    else: 
        return torch.allclose(out1, out0, atol=tol)

#%%
sigma = 0.08
cosn_d = disturb_frac(cosn, sigma=sigma)
vis_structure(cosn_d)
operation_identity2(cosn_d, opr, 0.8, tol, get_diff=True)

#%%
def diff_with_opes2(struct_in, oprs, r_max):
    vector = []
    for opr in oprs:
        diff = operation_identity2(struct_in, opr, r_max, tol, get_diff=True)
        vector.append(diff)
    return torch.cat(vector)

#%%


logvars = np.linspace(-3, 0, num=31) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys = []
# struct_in = mpdata['mp-1000']#kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
natms = len(mt.pstruct.sites)
r_max = 0.7
for l in logvars:
    sigma = 10**l
    dstruct = disturb_frac(struct_in, sigma=sigma)
    vis_structure(dstruct, title=str(sigma))
    plt.show()
    plt.close()
    dvec = diff_with_opes2(dstruct, oprs, r_max)
    ys.append(np.linalg.norm(dvec))

plt.plot(logvars, ys)
plt.yscale('log')




#%%

