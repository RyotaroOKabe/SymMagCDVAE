#%%
"""
https://www.notion.so/230403-04-use-e3nn-to-give-irreps-of-latent-variables-and-the-symmetry-operations-4c34c7e20616494aa39100c2e4c6529d

"""

from e3nn import o3
from e3nn.o3 import Irreps, Irrep
from torch import Tensor
import torch
import numpy as np
from pymatgen.core.structure import Structure
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
import pickle as pkl
import matplotlib.pyplot as plt

from utils.utils_plot import vis_structure
from utils.utils_material import MatSym, MatTrans, distance_sorted, Rx, Ry, Rz, rotate_cart, switch_latvecs
from utils.utils_e3nn import operation_loss, SimpleFTP, diffuse_frac, cerror_opes
import math as m
Pi = m.pi

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
cosn = mpdata['mp-20536']
#%%
# [0]  initial test
net = SimpleFTP()
logvars = np.linspace(-3, 0, num=41) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys = []
struct_in = cosn #mpdata['mp-1000']#kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
natms = len(mt.pstruct.sites)
r_max = 0.7
for l in logvars:
    sigma = 10**l
    dstruct = diffuse_frac(struct_in, sigma=sigma)
    vis_structure(dstruct, title=f"$log(\sigma)$={round(l, 7)}")
    plt.show()
    plt.close()
    dvec = cerror_opes(dstruct, oprs, net, r_max)
    ys.append(np.linalg.norm(dvec))

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(logvars, ys)
ax.set_ylabel(f"Mismatch by Space group operation")
ax.set_xlabel(f"$log(\sigma)$")
# ax.set_yscale('log')
fig.patch.set_facecolor('white')

#%%
# [1] rotation of 3x3 rotation matrix
# (1-1)
theta = 2*Pi/3
phi = Pi/2
rot = Rz(theta)
R = Ry(phi)
Rinv = np.linalg.inv(R)
rot_r1 = R@rot@Rinv # correct form!
rot_r2 = Rinv@rot@R
rot_ref = Rx(theta)
tol = 1e-03
print('[case 1]')
if np.allclose(rot_r1, rot_ref, atol=tol):
    print('rot_r1 is correct')
    print(rot_r1)
if np.allclose(rot_r2, rot_ref, atol=tol):
    print('rot_r2 is correct')
    print(rot_r2)

# (1-2)
theta = 2*Pi/3
phi = Pi/2
rot = Rx(theta)
R = Ry(phi)
Rinv = np.linalg.inv(R)
rot_r1 = R@rot@Rinv # correct form!
rot_r2 = Rinv@rot@R
rot_ref = Rz(-theta)
tol = 1e-03
print('[case 2]')
if np.allclose(rot_r1, rot_ref, atol=tol):
    print('rot_r1 is correct')
    print(rot_r1)
if np.allclose(rot_r2, rot_ref, atol=tol):
    print('rot_r2 is correct')
    print(rot_r2)
    
# (1-3)
# rot = np.array(o3.rand_matrix())
# R = np.array(o3.rand_matrix())
# Rinv = np.linalg.inv(R)
# rot_r1 = R@rot@Rinv # correct form!
# rot_r2 = Rinv@rot@R
# rot_ref = Rz(-theta)
# tol = 1e-03
# print('[case 3]')
# if np.allclose(rot_r1, rot_ref, atol=tol):
#     print('rot_r1 is correct')
#     print(rot_r1)
# if np.allclose(rot_r2, rot_ref, atol=tol):
#     print('rot_r2 is correct')
#     print(rot_r2)
#%%
# space group opes
pstruct = cosn #mpdata['mp-1000']#kstruct
mt = MatTrans(struct_in)
opes = list(set(mt.spgops))
oprs = [op.rotation_matrix for op in opes]
opts = [op.translation_vector for op in opes]
opr = oprs[0]
frac = pstruct.frac_coords  # frac = cart @ np.linalg.inv(lat)
cart = pstruct.cart_coords  # cart = frac @ lat
lat = pstruct.lattice.matrix
spec = pstruct.species
r_max = 0.7
net = SimpleFTP()
tol = 1e-03
#%%
# (2-1) z-axis rotation
phi = Pi/2
R = Rz(phi)
Rinv = np.linalg.inv(R)
opr_r = R@opr@Rinv
frac_r = frac@R.T
pstruct_r = Structure(
    lattice=lat,
    species=spec,
    coords=frac_r,
    coords_are_cartesian=False
)
loss0 = operation_loss(pstruct, opr, net, r_max)
loss1 = operation_loss(pstruct_r, opr_r, net, r_max)
print("[case 2-1] all close: ", np.allclose(loss0, loss1, atol=tol))
#%%
# (2-2) x-axis rotation
phi = 3*Pi/2
R = Rx(phi)
Rinv = np.linalg.inv(R)
opr_r = R@opr@Rinv
frac_r = frac@R.T
pstruct_r = Structure(
    lattice=lat,
    species=spec,
    coords=frac_r,
    coords_are_cartesian=False
)
loss0 = operation_loss(pstruct, opr, net, r_max)
loss1 = operation_loss(pstruct_r, opr_r, net, r_max)
print("[case 2-2] all close: ", np.allclose(loss0, loss1, atol=tol))

#%%
# (2-3) random rotation
R = np.array(o3.rand_matrix())
Rinv = np.linalg.inv(R)
opr_r = R@opr@Rinv.T
frac_r = frac@R.T
pstruct_r = Structure(
    lattice=lat,#@R.T,
    species=spec,
    coords=frac_r,
    coords_are_cartesian=False
)
loss0 = operation_loss(pstruct, opr, net, r_max)
loss1 = operation_loss(pstruct_r, opr, net, r_max, Rot=R)
# loss2 = operation_loss(pstruct, opr_r, net, r_max)
# loss3 = operation_loss(pstruct_r, opr, net, r_max)
print("[case 2-3] all close: ", np.allclose(loss0, loss1, atol=tol))
print('loss0: ', loss0)
print('loss1: ', loss1)
# print('loss2: ', loss2)
# print('loss3: ', loss3)

#%%
# operation_loss and simpleFTP might cause the issue if two of them are rotation invariant
astruct = Atoms(list(map(lambda x: x.symbol, pstruct.species)),

                    positions=frac, cell=np.eye(3), pbc=True) 
astruct_r = Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions=frac_r%1, cell=np.eye(3), pbc=True) 
edge_vec = neighbor_list("D", a = astruct, cutoff = r_max, self_interaction = True)
edge_vec_r = neighbor_list("D", a = astruct_r, cutoff = r_max, self_interaction = True)

astruct_r2 = Atoms(list(map(lambda x: x.symbol, pstruct.species)),  #This is the one!
                    positions=frac_r, cell=np.eye(3)@R.T, pbc=True) 
astruct_r3 = Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions=frac_r%1, cell=np.eye(3)@R.T, pbc=True) 
edge_vec_r2 = neighbor_list("D", a = astruct_r2, cutoff = r_max, self_interaction = True)
# print("[case 2-3] all close: ", np.allclose(edge_vec_r, edge_vec@R.T, atol=tol))
l_0 = np.array(astruct.get_cell())
c_0 = np.array(astruct.get_positions())
l_r2 = np.array(astruct_r2.get_cell())
c_r2 = np.array(astruct_r2.get_positions())
c_r2_2 = c_r2@np.linalg.inv(l_r2)
l_r3 = np.array(astruct_r3.get_cell())
c_r3 = np.array(astruct_r3.get_positions())
c_r3_2 = c_r3@np.linalg.inv(l_r3)
print(edge_vec_r2)
r_max2 = 0.5
edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = r_max2, self_interaction = True)
edge_src_r2, edge_dst_r2, edge_shift_r2, edge_vec_r2, edge_len_r2 = neighbor_list("ijSDd", a = astruct_r2, cutoff = r_max2, self_interaction = True)
