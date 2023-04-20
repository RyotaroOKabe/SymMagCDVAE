#%%
"""
https://www.notion.so/230406-relationship-between-the-physical-rotation-and-the-space-group-operations-6596cff40c044f90be2e7ceaeb42d638
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
from data2.generative.magcdvae.utils.utils_e3nn import operation_loss, SimpleFTP, diffuse_frac, cerror_opes
import math as m
Pi = m.pi
tol = 1e-03

#%%
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
mp_dicts = pkl.load(open('data/mp_dicts.pkl', 'rb'))

#%%
# pstruct prep
cosn = mpdata['mp-20536']
silicon = mpdata['mp-149']

kfrac = np.array([
    [0,0,0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0]
])
hex_latice = np.array([
    [6,0,0],
    [3, 3*np.sqrt(3),0],
    [0,0,4]
])
kspecies = np.array([6, 6, 6])  
kstruct = Structure(
    lattice=hex_latice,
    species=kspecies,
    coords=kfrac,
    coords_are_cartesian=False
)

unitcube = np.eye(3)
frac_coordA = np.array([[0,0,0]])
frac_coordB = np.array([[0,0,0],
                        [0.5,0.5,0.5]])

frac_coordR = np.random.rand(4,3)

simple1 = Structure(
    lattice=unitcube,
    species=['H' for _ in range(len(frac_coordA))],
    coords=frac_coordA,
    coords_are_cartesian=False
)

simple2 = Structure(
    lattice=unitcube,
    species=['H' for _ in range(len(frac_coordB))],
    coords=frac_coordB,
    coords_are_cartesian=False
)

simple3 = Structure(
    lattice=unitcube,
    species=[Atom(i).symbol for i in range(len(frac_coordB))],
    coords=frac_coordB,
    coords_are_cartesian=False
)

rstruct = Structure(
    lattice=unitcube,
    species=[Atom(i).symbol for i in range(len(frac_coordR))],
    coords=frac_coordR,
    coords_are_cartesian=False
)

#%%
pstruct1 = simple1
pstruct2 = simple2
pstruct3 = simple3
pstruct4 = silicon
pstruct5 = cosn
pstruct6 = kstruct
pstruct7 = rstruct

mt1 = MatTrans(pstruct1)
opes1 = list(set(mt1.spgops))
oprs1 = [op.rotation_matrix for op in opes1]
opts1 = [op.translation_vector for op in opes1]

mt2 = MatTrans(pstruct2)
opes2 = list(set(mt2.spgops))
oprs2 = [op.rotation_matrix for op in opes2]
opts2 = [op.translation_vector for op in opes2]

mt3 = MatTrans(pstruct3)
opes3 = list(set(mt3.spgops))
oprs3 = [op.rotation_matrix for op in opes3]
opts3 = [op.translation_vector for op in opes3]

mt4 = MatTrans(pstruct4)
opes4 = list(set(mt4.spgops))
oprs4 = [op.rotation_matrix for op in opes4]
opts4 = [op.translation_vector for op in opes4]

mt5 = MatTrans(pstruct5)
opes5 = list(set(mt5.spgops))
oprs5 = [op.rotation_matrix for op in opes5]
opts5 = [op.translation_vector for op in opes5]

mt6 = MatTrans(pstruct6)
opes6 = list(set(mt6.spgops))
oprs6 = [op.rotation_matrix for op in opes6]
opts6 = [op.translation_vector for op in opes6]

mt7 = MatTrans(pstruct7)
opes7 = list(set(mt7.spgops))
oprs7 = [op.rotation_matrix for op in opes7]
opts7 = [op.translation_vector for op in opes7]

print("pstruct1: ", mt1.sg, len(set(opes1)))
print("pstruct2: ", mt2.sg, len(set(opes2)))
print("pstruct3: ", mt3.sg, len(set(opes3)))
print("pstruct4: ", mt4.sg, len(set(opes4)))
print("pstruct5: ", mt5.sg, len(set(opes5)))
print("pstruct6: ", mt6.sg, len(set(opes6)))
print("pstruct7: ", mt7.sg, len(set(opes7)))

#%%
# rotate structure fucntion
def cart_rotation(pstruct, R, rot_lattice=True):
    cart = pstruct.cart_coords
    lat = pstruct.lattice.matrix
    spec = pstruct.species
    cart1 = cart@R.T
    if rot_lattice:
        lat1 = lat@R.T
    else:
        lat1 = lat
    pstruct_out = Structure(
        lattice=lat1,
        species=spec,
        coords=cart1,
        coords_are_cartesian=True
    )
    return pstruct_out

def get_opes(pstruct):
    mt = MatTrans(pstruct)
    opes = list(set(mt.spgops))
    oprs = [op.rotation_matrix for op in opes]
    opts = [op.translation_vector for op in opes]
    return mt, opes, oprs, opts

def test_rotation(pstruct, R):
    mt = MatTrans(pstruct)
    opes = list(set(mt.spgops))
    pstructa = cart_rotation(pstruct, R, rot_lattice=True)
    pstructb = cart_rotation(pstruct, R, rot_lattice=False)
    mta, opesa, oprsa, optsa = get_opes(pstructa)
    mtb, opesb, oprsb, optsb = get_opes(pstructb)
    print("opes: ", len(opes), len(opesa), len(opesb))
    print("sg: ", mt.sg, mta.sg, mtb.sg)
    return pstructa, pstructb, mta, opesa, oprsa, optsa, mtb, opesb, oprsb, optsb

def nondegenerate_arraylist(list1, tol):
    out = [list1[0]]
    for arr1 in list1:
        score = 0
        for arr2 in out:
            if np.allclose(arr1, arr2, atol=tol):
                score += 1
        if score < 1:
            out.append(arr1)
    return out

def oprs_table(list1, list2, tol):
    n1 = len(list1)
    n2 = len(list2)
    table = np.zeros((n1, n2))
    for i, arr1 in enumerate(list1):
        for j, arr2 in enumerate(list2):
            if np.allclose(arr1, arr2, atol=tol):
                table[i, j]+=1
    return table

def intersection_arraylist(list1, list2, tol):
    out = []
    for i, arr1 in enumerate(list1):
        for j, arr2 in enumerate(list2):
            if np.allclose(arr1, arr2, atol=tol):
                out.append(arr1)
    return nondegenerate_arraylist(out, tol)
            
#%%
# rotate both lattice and cell (rot_lattice=True)
# [1] rotation around x-axis 
theta = Tensor(2*Pi*Tensor(np.random.rand(1)))
R = np.array(o3.matrix_x(theta)).reshape((3,3))
print('[1]')
pstruct1a, pstruct1b, mt1a, opes1a, oprs1a, opts1a, mt1b, opes1b, oprs1b, opts1b = test_rotation(pstruct1, R)
print('[2]')
pstruct2a, pstruct2b, mt2a, opes2a, oprs2a, opts2a, mt2b, opes2b, oprs2b, opts2b = test_rotation(pstruct2, R)
print('[3]')
pstruct3a, pstruct3b, mt3a, opes3a, oprs3a, opts3a, mt3b, opes3b, oprs3b, opts3b = test_rotation(pstruct3, R)
print('[4]')
pstruct4a, pstruct4b, mt4a, opes4a, oprs4a, opts4a, mt4b, opes4b, oprs4b, opts4b = test_rotation(pstruct4, R)
print('[5]')
pstruct5a, pstruct5b, mt5a, opes5a, oprs5a, opts5a, mt5b, opes5b, oprs5b, opts5b = test_rotation(pstruct5, R)
print('[6]')
pstruct6a, pstruct6b, mt6a, opes6a, oprs6a, opts6a, mt6b, opes6b, oprs6b, opts6b = test_rotation(pstruct6, R)
print('[7]')
pstruct7a, pstruct7b, mt7a, opes7a, oprs7a, opts7a, mt7b, opes7b, oprs7b, opts7b = test_rotation(pstruct7, R)

#%%
# [2] rotation around y-axis 
theta = Tensor(2*Pi*Tensor(np.random.rand(1)))
R = np.array(o3.matrix_y(theta)).reshape((3,3))
print('[1]')
pstruct1a, pstruct1b, mt1a, opes1a, oprs1a, opts1a, mt1b, opes1b, oprs1b, opts1b = test_rotation(pstruct1, R)
print('[2]')
pstruct2a, pstruct2b, mt2a, opes2a, oprs2a, opts2a, mt2b, opes2b, oprs2b, opts2b = test_rotation(pstruct2, R)
print('[3]')
pstruct3a, pstruct3b, mt3a, opes3a, oprs3a, opts3a, mt3b, opes3b, oprs3b, opts3b = test_rotation(pstruct3, R)
print('[4]')
pstruct4a, pstruct4b, mt4a, opes4a, oprs4a, opts4a, mt4b, opes4b, oprs4b, opts4b = test_rotation(pstruct4, R)
print('[5]')
pstruct5a, pstruct5b, mt5a, opes5a, oprs5a, opts5a, mt5b, opes5b, oprs5b, opts5b = test_rotation(pstruct5, R)
print('[6]')
pstruct6a, pstruct6b, mt6a, opes6a, oprs6a, opts6a, mt6b, opes6b, oprs6b, opts6b = test_rotation(pstruct6, R)
print('[7]')
pstruct7a, pstruct7b, mt7a, opes7a, oprs7a, opts7a, mt7b, opes7b, oprs7b, opts7b = test_rotation(pstruct7, R)

#%%
# [3] rotation around z-axis
theta = Tensor(2*Pi*Tensor(np.random.rand(1)))
R = np.array(o3.matrix_z(theta)).reshape((3,3))
print('[1]')
pstruct1a, pstruct1b, mt1a, opes1a, oprs1a, opts1a, mt1b, opes1b, oprs1b, opts1b = test_rotation(pstruct1, R)
print('[2]')
pstruct2a, pstruct2b, mt2a, opes2a, oprs2a, opts2a, mt2b, opes2b, oprs2b, opts2b = test_rotation(pstruct2, R)
print('[3]')
pstruct3a, pstruct3b, mt3a, opes3a, oprs3a, opts3a, mt3b, opes3b, oprs3b, opts3b = test_rotation(pstruct3, R)
print('[4]')
pstruct4a, pstruct4b, mt4a, opes4a, oprs4a, opts4a, mt4b, opes4b, oprs4b, opts4b = test_rotation(pstruct4, R)
print('[5]')
pstruct5a, pstruct5b, mt5a, opes5a, oprs5a, opts5a, mt5b, opes5b, oprs5b, opts5b = test_rotation(pstruct5, R)
print('[6]')
pstruct6a, pstruct6b, mt6a, opes6a, oprs6a, opts6a, mt6b, opes6b, oprs6b, opts6b = test_rotation(pstruct6, R)
print('[7]')
pstruct7a, pstruct7b, mt7a, opes7a, oprs7a, opts7a, mt7b, opes7b, oprs7b, opts7b = test_rotation(pstruct7, R)

#%%
# [4] rotation around arbitrary axis
R = np.array(o3.rand_matrix(1)).reshape((3,3))
print('[1]')
pstruct1a, pstruct1b, mt1a, opes1a, oprs1a, opts1a, mt1b, opes1b, oprs1b, opts1b = test_rotation(pstruct1, R)
print('[2]')
pstruct2a, pstruct2b, mt2a, opes2a, oprs2a, opts2a, mt2b, opes2b, oprs2b, opts2b = test_rotation(pstruct2, R)
print('[3]')
pstruct3a, pstruct3b, mt3a, opes3a, oprs3a, opts3a, mt3b, opes3b, oprs3b, opts3b = test_rotation(pstruct3, R)
print('[4]')
pstruct4a, pstruct4b, mt4a, opes4a, oprs4a, opts4a, mt4b, opes4b, oprs4b, opts4b = test_rotation(pstruct4, R)
print('[5]')
pstruct5a, pstruct5b, mt5a, opes5a, oprs5a, opts5a, mt5b, opes5b, oprs5b, opts5b = test_rotation(pstruct5, R)
print('[6]')
pstruct6a, pstruct6b, mt6a, opes6a, oprs6a, opts6a, mt6b, opes6b, oprs6b, opts6b = test_rotation(pstruct6, R)
print('[7]')
pstruct7a, pstruct7b, mt7a, opes7a, oprs7a, opts7a, mt7b, opes7b, oprs7b, opts7b = test_rotation(pstruct7, R)


#%%










#%%