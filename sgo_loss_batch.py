#%%
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
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
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
job = "2023-06-10/mp_20_2"   #!
task = 'gen'
jobdir = join(hydradir, job)
use_path = join(jobdir, f'eval_{task}.pt') #!

lengths, angles, num_atoms, frac_coords, atom_types, all_frac_coords_stack, all_atom_types_stack, eval_setting, time_out =output_eval(use_path)
lattices = lattice_params_to_matrix_torch(lengths[0], angles[0])
num = len(lattices)
print("jobdir: ", jobdir)
#%%
#[1] check each structure aaa
idx = 5
astruct_list = get_astruct_list(use_path, idx)
pstruct_list = [AseAtomsAdaptor.get_structure(a) for a in astruct_list]
mts = [MatTrans(p) for p in pstruct_list]
plt.plot([mt.sg[0] for mt in mts])
plt.title(f'[{idx}] {astruct_list[-1].get_chemical_formula()}')
plt.show()
plt.close()

#%%
# space group distribution
# check only the final product
# astruct_lists = get_astruct_all_list(use_path)
# astructs = [astruct_list[-1] for astruct_list in astruct_lists] # only the final product
astructs = get_astruct_end(use_path)
pstructs = [AseAtomsAdaptor.get_structure(a) for a in astructs] # only the final product
mts = [MatTrans(p) for p in pstructs]
sgs = [mt.sg[0] for mt in mts]
len_ = len(sgs)
plt.hist(np.array(sgs), bins=len_)
plt.title(f'space group distributions')
plt.show()
plt.close()


#%%
# symprec=0.09 # default: 0.01
# angle_tolerance=20.0 # default: 5.0
# mts = [MatTrans(p, symprec, angle_tolerance) for p in pstructs]
# sgs = [mt.sg[0] for mt in mts]
# len_ = len(sgs)
# # plt.hist(np.array(sgs), bins=len_)
# plt.plot(np.array(sgs))
# plt.title(f'space group distributions (symprec={symprec}, ang tol={angle_tolerance})')
# plt.show()
# plt.close()

#%%
run1=False

if run1:
    pstruct = pstructs[0]
    r_max=0.8
    sgloss_prod = SGO_Loss_Prod(r_max)
    sgloss_perm = SGO_Loss_Perm(r_max)
    grad=True
    frac = torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
    frac.requires_grad = grad
    mt = MatTrans(pstruct)
    opes = list(set(mt.spgops))
    # oprs = [op.rotation_matrix for op in opes]
    # opts = [op.translation_vector for op in opes]
    # oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
    oprs = torch.stack([torch.tensor(op.rotation_matrix) for op in opes])
    opts = torch.stack([torch.tensor(op.translation_vector) for op in opes])
    frac0 = torch.tensor(pstruct.frac_coords)
    frac0.requires_grad = grad
    loss_prod = sgloss_perm(frac0, oprs)
    loss_prod.backward()
    print('loss_prod: ', loss_prod)
    print('loss_prod.grad: ', loss_prod.grad)
    print('frac0.grad: ', frac0.grad)

    frac1 = torch.tensor(pstruct.frac_coords)
    frac1.requires_grad = grad
    loss_perm = sgloss_perm(frac1, oprs)
    loss_perm.backward()
    print('loss_perm: ', loss_perm)
    print('loss_perm.grad: ', loss_perm.grad)
    print('frac1.grad: ', frac1.grad)


#%%
run2=False
if run2:
    # impose Silicon's symmetry (sg=227)
    si = mpdata['mp-149']
    pstruct = pstructs[0]
    r_max=0.8
    sgloss_prod = SGO_Loss_Prod(r_max)
    sgloss_perm = SGO_Loss_Perm(r_max)
    grad=True
    frac = torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
    frac.requires_grad = grad
    mt = MatTrans(si)   #!
    opes = list(set(mt.spgops)) #!
    # oprs = [op.rotation_matrix for op in opes]
    # opts = [op.translation_vector for op in opes]
    # oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
    oprs = torch.stack([torch.tensor(op.rotation_matrix) for op in opes])
    opts = torch.stack([torch.tensor(op.translation_vector) for op in opes])
    frac0 = torch.tensor(pstruct.frac_coords)
    frac0.requires_grad = grad
    loss_prod = sgloss_perm(frac0, oprs)
    loss_prod.backward()
    print('loss_prod: ', loss_prod)
    print('loss_prod.grad: ', loss_prod.grad)
    print('frac0.grad: ', frac0.grad)

    frac1 = torch.tensor(pstruct.frac_coords)
    frac1.requires_grad = grad
    loss_perm = sgloss_perm(frac1, oprs)
    loss_perm.backward()
    print('loss_perm: ', loss_perm)
    print('loss_perm.grad: ', loss_perm.grad)
    print('frac1.grad: ', frac1.grad)

#%%
# use Si symmetry, use the batch input

# impose Silicon's symmetry (sg=227)
si = mpdata['mp-149']
pstruct = pstructs[0]
max_idx = 20
r_max=0.8
sgloss_prod = SGO_Loss_Prod(r_max)
sgloss_perm = SGO_Loss_Perm(r_max)
grad=True
frac = torch.tensor(frac_coords[0, :num_atoms[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
# frac.requires_grad = grad
mt = MatTrans(si)   #!
opes = list(set(mt.spgops)) #!
# oprs = [op.rotation_matrix for op in opes]
# opts = [op.translation_vector for op in opes]
# oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
oprs = torch.stack([torch.tensor(op.rotation_matrix) for op in opes]).float()
opts = torch.stack([torch.tensor(op.translation_vector) for op in opes]).float()
oprss = torch.concatenate([oprs for _ in range(num_atoms.shape[-1])])
optss = torch.concatenate([opts for _ in range(num_atoms.shape[-1])])
noprs = torch.tensor([oprs.shape[0] for  _ in range(num_atoms.shape[-1])])[None, :]
fracs0 = torch.tensor(frac_coords[0, :num_atoms[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)
# fracs0.requires_grad = grad
# loss_prod = 0
# for idx in range(len(num_atoms)):
#     sum_idx_bef = num_atoms[0, :idx].sum()
#     sum_idx_aft = num_atoms[0, :idx+1].sum()
#     frac0 = fracs0[sum_idx_bef:sum_idx_aft, :]
loss_prod = sgloss_prod(fracs0, num_atoms[:, :max_idx], oprss, noprs)
loss_prod.backward()
print('loss_prod: ', loss_prod)
print('loss_prod.grad: ', loss_prod.grad)
print('fracs0.grad: ', fracs0.grad)

fracs1 = torch.tensor(frac_coords[0, :num_atoms[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)
loss_perm = sgloss_perm(fracs1, num_atoms[:, :max_idx], oprss, noprs)
loss_perm.backward()
print('loss_prod: ', loss_perm)
print('loss_prod.grad: ', loss_perm.grad)
print('fracs1.grad: ', fracs1.grad)



#%% 
# comparison with the ground truth
# load reconstruction
task1 = 'recon'
use_path1 = join(jobdir, f'eval_{task1}.pt') #!

lengths1, angles1, num_atoms1, frac_coords1, atom_types1, all_frac_coords_stack1, all_atom_types_stack1, eval_setting1, time1 =output_eval(use_path1)
lattices1 = lattice_params_to_matrix_torch(lengths1[0], angles1[0])
num1 = len(lattices1)

idx = 5
astruct_list1 = get_astruct_list(use_path1, idx)
pstruct_list1 = [AseAtomsAdaptor.get_structure(a) for a in astruct_list1]
astructs1 = get_astruct_end(use_path1)
pstructs1 = [AseAtomsAdaptor.get_structure(a) for a in astructs1] # only the final product
symprec=0.09 # default: 0.01
angle_tolerance=20.0 # default: 5.0
mts1 = [MatTrans(p, symprec, angle_tolerance) for p in pstruct_list1]
plt.plot([mt.sg[0] for mt in mts1])
plt.title(f'[{idx}] {astruct_list1[-1].get_chemical_formula()}')
plt.show()
plt.close()

# symprec=0.09 # default: 0.01
# angle_tolerance=20.0 # default: 5.0
# mts1 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs1]
# sgs1 = [mt.sg[0] for mt in mts1]
# len1_ = len(sgs1)
# # plt.hist(np.array(sgs), bins=len_)
# plt.plot(np.array(sgs1))
# plt.title(f'space group distributions (symprec={symprec}, ang tol={angle_tolerance})')
# plt.show()
# plt.close()

#%%
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

# # symmetry distribution (for higher symmetry GT, the space group analyzer requires much longer time!!!)
# symprec=0.09 # default: 0.01
# angle_tolerance=20.0 # default: 5.0
# mts0 = [MatTrans(p, symprec, angle_tolerance) for p in pstructs0]
# sgs0 = [mt.sg[0] for mt in mts0]
# len0_ = len(sgs0)
# # plt.hist(np.array(sgs), bins=len_)
# plt.plot(np.array(sgs0))
# plt.title(f'space group distributions (symprec={symprec}, ang tol={angle_tolerance})')
# plt.show()
# plt.close()


#%%
# use the same oprs
si = mpdata['mp-149']
pstruct = pstructs[0]
max_idx = 20
r_max=0.8
sgloss_prod = SGO_Loss_Prod(r_max)
sgloss_perm = SGO_Loss_Perm(r_max)
grad=True
frac = torch.tensor(frac_coords1[0, :num_atoms1[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
# frac.requires_grad = grad
mt = MatTrans(si)   #!
opes = list(set(mt.spgops)) #!
# oprs = [op.rotation_matrix for op in opes]
# opts = [op.translation_vector for op in opes]
# oprs = [torch.tensor(opr, requires_grad=False) for opr in oprs]
oprs = torch.stack([torch.tensor(op.rotation_matrix) for op in opes]).float()
opts = torch.stack([torch.tensor(op.translation_vector) for op in opes]).float()
oprss = torch.concatenate([oprs for _ in range(num_atoms1.shape[-1])])
optss = torch.concatenate([opts for _ in range(num_atoms1.shape[-1])])
noprs = torch.tensor([oprs.shape[0] for  _ in range(num_atoms1.shape[-1])])[None, :]
fracs0 = torch.tensor(frac_coords[0, :num_atoms1[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)
# fracs0.requires_grad = grad
# loss_prod = 0
# for idx in range(len(num_atoms)):
#     sum_idx_bef = num_atoms[0, :idx].sum()
#     sum_idx_aft = num_atoms[0, :idx+1].sum()
#     frac0 = fracs0[sum_idx_bef:sum_idx_aft, :]
loss_prod = sgloss_prod(fracs0, num_atoms1[:, :max_idx], oprss, noprs)
loss_prod.backward()
print('loss_prod: ', loss_prod)
print('loss_prod.grad: ', loss_prod.grad)
print('fracs0.grad: ', fracs0.grad)


#%%
# use different oprs (I do not know why but this block takes time..!)
run3=False
if run3:
    si = mpdata['mp-149']
    pstruct = pstructs[0]
    max_idx = 20
    r_max=0.8
    sgloss_prod = SGO_Loss_Prod(r_max)
    sgloss_perm = SGO_Loss_Perm(r_max)
    grad=True
    frac = torch.tensor(frac_coords1[0, :num_atoms1[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)#.clone().detach().requires_grad_(grad)
    # frac.requires_grad = grad
    mt = MatTrans(si)   #!
    opess = [list(set(MatTrans(ps).spgops)) for ps in pstructs0] #!
    oprss = [torch.stack([torch.tensor(op.rotation_matrix) for op in opes]).float() for opes in opess]
    optss = [torch.stack([torch.tensor(op.translation_vector) for op in opes]).float() for opes in opess]
    oprss = torch.concatenate(oprss)
    optss = torch.concatenate(optss)
    noprs = torch.tensor([oprs.shape[0] for  _ in range(num_atoms1.shape[-1])])[None, :]
    fracs0 = torch.tensor(frac_coords[0, :num_atoms1[0, :max_idx].sum()], requires_grad=grad) #torch.tensor(pstruct.frac_coords)
    loss_prod0 = sgloss_prod(fracs0, num_atoms0[:, :max_idx], oprss, noprs)
    loss_prod0.backward()
    print('loss_prod: ', loss_prod0)
    print('loss_prod.grad: ', loss_prod0.grad)
    print('fracs0.grad: ', fracs0.grad)


#%%
# https://www.notion.so/231005-symmetry-enforcement-evaluation-9d71492bd7244f2bb682f76e5954bb90?pvs=4#fee2bcbabb90490cade476300015e5a7
# structure diffusion (Si only)
batch_size = 100
mpid = 'mp-149'
pstruct = mpdata[mpid]
pstructs00 = random.sample(pstructs0, batch_size)
# pstructs00 = pstructs0[:batch_size] # multiple different structures in GT
# pstructs00 = [pstruct for _ in range(batch_size)]   #! change here!
fcoords00 = torch.cat([torch.tensor(ps.frac_coords) for ps in pstructs00])[None, :].float()
num_atoms00 = torch.tensor([len(ps.sites) for ps in pstructs00])[None, :]
opess00 = [list(set(MatTrans(ps).spgops)) for ps in pstructs00] #!
oprss00_list = [torch.stack([torch.tensor(op.rotation_matrix) for op in opes]).float() for opes in opess00]
# optss00 = [torch.stack([torch.tensor(op.translation_vector) for op in opes]).float() for opes in opess00]
oprss00 = torch.concatenate(oprss00_list)
noprs00 = torch.tensor([len(oprs) for oprs in oprss00_list])[None, :]

# single loss 
r_max = 0.7
sgloss_prod = SGO_Loss_Prod(r_max)
sgloss_perm = SGO_Loss_Perm(r_max)
loss_prod = sgloss_prod(fcoords00[0], num_atoms00, oprss00, noprs00)
loss_prod.backward()
print('loss_prod: ', loss_prod)
print('loss_prod.grad: ', loss_prod.grad)
print('fracs0.grad: ', fracs0.grad)

# structure diffusion 
logvars = np.linspace(-2, 0, num=5) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys0, ys1 = [], []
frac0_list, grads0_list = [], []
frac1_list, grads1_list = [], []

# h = 1e-07
# tol = 1e-03
n_dstructs = 2
grad=True
for i, lv in enumerate(logvars):
    sigma = 10**lv
    # dstruct = diffuse_frac(pstruct, sigma=sigma)
    noisy_coords = (fcoords00.clone() + torch.rand_like(fcoords00)*sigma)%1 # [diffuse_frac(pstruct_stack, sigma=sigma) for _ in range(n_dstructs)]
    loss_0, loss_1 = torch.zeros(1),torch.zeros(1)
    grads_0, grads_1 = torch.zeros(1),torch.zeros(1)
    # vis_structure(dstruct, title=f"$log(\sigma)$={round(l, 7)}")
    # fracs_0, fracs_1 = noisy_coords.clone().detach().float(), noisy_coords.clone().detach().float()
    fracs_00 = torch.tensor(noisy_coords, requires_grad=grad)[0].float()
    fracs_01 = torch.tensor(noisy_coords, requires_grad=grad)[0].float()
    # fracs_0.requires_grad, fracs_1.requires_grad = grad, grad
    loss_0 = sgloss_prod(fracs_00, num_atoms00, oprss00, noprs00)
    # ys0.append(loss0)
    loss_0.backward()
    grads_0 = fracs_00.grad
    frac0_list.append(fracs_00.detach().numpy())
    # grads0_list.append(-grads_0.detach().numpy())
    loss_1 = sgloss_perm(fracs_01, num_atoms00, oprss00, noprs00)
    # ys1.append(loss1)
    loss_1.backward()
    grads1 = fracs_01.grad
    frac1_list.append(fracs_01.detach().numpy())
    # grads1_list.append(-grads1.detach().numpy())
    loss_0 += loss_0/n_dstructs
    loss_1 += loss_1/n_dstructs
    ys0.append(loss_0)
    ys1.append(loss_1)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ys0 = [y.detach().numpy() for y in ys0]
ys1 = [y.detach().numpy() for y in ys1]
ax.plot(logvars, ys0/max(ys0), label='prod')
ax.plot(logvars, ys1/max(ys1), label='perm')
ax.legend()
ax.set_ylabel(f"Mismatch term by space group operation")
ax.set_xlabel(f"$log(\sigma)$")
ax.set_title(f'batch size: {batch_size}')
# ax.set_yscale('log')
fig.patch.set_facecolor('white')


#%%
# structure diffusion (samme space group, diffferent structures)
# https://www.notion.so/231005-symmetry-enforcement-evaluation-9d71492bd7244f2bb682f76e5954bb90?pvs=4#98bff47cc958435b893048955ee85759
batch_size = 20
spg_number = 225
pstructs00_double = random.sample(list(mp_dicts[spg_number].values()), batch_size*2)
# Split the list into two halves
pstructs00 = pstructs00_double[:batch_size]
pstructs00_oprs = pstructs00_double[batch_size:]
# pstructs00 = pstructs0[:batch_size] # multiple different structures in GT
# pstructs00 = [pstruct for _ in range(batch_size)]   #! change here!
fcoords00 = torch.cat([torch.tensor(ps.frac_coords) for ps in pstructs00])[None, :].float()
num_atoms00 = torch.tensor([len(ps.sites) for ps in pstructs00])[None, :]
opess00 = [list(set(MatTrans(ps).spgops)) for ps in pstructs00_oprs] #!
oprss00_list = [torch.stack([torch.tensor(op.rotation_matrix) for op in opes]).float() for opes in opess00]
# optss00 = [torch.stack([torch.tensor(op.translation_vector) for op in opes]).float() for opes in opess00]
oprss00 = torch.concatenate(oprss00_list)
noprs00 = torch.tensor([len(oprs) for oprs in oprss00_list])[None, :]

# single loss 
r_max = 0.7
sgloss_prod = SGO_Loss_Prod(r_max)
sgloss_perm = SGO_Loss_Perm(r_max)
loss_prod = sgloss_prod(fcoords00[0], num_atoms00, oprss00, noprs00)
loss_prod.backward()
print('loss_prod: ', loss_prod)
print('loss_prod.grad: ', loss_prod.grad)
print('fracs0.grad: ', fracs0.grad)

# structure diffusion 
logvars = np.linspace(-2, 0, num=51) #range(10, -5, -1)
xs = [10**l for l in logvars]
ys0, ys1 = [], []
frac0_list, grads0_list = [], []
frac1_list, grads1_list = [], []

# h = 1e-07
# tol = 1e-03
n_dstructs = 2
grad=True
for i, lv in enumerate(logvars):
    sigma = 10**lv
    # dstruct = diffuse_frac(pstruct, sigma=sigma)
    noisy_coords = (fcoords00.clone() + torch.rand_like(fcoords00)*sigma)%1 # [diffuse_frac(pstruct_stack, sigma=sigma) for _ in range(n_dstructs)]
    loss_0, loss_1 = torch.zeros(1),torch.zeros(1)
    grads_0, grads_1 = torch.zeros(1),torch.zeros(1)
    # vis_structure(dstruct, title=f"$log(\sigma)$={round(l, 7)}")
    # fracs_0, fracs_1 = noisy_coords.clone().detach().float(), noisy_coords.clone().detach().float()
    fracs_00 = torch.tensor(noisy_coords, requires_grad=grad)[0].float()
    fracs_01 = torch.tensor(noisy_coords, requires_grad=grad)[0].float()
    # fracs_0.requires_grad, fracs_1.requires_grad = grad, grad
    loss_0 = sgloss_prod(fracs_00, num_atoms00, oprss00, noprs00)
    # ys0.append(loss0)
    loss_0.backward()
    grads_0 = fracs_00.grad
    frac0_list.append(fracs_00.detach().numpy())
    # grads0_list.append(-grads_0.detach().numpy())
    loss_1 = sgloss_perm(fracs_01, num_atoms00, oprss00, noprs00)
    # ys1.append(loss1)
    loss_1.backward()
    grads1 = fracs_01.grad
    frac1_list.append(fracs_01.detach().numpy())
    # grads1_list.append(-grads1.detach().numpy())
    loss_0 += loss_0/n_dstructs
    loss_1 += loss_1/n_dstructs
    ys0.append(loss_0)
    ys1.append(loss_1)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ys0 = [y.detach().numpy() for y in ys0]
ys1 = [y.detach().numpy() for y in ys1]
ax.plot(logvars, ys0/max(ys0), label='prod')
ax.plot(logvars, ys1/max(ys1), label='perm')
ax.legend()
ax.set_ylabel(f"Mismatch term by space group operation")
ax.set_xlabel(f"$log(\sigma)$")
ax.set_title(f'batch size: {batch_size} | sg: {spg_number}')
# ax.set_yscale('log')
fig.patch.set_facecolor('white')




#%%