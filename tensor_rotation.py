#%%
import torch
import numpy as np
from e3nn.o3 import wigner_D
from e3nn.o3._wigner import _so3_clebsch_gordan

#%%
lss = [[1], [0, 1], [0, 1, 2]]
rotation = torch.tensor([0.5, 0.2, 0.1])
dims = [sum([2*l+1 for l in ls]) for ls in lss]

tensor = torch.tensor(np.random.rand(*dims))

print(f'Create random tensor of rank {len(lss)}')
for i, ls in enumerate(lss):
    print(f'Axis {i} of the tensor is in the basis of spherical harmonic with l\'s equal {ls}')
print(tensor)

#%%
WDss = [[wigner_D(l, *rotation).type(tensor.dtype) for l in ls] for ls in lss]
Ds = [torch.block_diag(*WDs) for WDs in WDss]


#%%
rotated_tensor = torch.clone(tensor)
print('check0 rt: ', rotated_tensor.shape)
for i, D in enumerate(Ds):
    rotated_tensor = rotated_tensor.transpose(i, 0)
    print('check1 rt: ', rotated_tensor.shape)
    print('check2 D: ', D.shape)
    rotated_tensor = torch.einsum('ij,j...->i...', D, rotated_tensor)
    print('check3 rtD: ', rotated_tensor.shape)
    rotated_tensor = rotated_tensor.transpose(i, 0)
    print('check4 rt: ', rotated_tensor.shape)
print(rotated_tensor)

#%%
ls_out = [0]
dim_out = 1
CG = torch.tensor([1.0], dtype = tensor.dtype)
for ls, dim in zip(lss, dims):
    ls_tmp = []
    CG_tmp = torch.zeros((dim_out, dim, dim_out * dim), dtype = tensor.dtype)
    start1 = 0
    start3 = 0
    for l_out in ls_out:
        stop1 = start1 + 2 * l_out + 1
        start2 = 0
        for l in ls:
            stop2 = start2 + 2 * l + 1
            for l_tmp in range(abs(l-l_out), l+l_out+1):
                stop3 = start3 + 2 * l_tmp + 1
                ls_tmp.append(l_tmp)
                CG_tmp[start1:stop1, start2:stop2, start3:stop3] = _so3_clebsch_gordan(int(l_out), int(l), l_tmp) * (2*l_tmp + 1) ** 0.5
                start3 = stop3
            start2 = stop2
        start1 = stop1
    ls_out = ls_tmp
    dim_out = sum([2 * l_out + 1 for l_out in ls_out])
    CG = torch.einsum('...i,ijk->...jk', CG, CG_tmp)

#%%

Ds = [wigner_D(l_out, *rotation).type(tensor.dtype) for l_out in ls_out]
D = torch.block_diag(*Ds)

#%%
irrep_tensor = torch.einsum('...k,...->k', CG, tensor)
rotated_irrep_tensor = torch.einsum('ij,j->i', D, irrep_tensor)
E3_rotated_tensor = torch.einsum('...k,k->...', CG, rotated_irrep_tensor)
print(E3_rotated_tensor)

#%%
