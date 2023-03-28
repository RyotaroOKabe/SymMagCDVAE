#%%
from e3nn import o3
from e3nn.o3 import Irrep, Irreps
from e3nn.io import CartesianTensor
import torch
from torch import Tensor
import math as m
tol = 1e-03

#%%
ct = CartesianTensor('ij=ij')

#%%
# trace
x1 = ct.to_cartesian(Tensor([1,0,0,0,0,0,0,0,0]))

# antisymmetry
y1 = ct.to_cartesian(Tensor([0,1,0,0,0,0,0,0,0]))
y2 = ct.to_cartesian(Tensor([0,0,1,0,0,0,0,0,0]))
y3 = ct.to_cartesian(Tensor([0,0,0,1,0,0,0,0,0]))

# symmetric traceless
z1 = ct.to_cartesian(Tensor([0,0,0,0,1,0,0,0,0]))
z2 = ct.to_cartesian(Tensor([0,0,0,0,0,1,0,0,0]))
z3 = ct.to_cartesian(Tensor([0,0,0,0,0,0,1,0,0]))
z4 = ct.to_cartesian(Tensor([0,0,0,0,0,0,0,1,0]))
z5 = ct.to_cartesian(Tensor([0,0,0,0,0,0,0,0,1]))

xyz=[x1, y1, y2, y3, z1, z2, z3, z4, z5]
cmatrix = torch.stack([m.reshape(-1) for m in xyz])

#%%
tr = 3 ** -0.5
tw = 2 ** -0.5
irrep2tens = Tensor([[    tr,  0,   0,   0,      tr,  0,  0,   0,     tr],
                                [     0,  0,   0,   0,       0, tw,  0, -tw,      0],
                                [     0,  0, -tw,   0,       0,  0, tw,   0,      0],
                                [     0, tw,   0, -tw,       0,  0,  0,   0,      0],
                                [     0,  0,  tw,   0,       0,  0, tw,   0,      0],
                                [     0, tw,   0,  tw,       0,  0,  0,   0,      0],
                                [-tw*tr,  0,   0,   0, 2*tw*tr,  0,  0,   0, -tw*tr],
                                [     0,  0,   0,   0,       0, tw,  0,  tw,      0],
                                [   -tw,  0,   0,   0,       0,  0,  0,   0,     tw]])

print([tw, tr, tw*tr])

torch.allclose(cmatrix, irrep2tens)

#%%
data = torch.randn(1, 9) # (N, 3**2)

#1 irreps -> cart -> rot
opes = o3.rand_matrix(1) # (M, 3, 3)
# opes = torch.eye(3).reshape(1,3,3)
output1 = torch.einsum('ni, ij -> nj', data, cmatrix)
output1 = output1.reshape((-1,3,3)) # (N, 3,3)
output1 = torch.einsum('nij, mkj -> nmik', output1, opes)

# #2 irreps -> rot -> cart
D = ct.D_from_matrix(opes) # (M, 9 ,9)
M, _, _ = D.shape
N, _ = data.shape
data2 = torch.einsum('mij,nj->nmi', D, data)
output2 = torch.einsum('nmi, ij-> nmj', data2, cmatrix)
output2 = output2.reshape(N, M, 3,3)


#%%
data = torch.randn(9) # (N, 3**2)

#1 irreps -> cart -> rot
opes = o3.rand_matrix(1)
opes = o3.matrix_z(Tensor([0.05*m.pi/2]))#torch.eye(3)
opes1 = opes.reshape((3,3)) # (M, 3, 3)
# opes = torch.eye(3).reshape(1,3,3)
output1 = data@cmatrix
output1 = output1.reshape((3,3)) # (N, 3,3)
output1 = output1@opes1.T    #!!!

# #2 irreps -> rot -> cart
D = ct.D_from_matrix(opes).reshape(9,9) # (M, 9 ,9)
data2 = torch.einsum('ij,j->i', D, data)
output2 = data2@cmatrix
output2 = output2.reshape(3,3)

print(torch.allclose(output1, output2))
print(output1)
print(output2)

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
import torch
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

#%%
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
