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

def operation_loss(pstruct, opr, net, r_max):
    frac = Tensor(pstruct.frac_coords)
    opr = Tensor(opr)
    frac0 = frac
    frac1 = frac@opr.T
    assert frac0.shape == frac1.shape
    astruct0 = Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                        positions=frac0, cell=np.eye(3), pbc=True) 
    astruct1 = Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                        positions=frac1, cell=np.eye(3), pbc=True) 
    _, _, _, edge_vec0, _ = neighbor_list("ijSDd", a = astruct0, cutoff = r_max, self_interaction = True)
    _, _, _, edge_vec1, _ = neighbor_list("ijSDd", a = astruct1, cutoff = r_max, self_interaction = True)
    out0 = net(Tensor(edge_vec0))
    out1 = net(Tensor(edge_vec1))
    return out1-out0


class SimpleTP(torch.nn.Module):
    def __init__(self, 
                 irreps_in1='1x1o',
                 irreps_in2='1x1o',
                 ):
        super().__init__()
        
        self.irreps_in1=Irreps(irreps_in1)
        self.irreps_in2=Irreps(irreps_in2)
    
        self.tp = o3.FullTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in1
        )
        self.irreps_out=self.tp.irreps_out

    def forward(self, frac):
        output = self.tp(frac, frac)
        return torch.sum(output, dim=0)

def diffuse_frac(pstruct, sigma=0.1):
    frac = pstruct.frac_coords
    lat = pstruct.lattice.matrix
    spec = pstruct.species
    dist = np.random.normal(loc=0.0, scale=sigma, size=frac.shape)
    frac1 = frac + dist
    struct_out = Structure(
        lattice=lat,
        species=spec,
        coords=frac1,
        coords_are_cartesian=False
    )
    return struct_out

def difference_by_opes(pstruct, oprs, net, r_max):
    vector = []
    for opr in oprs:
        diff = operation_loss(pstruct, opr, net, r_max)
        vector.append(diff)
    return torch.cat(vector)


if __name__ == "__main__":
    mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
    mpids = sorted(list(mpdata.keys()))
    cosn = mpdata['mp-20536']
    net = SimpleTP()
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
        dvec = difference_by_opes(dstruct, oprs, net, r_max)
        ys.append(np.linalg.norm(dvec))

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(logvars, ys)
    ax.set_ylabel(f"Mismatch by Space group operation")
    ax.set_xlabel(f"$log(\sigma)$")
    # ax.set_yscale('log')
    fig.patch.set_facecolor('white')
