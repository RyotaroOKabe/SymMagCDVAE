#%%
"""
https://www.notion.so/230513-include-the-space-group-data-into-the-dataset-471dcd60234b4df384268f776e6e1b06
Add sgo data into the csv file. 

"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.analyzer import *
from ase import Atom, Atoms
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
# from cdvae.pl_modules.space_group import struct2spgop
from utils.utils_material import MatTrans

data_dir = '/home/rokabe/data2/generative/symcdvae/data/mp_20'
file_name = 'train.csv'
file = data_dir + '/' + file_name

#%%
df = pd.read_csv(file, index_col=0)
print(df.keys())
print(f"{file}: {len(df)}")
df
#%%
# useful functions
def str2pymatgen(cif):
    pstruct=Structure.from_str(cif, "CIF")
    return pstruct

# pymatgen > ase.Atom
def pymatgen2ase(pstruct):
    return Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions = pstruct.cart_coords.copy(),
                    cell = pstruct.lattice.matrix.copy(), 
                    pbc=True)
    
# plot structure
def plot_structures(astruct, title=None):
    formula = astruct.get_chemical_formula()
    symbols = np.unique(list(astruct.symbols))
    z = dict(zip(symbols, range(len(symbols))))
    fig, ax = plt.subplots(figsize=(6,5))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(astruct.symbols)]))]
    plot_atoms(astruct, ax, radii=0.25, colors=color, rotation=('30x,30y,0z'))
    ax.set_xlabel(r'$x_1\ (\AA)$')
    ax.set_ylabel(r'$x_2\ (\AA)$')
    ax.set_title(f"{title} / {formula.translate(sub)}")

# copy from data_utils.py
def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal

def struct2oprs(pstruct):
    sga = SpacegroupAnalyzer(pstruct)
    spgops=sga.get_space_group_operations()
    opes = list(set(spgops))
    oprs = [op.rotation_matrix for op in opes]
    return oprs

#%%
idx = 100
df_line = df.iloc[idx]
mpid = df_line['material_id']
# formula = df_line['pretty_formula']
crystal_str = df_line['cif']
pstruct = str2pymatgen(crystal_str)
astruct = pymatgen2ase(pstruct)
pstruct = build_crystal(crystal_str)
mt = MatTrans(pstruct)
oprs = np.stack(mt.oprs)


#%%
data_dir = '/home/rokabe/data2/generative/symcdvae/data/mp_20'
data_dir_new = '/home/rokabe/data2/generative/symcdvae/data/mp_20_sgo'
files =['val', 'test']
for file in files:
    df_ = pd.read_csv(os.path.join(data_dir, file+'.csv'), index_col=0)
    print(df.keys())
    print(f"{file}: {len(df)}")
    df_['sgo'] = df_['cif'].map(lambda x: np.stack(struct2oprs(build_crystal(x, niggli=True, primitive=False))))
    df_.to_csv(os.path.join(data_dir_new, file+'.csv'))


#%%
file_name = 'train.csv'
file = data_dir_new + '/' + file_name
df_tr = pd.read_csv(file, index_col=0)
df_tr['sgo'] = df_tr['sgo'].map(lambda x: np.fromstring(x.replace('[', ' ').replace(']', ' ').replace('\n', ' '), sep=' ').reshape(-1, 3, 3))
print(df_tr.keys())
print(f"{file}: {len(df_tr)}")
df_tr









#%%
