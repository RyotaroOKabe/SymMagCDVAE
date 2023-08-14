#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from ase import Atom, Atoms
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

data_dir = '/home/rokabe/data2/generative/symcdvae/data/mp_20'
file_name = 'train.csv'
file = data_dir + '/' + file_name

#%%
df = pd.read_csv(file)
print(df.keys())
print(f"{file}: {len(df)}")

#%%

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

#%%
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
#%%
idx = 100
df_line = df.iloc[idx]
mpid = df_line['material_id']
# formula = df_line['pretty_formula']
crystal_str = df_line['cif']
pstruct = str2pymatgen(crystal_str)
astruct = pymatgen2ase(pstruct)
build_crystal(crystal_str)

#%%
