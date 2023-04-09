#%%
"""
https://www.notion.so/230402-review-zeolite-phonon-ba985314a45441749f258234ec7931c8?pvs=4#7a3c18307f494a21bfb583eb37359e75
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from glob import glob
from os.path import join as opj
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from ase import Atoms
from copy import copy
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
api_key = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from ase.build import make_supercell
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from ase import Atoms
from ase.visualize.plot import plot_atoms
from copy import copy
import imageio

# utilities
from tqdm import tqdm

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize


# colors for datasets
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
datasets = ['train', 'valid', 'test']
datasets2 = ['train', 'test']
colors = dict(zip(datasets, palette[:-1]))
colors2 = dict(zip(datasets2, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
# cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.site_symmetries import get_site_symmetries
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice



def vis_structure(struct_in, supercell=np.eye(3), title=None, rot='5x,5y,90z', savedir=None, palette=palette):
    if type(struct_in)==Structure:
        struct = Atoms(list(map(lambda x: x.symbol, struct_in.species)) , # list of symbols got from pymatgen
                positions=struct_in.cart_coords.copy(),
                cell=struct_in.lattice.matrix.copy(), pbc=True) 
    elif type(struct_in)==Atoms:
        struct=struct_in
    struct = make_supercell(struct, supercell)
    symbols = np.unique(list(struct.symbols))
    len_symbs = len(list(struct.symbols))
    z = dict(zip(symbols, range(len(symbols))))

    fig, ax = plt.subplots(figsize=(6,5))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(struct.symbols)]))]
    plot_atoms(struct, ax, radii=0.25, colors=color, rotation=(rot))

    ax.set_xlabel(r'$x_1\ (\AA)$')
    ax.set_ylabel(r'$x_2\ (\AA)$');
    fig.patch.set_facecolor('white')
    if title is None:
        ftitle = f"{struct.get_chemical_formula()}"
        fname =  struct.get_chemical_formula()
    else: 
        ftitle = f"{title} / {struct.get_chemical_formula()}"
        fname = f"{title}_{struct.get_chemical_formula()}"
    fig.suptitle(ftitle, fontsize=15)
    if savedir is not None:
        path = savedir
        if not os.path.isdir(f'{path}'):
            os.mkdir(path)
        fig.savefig(f'{path}/{fname}.png')


def movie_structs(astruct_list, name, savedir=None, supercell=np.diag([1,1,1])):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, struct_in in enumerate(astruct_list):
        vis_structure(struct_in, supercell, title=f"{{0:04d}}".format(i), savedir=savedir)
    
    with imageio.get_writer(os.path.join(savedir, f'{name}.gif'), mode='I') as writer:
        for figurename in sorted(os.listdir(savedir)):
            if figurename.endswith('png'):
                image = imageio.imread(os.path.join(savedir, figurename))
                writer.append_data(image)



    

