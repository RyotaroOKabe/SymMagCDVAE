#%%
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pickle as pkl
from utils.utils_plot import *

def get_space_group_indices(structures):
    space_group_indices = {}

    for i, structure in enumerate(structures):
        sga = SpacegroupAnalyzer(structure)
        space_group_number = sga.get_space_group_number()

        if space_group_number in space_group_indices:
            space_group_indices[space_group_number].append(i)
        else:
            space_group_indices[space_group_number] = [i]

    return space_group_indices

#%%
# load data
# (0) MP data
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
print(f'total data (MP): {len(mpids)}')
mpdata_hex = pkl.load(open('data/mp_hex.pkl', 'rb'))
mpids_hex = sorted(list(mpdata_hex.keys()))
print(f'total data (MP hex): {len(mpids_hex)}')
#%%
# (1) pbe 3d
"""
https://archive.materialscloud.org/record/2022.126
"""
import json
import bz2
import os
from pymatgen.entries.computed_entries import ComputedStructureEntry

# Directory where the downloaded files are saved
download_pbe3d=False

if download_pbe3d:
    download_directory = '/home/rokabe/data2/generative/database/pbe3d'

    # Initialize an empty list to store the entries
    entries = []

    # Iterate over the downloaded files (takes time!)
    for filename in os.listdir(download_directory):
        # Check if the file is a JSON.bz2 file
        if filename.endswith('.json.bz2'):
            print(filename)
            # Read the file and load the JSON data
            filepath = os.path.join(download_directory, filename)
            with bz2.open(filepath, 'rb') as fh:
                data = json.load(fh)

            # Extract the entries and append them to the list
            file_entries = [ComputedStructureEntry.from_dict(entry) for entry in data['entries']]
            entries.extend(file_entries)

    # Now 'entries' contains all the entries from the downloaded files
    print(f'total data (pbe3d): {len(mpids)}')
 
    structs = [entry.structure for entry in entries]
    filename1=f'/home/rokabe/data2/generative/database/pbe3d_structs{len(structs)}.pkl'
    with open(filename1, 'wb') as file:
        pkl.dump(structs, file)
    space_group_indices = get_space_group_indices(structs)
    filename2=f'/home/rokabe/data2/generative/database/pbe3d_sgns{len(structs)}.pkl'
    with open(filename2, 'wb') as file:
        pkl.dump(space_group_indices, file)

else: 
    num_structs = 2360076
    filename1=f'/home/rokabe/data2/generative/database/pbe3d_structs{num_structs}.pkl'
    filename2=f'/home/rokabe/data2/generative/database/pbe3d_sgns{num_structs}.pkl'
    with open(filename1, 'rb') as file:
        structs = pkl.load(file)
    with open(filename2, 'rb') as file:
        space_group_indices = pkl.load(file)

# Print the dictionary of space group indices
# for space_group_number, indices in space_group_indices.items():
#     print(f"Space Group {space_group_number}: {indices}")



#%%
# mini test: CoSn and Silicon
cosn = mpdata['mp-20536']
silicon = mpdata['mp-149']
# print('[CoSn] has Kagome lattice: ', has_kagome_lattice(cosn))
# print('[Si2] has Kagome lattice: ', has_kagome_lattice(silicon))


#%%
# 230724
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

def decompose_by_atomic_species(crystal_structure):
    """
    Decompose a crystal into components based on atomic species.

    Args:
        crystal_structure (pymatgen.core.Structure): The input crystal structure.

    Returns:
        list: A list of pymatgen.core.Structure objects, each containing only one atomic species.
    """
    species_map = {Element(s): [] for s in set(crystal_structure.species)}  # Create an empty list for each species

    for site in crystal_structure:
        species_map[site.specie].append(site)  # Append the site to the appropriate species list

    decomposed_structures = []

    for species, sites in species_map.items():
        # Create a new structure with sites containing the current species
        species_structure = Structure.from_sites(sites)
        decomposed_structures.append(species_structure)

    return decomposed_structures


def find_minimum_interatomic_distance(crystal_structure):
    """
    Find the minimum interatomic distance in the crystal structure.

    Args:
        crystal_structure (pymatgen.core.Structure): The input crystal structure.

    Returns:
        float: The minimum interatomic distance in Angstroms.
    """
    min_distance = float("inf")  # Initialize with a large value

    # Loop over all pairs of atoms and calculate the distance between them
    for i in range(len(crystal_structure)):
        for j in range(i + 1, len(crystal_structure)):
            distance = crystal_structure.get_distance(i, j)
            if distance < min_distance:
                min_distance = distance

    return min_distance

# Example usage:
# Assuming you have a pymatgen.core.Structure object named 'crystal'
# minimum_distance = find_minimum_interatomic_distance(crystal)

# print(f"Minimum interatomic distance: {minimum_distance:.3f} Å")

def search_archimedean_lattice_layer_structure(crystal_structure, show_struct=False):
    """
    Search for Archimedean lattice layer structures in a crystal considering periodic boundary condition.

    Args:
        crystal_structure (pymatgen.core.Structure): The input crystal structure.

    Returns:
        list: A list of tuples containing the atomic species and the properties of each layer structure found.
              Each tuple has the following format: (species, min_interatomic_distance, neighbor_counts, angles, same_plane)
    """
    if not isinstance(crystal_structure, Structure) or not crystal_structure.is_ordered:
        raise ValueError("The input structure must be a valid ordered Structure object.")

    # Check if the lattice is hexagonal
    analyzer = SpacegroupAnalyzer(crystal_structure)
    lattice_type = analyzer.get_lattice_type()

    if lattice_type != "hexagonal":
        raise ValueError("The lattice type must be hexagonal to search for Archimedean lattice layer structures.")

    # Decompose the structure by the atomic species
    decomposed_structures = decompose_by_atomic_species(crystal_structure)

    num_neighbors = []
    results = []

    # Create a NearNeighbors object using the crystal structure
    # nn = NearNeighbors()

    for i, structure in enumerate(decomposed_structures):
        # print(f"Structure {idx + 1}: {structure.formula}")
        if show_struct:
            vis_structure(structure, supercell=np.eye(3), title=None, rot='5x,5y,90z', savedir=None, palette=palette)

        dmin = find_minimum_interatomic_distance(structure)
        for site in structure:
            print('site: ', site.frac_coords)
            neighbors =structure.get_neighbors(site, dmin+1e-2)  # Adjust the distance cutoff as needed
            print('neighbor: ', [neighbor.frac_coords for neighbor in neighbors])
            num_neighbors.append(len(neighbors))
    if 4 in num_neighbors:
        results.append('kagome')
    if 3 in num_neighbors:
        results.append('honeycomb')
    if 0 in num_neighbors or 6 in num_neighbors:
        results.append('triangular')

    return results

    # for species_structure in decomposed_structures:
    #     species = species_structure.species[0].name

    #     # Check the minimum distance of interatomic distances
    #     min_distance = find_minimum_interatomic_distance(species_structure)

    #     # Check the neighbors for each atom using the min distance
    #     nn_info_list = nn.get_all_nn_info(species_structure, min_distance)

    #     # Check the counts of the nearest neighbor atoms
    #     neighbor_counts = [len(nn_info) for nn_info in nn_info_list]

    #     # Check the angle lists (not implemented in this example)

    #     # Check if all neighbors are on the same plane (not implemented in this example)

    #     # Append the results to the final list
    #     result.append((species, min_distance, neighbor_counts, None, None))

# Example usage:
# Assuming you have a pymatgen.core.Structure object named 'crystal'
archimedean_results = search_archimedean_lattice_layer_structure(structs[94794])
print(archimedean_results)
# for species, min_distance, neighbor_counts, angles, same_plane in archimedean_results:
#     print(f"Species: {species}")
#     print(f"Minimum interatomic distance: {min_distance:.2f} Å")
#     print(f"Neighbor counts: {neighbor_counts}")
#     print("-" * 40)

#%%
# screen the material of specific space group index
sg_idx = 191
idx_targets = space_group_indices[sg_idx]

#%%
kagomes, honeys, triangs = [], [], []
for i, idx in enumerate(idx_targets):
    struct = structs[idx]
    result = search_archimedean_lattice_layer_structure(struct)
    if 'kagome' in result:
        kagomes.append(idx)
    if 'honeycomb' in result:
        honeys.append(idx)
    if 'triangular' in result:
        triangs.append(idx)

print('Kagome materials')
print(kagomes)
print('Honeycomb materials')
print(honeys)
print('Triangular materials')
print(triangs)

print('[kagomes, honeycombs, triangulars: ', [len(kagomes), len(honeys), len(triangs)])

#%%

