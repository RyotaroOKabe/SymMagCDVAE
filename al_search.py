#%%
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pickle as pkl

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

def has_kagome_lattice(structure):
    """
    Function to test the presence of a Kagome lattice structure in a crystal material.

    Args:
        structure (Structure): The crystal material as a pymatgen Structure object.

    Returns:
        bool: True if the material has a Kagome lattice structure, False otherwise.
    """
    # Check if the lattice is periodic
    if not isinstance(structure, Structure) or not structure.is_ordered:
        return False

    # Check the connectivity pattern of the lattice to identify the Kagome lattice
    for site in structure:
        neighbors = structure.get_neighbors(site, 3.0)  # Adjust the distance cutoff as needed
        if len(neighbors) != 3:
            return False

        neighbor_coords = [neighbor[0].coords for neighbor in neighbors]
        if len(set(neighbor_coords)) != 3:
            return False

    return True

def has_honeycomb_lattice(structure):
    """
    Check if a crystal material structure contains a honeycomb lattice structure.

    Args:
        structure (Structure): The crystal material structure as a pymatgen Structure object.

    Returns:
        bool: True if the structure contains a honeycomb lattice structure, False otherwise.
    """
    # Check if the lattice is hexagonal
    if not isinstance(structure, Structure) or not structure.is_ordered:
        return False
    
    analyzer = SpacegroupAnalyzer(structure)
    lattice_type = analyzer.get_lattice_type()

    if lattice_type != "hexagonal":
        return False

    # Check if the structure has two unique atomic species
    unique_species = set(site.species_string for site in structure)
    if len(unique_species) != 2:
        return False

    # Check if the angles between lattice vectors are close to 120 degrees
    angles = structure.lattice.angles
    angle_tolerance = 5.0  # Adjust the tolerance as needed
    if all(abs(angle - 120) < angle_tolerance for angle in angles):
        return True

    return False

def has_triangular_lattice(structure):
    """
    Check if a crystal material structure contains a triangular lattice structure.

    Args:
        structure (Structure): The crystal material structure as a pymatgen Structure object.

    Returns:
        bool: True if the structure contains a triangular lattice structure, False otherwise.
    """
    # Check if the lattice is hexagonal
    if not isinstance(structure, Structure) or not structure.is_ordered:
        return False
    
    analyzer = SpacegroupAnalyzer(structure)
    lattice_type = analyzer.get_lattice_type()

    if lattice_type != "hexagonal":
        return False

    # Check if the structure has one unique atomic species
    unique_species = set(site.species_string for site in structure)
    if len(unique_species) != 1:
        return False

    # Check if the angles between lattice vectors are close to 60 degrees
    angles = structure.lattice.angles
    angle_tolerance = 5.0  # Adjust the tolerance as needed
    if all(abs(angle - 60) < angle_tolerance for angle in angles):
        return True

    return False



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
download_pbe3d=True
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

# Print the dictionary of space group indices
for space_group_number, indices in space_group_indices.items():
    print(f"Space Group {space_group_number}: {indices}")



#%%
# mini test: CoSn and Silicon
cosn = mpdata['mp-20536']
silicon = mpdata['mp-149']
print('[CoSn] has Kagome lattice: ', has_kagome_lattice(cosn))
print('[Si2] has Kagome lattice: ', has_kagome_lattice(silicon))


#%%
# Material screening
def screen_materials(mpdata):
    """
    Function to screen material structures in a database and identify specific lattice types.

    Args:
        mpdata (dict): Dictionary containing material structures as pymatgen Structure objects.
                       The keys of the dictionary are the Materials Project IDs.

    Returns:
        dict: Dictionary containing the Materials Project IDs as keys and lattice type as values.
              The lattice type can be 'Kagome', 'Honeycomb', 'Triangular', or 'Unknown'.
    """
    screened_materials = {}

    for mp_id, structure in mpdata.items():
        lattice_type = 'Unknown'

        # Check for Kagome lattice structure
        if has_kagome_lattice(structure):
            lattice_type = 'Kagome'
        # Check for honeycomb lattice structure
        elif has_honeycomb_lattice(structure):
            lattice_type = 'Honeycomb'
        # Check for triangular lattice structure
        elif has_triangular_lattice(structure):
            lattice_type = 'Triangular'

        screened_materials[mp_id] = lattice_type

    return screened_materials

#%%
# Create a new dictionary with Kagome lattice materials only
mpdata_screened = screen_materials(mpdata)
kagome_materials = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened.items() if lattice_type == 'Kagome'}
honeycomb_materials = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened.items() if lattice_type == 'Honeycomb'}
triangular_materials = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened.items() if lattice_type == 'Triangular'}
print('Screening result of mpdata [Kagome]: ', kagome_materials)
print('Screening result of mpdata [Honeycomb]: ', honeycomb_materials)
print('Screening result of mpdata [Triangular]: ', triangular_materials)

mpdata_screened1 = screen_materials(mpdata_hex)
kagome_materials1 = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened1.items() if lattice_type == 'Kagome'}
honeycomb_materials1 = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened1.items() if lattice_type == 'Honeycomb'}
triangular_materials1 = {mp_id: lattice_type for mp_id, lattice_type in mpdata_screened1.items() if lattice_type == 'Triangular'}
print('Screening result of mpdata_hex [Kagome]: ', kagome_materials1)
print('Screening result of mpdata_hex [Honeycomb]: ', honeycomb_materials1)
print('Screening result of mpdata_hex [Triangular]: ', triangular_materials1)







#%%

