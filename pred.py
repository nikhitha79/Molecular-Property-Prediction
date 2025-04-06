import torch
import schnetpack as spk
import pytorch_lightning as pl
import schnetpack.transform as trn
import torchmetrics
import os
import re
import numpy as np
import torch
from ase import Atoms
import pandas as pd
from schnetpack.data import ASEAtomsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

    
atomic_number_map = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
}

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0])
    atoms = []
    positions = []
    
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        positions.append([x, y, z])
    
    return atoms, positions

def symbols_to_atomic_numbers(symbols):
    return [atomic_number_map[symbol] for symbol in symbols]

def split_roman_and_integer(file_name):
    match = re.match(r'(mol+)_([0-9]+)\.xyz', file_name)
    if match:
        roman = match.group(1)
        integer = int(match.group(2))
        return roman, integer
    return None, None

def get_xyz_files(directory):
    xyz_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.xyz'):
            roman, integer = split_roman_and_integer(file_name)
            if roman is not None and integer is not None:
                xyz_files.append((roman, integer, os.path.join(directory, file_name)))
    
    xyz_files.sort(key=lambda x: (x[0], x[1]))
    return [file[2] for file in xyz_files]

def process_xyz_files(file_paths):
    dataset_entries = []
    
    for file_path in file_paths:
        atoms, positions = read_xyz(file_path)
        atomic_numbers = symbols_to_atomic_numbers(atoms)
        positions = np.array(positions)
        
        entry = {
            'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float32),
        }
        
        dataset_entries.append(entry)
    
    return dataset_entries

def process_xyz_folder_to_schnet_format(directory):
    xyz_files = get_xyz_files(directory)
    dataset_entries = process_xyz_files(xyz_files)
    
    all_atomic_numbers = []
    all_positions = []
    
    for entry in dataset_entries:
        all_atomic_numbers.append(entry['atomic_numbers'].tolist())
        all_positions.append(entry['positions'].tolist())
    
    return all_atomic_numbers, all_positions

folder_test = r"C:\Users\nikhi\Downloads\molecular-property-prediction-challenge\structures_test"
# xyz_files = get_xyz_files(folder_train)[:100]
test_data = process_xyz_folder_to_schnet_format(folder_test)
test = pd.read_csv(r"C:\Users\nikhi\Downloads\molecular-property-prediction-challenge\dipole_moments_test.csv")
atoms_list = test_data[0] 
positions_list = test_data[1] 

dpm_test = test['dipole_moment'].tolist()
atoms_objects = []
property_list = []

for atomic_numbers, positions, dpm in zip(atoms_list, positions_list, dpm_test):
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    properties = {

        'dpm': np.array([dpm]),

    }
    atoms_objects.append(atoms)
    property_list.append(properties)
    
    
import schnetpack as spk

db_path = './new_dataset70.db'
if os.path.exists(db_path):
    os.remove(db_path)

new_dataset = spk.data.ASEAtomsData.create(
    './new_dataset70.db',
    distance_unit='Ang',
    property_unit_dict={
        'dpm': 'Debye'
    }
)

new_dataset.add_systems(property_list, atoms_objects)

new_datatut = './new_dataset70'
if not os.path.exists(new_datatut):
    os.makedirs(new_datatut) 
    
    
print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

for p in new_dataset.available_properties:
    print('-', p)
print()    

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)
    
import schnetpack as spk
import schnetpack.transform as trn
import json

xyz_file_paths = get_xyz_files(folder_test)
all_mol_names = [os.path.splitext(os.path.basename(f))[0] for f in xyz_file_paths]
#added two dummy data for dummy train and val
excluded_mols = ['mol_0']
val_mols = ['mol_00']

exclude_indices = [i for i, mol in enumerate(all_mol_names) if mol in excluded_mols]
val_indices = [i for i, mol in enumerate(all_mol_names) if mol in val_mols]
all_indices = list(range(len(all_mol_names)))

remaining_indices = list(set(all_indices) - set(exclude_indices + val_indices))

train_indices = exclude_indices
test_indices = remaining_indices

split_data = {
    'train_idx': train_indices,
    'val_idx': val_indices,
    'test_idx': test_indices
}

import json

split_file_path = r"C:\Users\nikhi\dyad\new_dataset70\fixed_split_excluding_values.npz"

np.savez(
    split_file_path,
    train_idx=np.array(train_indices),
    val_idx=np.array(val_indices),
    test_idx=np.array(test_indices)
)

print(f"Saved .npz split file to {split_file_path}")

test_mol_ids = [all_mol_names[i] for i in test_indices]


newdata = spk.data.AtomsDataModule(
    './new_dataset70.db',
    batch_size=100,
    num_train=1,
    num_val=1,
    distance_unit='Ang',
    property_units={'dpm': 'Debye'},
    transforms=[
        spk.transform.ASENeighborList(cutoff=5.),
        spk.transform.CastTo32()
    ],
    num_workers=8,
    pin_memory=True,
    split_file= split_file_path,
    load_properties=['dpm']
)

newdata.prepare_data()
newdata.setup() 

best_model = torch.load(r"C:\Users\nikhi\dyad\new_dataset70\best_inference_model_fold", map_location='cpu')
testdata = newdata.test_dataset
train_data=newdata.train_dataset
pred=[]
with torch.no_grad(): 
        for batch in newdata.test_dataloader():
            result = best_model(batch)
            batch_preds = result['dpm'].cpu().detach().numpy()
            pred.extend(batch_preds)
            
df4 = pd.DataFrame({
    'ID' :test_mol_ids,
    'dipole_moment' : pred
})

df4.to_csv(r'C:\Users\nikhi\Downloads\molecular-property-prediction-challenge\predicted200.csv', index=False)
print(df4)