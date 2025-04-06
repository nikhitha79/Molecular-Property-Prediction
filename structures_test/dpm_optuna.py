import optuna
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

def objective(trial):
    
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

    folder_train = 'structure_train_filtered'
    # folder_test = 'structure_test_filtered'

    train_data = process_xyz_folder_to_schnet_format(folder_train)
    # test_data = process_xyz_folder_to_schnet_format(folder_test)

    atoms_list = train_data[0] 
    positions_list = train_data[1] 

    train = pd.read_csv("dipole_train_filtered.csv")
    # test = pd.read_csv("dipole_test_filtered.csv")



    dpm_train = train['dipole_moment'].tolist()



    atoms_objects = []
    property_list = []

    for atomic_numbers, positions, dpm in zip(atoms_list, positions_list, dpm_train):
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
            'dpm': 'kcal/mol'
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

    
    total_train = len(train)
   

    total_data = len(train) 
    total_val = 3000
    total_test = 1500
    total_train = total_data - total_val - total_test

    indices = np.arange(total_data)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:total_train].tolist()
    val_indices = indices[total_train:total_train + total_val].tolist()
    test_indices = indices[total_train + total_val:].tolist()

    split_data = {
        "train_idx": train_indices,
        "val_idx": val_indices,
        "test_idx": test_indices
    }

    with open("split_file2.json", 'w') as f:
        json.dump(split_data, f)

    print("✅ Split file saved with:")
    print(f"- Train: {len(train_indices)}")
    print(f"- Val:   {len(val_indices)}")
    print(f"- Test:  {len(test_indices)}")



    split_file_path = 'split_file2.json' 
    with open(split_file_path, 'r') as split_file:
        folds = json.load(split_file)

    print(f"Loaded split file from {split_file_path}")
    print(f"Split file saved to {split_file_path}")

    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    assert train_set.isdisjoint(val_set), "Training and validation sets overlap!"
    assert train_set.isdisjoint(test_set), "Training and test sets overlap!"
    assert val_set.isdisjoint(test_set), "Validation and test sets overlap!"

    print("No overlaps found in train, validation, and test sets.")

    bs=trial.suggest_categorical('bs', [300, 400, 500, 600, 700, 800, 900, 1000])
    newdata= spk.data.AtomsDataModule(
    './new_dataset70.db',
    batch_size=bs,
    num_train=len(train_indices),
    num_val=len(val_indices),
    distance_unit='Ang',
    property_units={'dpm': 'kcal/mol'},
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets('dpm', remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=2,
    split_file=os.path.join(new_datatut, "split_file2.json"),
    pin_memory=True,
    load_properties=['dpm']
    
)

    newdata.prepare_data()
    newdata.setup() 

    lr = trial.suggest_categorical('lr', [ 1e-4, 5e-4, 1e-5, 5e-5])  # Learning rate range
    patience = trial.suggest_int('patience', 1, 15)  # Patience for early stopping
    n_rbf = trial.suggest_int('n_rbf', 15,20)  # Number of radial basis functions
    n_interactions = trial.suggest_categorical('n_interactions',[ 3, 4, 5, 6, 7])  # Interaction layers
    n_atom_basis = trial.suggest_categorical('n_atom_basis', [30, 64, 128])  # Atom basis options
    cutoff = trial.suggest_int('cutoff', 3, 10)  # Cutoff range
    pat2 = trial.suggest_int('pat2', 1, 20)  # Patience for learning rate scheduler
    
    pairwise_distance=spk.atomistic.PairwiseDistances()
    radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    schnet=spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=n_interactions,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
     activation=F.silu
)

    pred_correction = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='dpm')
    nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_correction],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets('dpm', add_mean=True, add_atomrefs=False)
    ]
)

    output_corr=spk.task.ModelOutput(
    name='dpm',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
            'MAE': torchmetrics.MeanAbsoluteError(),
             'RMSE': torchmetrics.MeanSquaredError()
    }
    
)
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_corr],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': lr}, 
        scheduler_cls=spk.train.ReduceLROnPlateau,
        scheduler_monitor='val_loss',
        scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': pat2, 'threshold_mode': 'rel', 'cooldown': 5},

    )

    logger=pl.loggers.TensorBoardLogger(save_dir=new_datatut)

    trainer = pl.Trainer(
        callbacks=[
            spk.train.ModelCheckpoint(
                model_path=os.path.join(new_datatut, 'best_inference_model'),
                save_top_k=1,
                monitor='val_loss'
            ),
            pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience, min_delta=0),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            spk.train.ExponentialMovingAverage(decay=0.995)
        ],
        logger=logger,
        default_root_dir=new_datatut,
        max_epochs=100,
        accelerator='gpu',
        devices='auto',
        accumulate_grad_batches= 1,                                                                                                                                                                          
        val_check_interval= 1.0 ,                                                                                                                                                                            
        check_val_every_n_epoch= 1 ,                                                                                                                                                                         
        num_sanity_val_steps= 0,                                                                                                                                                                             
        fast_dev_run= False,
        enable_checkpointing=True,
        overfit_batches= 0 ,                                                                                                                                                                                 
        limit_train_batches= 1.0  ,                                                                                                                                                                          
        limit_val_batches= 1.0  ,                                                                                                                                                                            
        limit_test_batches= 1.0,
        log_every_n_steps=5
    )

    # Train the model
    trainer.fit(task, datamodule=newdata)

    return trainer.callback_metrics['val_loss'].item()

study = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(), storage='sqlite:///db.sqlite3')  # 'minimize' because we want to minimize the validation loss
study.optimize(objective, n_trials=10) 
print(f"Best parameters: {study.best_params}")
print(f"Best trial: {study.best_trial}")
optuna.logging.set_verbosity(optuna.logging.DEBUG)

