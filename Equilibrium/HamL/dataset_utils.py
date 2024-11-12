"""
Filename: file_utils.py
Description: Module containing functions that prepare the graph datasets fed into the PNA GNN (training_loader,validation_loader,testing_loader). The datasets are prepared according 
to the study case chosen.
Author: Olivier Simard
Date: 2024-08-11
License: MIT License
"""
import numpy as np
import torch
#dataloader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.linalg import norm
import pickle
from typing import List, Dict, Tuple

__all__=["split_string_around_substring","tup_edges","load_datasets_mag_NN_NNN_δ","load_datasets_mag_NN_δ","load_datasets_mag_δ","load_datasets_mag_NN_NNN_one_δ"]

def split_string_around_substring(s: str, substring: str) -> Tuple[str,str]:
    # Find the index of the first occurrence of the substring
    index = s.find(substring)
    
    if index != -1:
        # Split the string at the found index
        part1 = s[:index]
        part2 = s[index + len(substring):]
        return part1, part2
    else:
        # If no match is found, return the original string and an empty string
        return s, ""
    
def tup_edges(Lx: int,Ly: int,*,is_NNN: bool=False,is_XZ: bool=False) -> Tuple[int,int]:
    a1, a2 = np.array([1.,0.]), np.array([0.,1.]) # rectangular lattice
    coordinates = []
    for y in range(Ly):
        for x in range(Lx):
            a_tmp = x*a1+y*a2
            coordinates.append(a_tmp)
    # create graph
    edges = []
    for idx_i in range(len(coordinates)):
        ii = coordinates[idx_i]
        for idx_j in range(len(coordinates)):
            jj = coordinates[idx_j]
            dist = norm(jj-ii)
            edges.append((idx_i,idx_j,dist))
    paths_by_length = {}
    for ee in edges:
        path_length = np.round(ee[2],8)
        if path_length not in paths_by_length:
            paths_by_length[path_length] = []
        paths_by_length[path_length].append((ee[0], ee[1]))
    keys = sorted(list(paths_by_length.keys()))

    edges = None
    if not is_NNN:
        if not is_XZ:
            edges = paths_by_length[keys[1]]
        else:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[1]] # Z, X NNs
    else:
        if not is_XZ:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[2]]
        else:
            edges = paths_by_length[keys[1]] + paths_by_length[keys[2]] + paths_by_length[keys[1]] + paths_by_length[keys[2]] # Z, X NNs+NNNs
    
    return edges

def load_datasets_mag_NN_NNN_δ(num_realizations: List[int], Ls: List[str], num_δs: int, *, incl_scnd: bool = False, trgt_diff: bool = True, meas_basis: str = "Z", case_study: int = 2,
                               data_folder: str = "./dataset_mps_NNN", datasets: List[str] = ["training", "validation", "test"], batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64):
    """
    Prepares the batches of data for training. It loads the training datasets and properly set up the `Dataloader` objects, 
    which consist in lists of `Data` objects.

        Args:
            num_realizations: total number of `Data` objects that will be split into training, validation and test sets.
            Ls: array of square lattice lengths describing the Rydberg arrays.
            num_δs: length of the array of Rabi frequencies pacted into one graph dataset sample.
            *
            incl_scnd: whether to consider NNN correlators in the loss function during the training.
            trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances.
            meas_basis: basis in which the observables (training input) are measured.
            case_study: training scenario under consideration ranged between 0,...,6, inclusively.
            data_folder: path string to the dataset folder.
            datasets: datasets to prepare.
            batch_sizes: array of batch sizes for the training, validation and test sets.
            dtype: torch data type.

        Returns:
            A three-tuple of training, validation and test sets.
    """
    train_loader, validation_loader, test_loader = None, None, None
    for it in range(len(datasets)):
        data_list = []
        
        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            edges = tup_edges(Lx,Ly,is_NNN=True,is_XZ=(True if meas_basis=="ZX" else False))
            num_edges = len(edges)
            num_nodes = Lx*Ly
            # Load datasets
            with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=datasets[it]), "rb") as tf:
                results_dict = pickle.load(tf)
            
            edge_index = torch.as_tensor(edges, dtype=torch.long).T

            Rs_nom = np.load(data_folder + "/{Lx}x{Ly}/Rs_nom.npy".format(Lx = Lx, Ly = Ly))
            Rps_nom = np.load(data_folder + "/{Lx}x{Ly}/Rps_nom.npy".format(Lx = Lx, Ly = Ly))
            for realization in range(num_realizations[it]):
                
                edge_attr = torch.empty((num_edges,num_δs), dtype=dtype)
                x_labels = torch.empty((num_nodes,num_δs), dtype=dtype)
                x = torch.empty((num_nodes,num_δs), dtype=dtype)
                if trgt_diff:
                    ΔRs = np.array(results_dict[realization][0]['Rs']) - Rs_nom
                    ΔRps = np.array(results_dict[realization][0]['Rps']) - Rps_nom
                else:
                    ΔRs = np.array(results_dict[realization][0]['Rs'])
                    ΔRps = np.array(results_dict[realization][0]['Rps'])
                
                if not incl_scnd: # decide to include the second-nearest neighbor targets in the training or not
                    if not meas_basis=="ZX":
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                    else:
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,np.zeros_like(ΔRps),np.zeros_like(ΔRs),np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                else:
                    if not meas_basis=="ZX":
                        edge_labels = torch.as_tensor(
                                np.concatenate(
                                    (ΔRs,ΔRps), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                    else:
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,ΔRps,np.zeros_like(ΔRs),np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                
                for δ_snapshot in range(num_δs):

                    x_labels_δ = torch.as_tensor(
                            results_dict[realization][δ_snapshot]['hs'], dtype=dtype
                            )
                    x_labels[:,δ_snapshot] = x_labels_δ
                    
                    if meas_basis=="Z":
                        if case_study==2 or case_study==3:
                            x_δ = torch.as_tensor(
                                np.array(results_dict[realization][δ_snapshot]['Mg']), 
                                dtype=dtype
                            ) # time-independent
                        elif case_study==4:
                            x_δ = torch.as_tensor(
                                np.ones_like(results_dict[realization][δ_snapshot]['Mg']), 
                                dtype=dtype
                            ) # time-independent

                        if case_study==2:
                            edge_attr_δ =  torch.as_tensor(
                                np.concatenate(
                                    (np.array(results_dict[realization][δ_snapshot]['NN_corrs']),np.array(results_dict[realization][δ_snapshot]['NNN_corrs'])), axis=0
                                ), 
                                dtype=dtype
                            )
                        elif case_study==3 or case_study==4:
                            edge_attr_δ =  torch.as_tensor(
                                np.concatenate(
                                    (np.array(results_dict[realization][δ_snapshot]['NN_corrs']),np.ones_like(results_dict[realization][δ_snapshot]['NNN_corrs'])), axis=0
                                ), 
                                dtype=dtype
                            )
                    elif meas_basis=="ZX":
                        
                        x_δ = torch.as_tensor(
                            np.array(results_dict[realization][δ_snapshot]['Mg']), 
                            dtype=dtype
                        ) # time-independent
                        
                        edge_attr_δ =  torch.as_tensor(
                            np.concatenate(
                                (
                                    np.array(results_dict[realization][δ_snapshot]['NN_corrs']),
                                    np.array(results_dict[realization][δ_snapshot]['NNN_corrs']),
                                    np.array(results_dict[realization][δ_snapshot]['NN_corrs_X']),
                                    np.array(results_dict[realization][δ_snapshot]['NNN_corrs_X'])
                                ), axis=0
                            ), 
                            dtype=dtype
                        )

                    x[:,δ_snapshot] = x_δ
                    edge_attr[:,δ_snapshot] = edge_attr_δ

                graph = Data(
                    x = x, # node features
                    edge_index = edge_index,
                    edge_attr = edge_attr, # edge features
                    x_labels = x_labels,
                    edge_labels = edge_labels
                )

                data_list.append(graph)

                if realization==0:
                    print(f"For size {Lx}x{Ly} and dataset {datasets[it]}, x.shape = {x.shape}, edge_index.shape = {edge_index.shape}, edge_attr = {edge_attr.shape}")

        if datasets[it] == "training":
            train_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "validation":
            validation_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "test":
            test_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        else:
            raise ValueError("Nonexisting dataset keyword inputted!")
    
    return train_loader, validation_loader, test_loader


def load_datasets_mag_NN_δ(num_realizations: List[int], Ls: List[str], num_δs: int, *, trgt_diff: bool = True, data_folder: str = "./dataset_mps_NNN", 
                           datasets: List[str] = ["training", "validation", "test"], batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64):
    """
    Prepares the batches of data for training when NNN correlators/edges are left out. It loads the training datasets and properly set up the `Dataloader` objects, 
    which consist in lists of `Data` objects.

        Args:
            num_realizations: total number of `Data` objects that will be split into training, validation and test sets.
            Ls: array of square lattice lengths describing the Rydberg arrays.
            num_δs: length of the array of Rabi frequencies pacted into one graph dataset sample.
            *
            trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances.
            data_folder: path string to the dataset folder.
            datasets: datasets to prepare.
            batch_sizes: array of batch sizes for the training, validation and test sets.
            dtype: torch data type.

        Returns:
            A three-tuple of training, validation and test sets.
    """
    train_loader, validation_loader, test_loader = None, None, None
    for it in range(len(datasets)):
        data_list = []
        
        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            edges = tup_edges(Lx,Ly,is_NNN=False)
            num_edges = len(edges)
            num_nodes = Lx*Ly
            # Load datasets
            with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=datasets[it]), "rb") as tf:
                results_dict = pickle.load(tf)
            
            edge_index = torch.as_tensor(edges, dtype=torch.long).T

            Rs_nom = np.load(data_folder + "/{Lx}x{Ly}/Rs_nom.npy".format(Lx = Lx, Ly = Ly))
            for realization in range(num_realizations[it]):
                
                edge_attr = torch.empty((num_edges,num_δs), dtype=dtype)
                x_labels = torch.empty((num_nodes,num_δs), dtype=dtype)
                x = torch.empty((num_nodes,num_δs), dtype=dtype)
                if trgt_diff:
                    ΔRs = np.array(results_dict[realization][0]['Rs']) - Rs_nom
                else:
                    ΔRs = np.array(results_dict[realization][0]['Rs'])
                
                edge_labels = torch.as_tensor(ΔRs,dtype=dtype
                        ).reshape(-1,1) # Jzzs remain constant
                
                for δ_snapshot in range(num_δs):

                    x_labels_δ = torch.as_tensor(
                            results_dict[realization][δ_snapshot]['hs'], dtype=dtype
                            )
                    x_labels[:,δ_snapshot] = x_labels_δ
                    
                    x_δ = torch.as_tensor(
                        np.array(results_dict[realization][δ_snapshot]['Mg']), 
                        dtype=dtype
                    ) # time-independent
                    x[:,δ_snapshot] = x_δ

                    edge_attr_δ =  torch.as_tensor(np.array(results_dict[realization][δ_snapshot]['NN_corrs']), 
                        dtype=dtype
                    )
                    edge_attr[:,δ_snapshot] = edge_attr_δ
                # print(f"edge_attr = {edge_attr.shape} and edge_labels = {edge_labels.shape} and edge_index = {edge_index.shape}")
                graph = Data(
                    x = x, # node features
                    edge_index = edge_index,
                    edge_attr = edge_attr, # edge features
                    x_labels = x_labels,
                    edge_labels = edge_labels
                )

                data_list.append(graph)

                if realization==0:
                    print(f"For size {Lx}x{Ly} and dataset {datasets[it]}, x.shape = {x.shape}, edge_index.shape = {edge_index.shape}, edge_attr = {edge_attr.shape}")

        if datasets[it] == "training":
            train_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "validation":
            validation_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "test":
            test_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        else:
            raise ValueError("Nonexisting dataset keyword inputted!")
    
    return train_loader, validation_loader, test_loader


def load_datasets_mag_δ(num_realizations: List[int], Ls: List[str], num_δs: int, *, trgt_diff: bool = True, data_folder: str = "./dataset_mps_NNN", 
                           datasets: List[str] = ["training", "validation", "test"], batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64):
    """
    Prepares the batches of data for training when only magnetizations are considered. It loads the training datasets and properly set up the `Dataloader` objects, 
    which consist in lists of `Data` objects.

        Args:
            num_realizations: total number of `Data` objects that will be split into training, validation and test sets.
            Ls: array of square lattice lengths describing the Rydberg arrays.
            num_δs: length of the array of Rabi frequencies pacted into one graph dataset sample.
            *
            trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances.
            data_folder: path string to the dataset folder.
            datasets: datasets to prepare.
            batch_sizes: array of batch sizes for the training, validation and test sets.
            dtype: torch data type.

        Returns:
            A three-tuple of training, validation and test sets.
    """
    train_loader, validation_loader, test_loader = None, None, None
    for it in range(len(datasets)):
        data_list = []
        
        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            edges = tup_edges(Lx,Ly,is_NNN=False)
            num_nodes = Lx*Ly
            # Load datasets
            with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=datasets[it]), "rb") as tf:
                results_dict = pickle.load(tf)
            
            edge_index = torch.as_tensor(edges, dtype=torch.long).T

            Rs_nom = np.load(data_folder + "/{Lx}x{Ly}/Rs_nom.npy".format(Lx = Lx, Ly = Ly))
            for realization in range(num_realizations[it]):
                
                x_labels = torch.empty((num_nodes,num_δs), dtype=dtype)
                x = torch.empty((num_nodes,num_δs), dtype=dtype)
                if trgt_diff:
                    ΔRs = np.array(results_dict[realization][0]['Rs']) - Rs_nom
                else:
                    ΔRs = np.array(results_dict[realization][0]['Rs'])
                
                edge_labels = torch.as_tensor(ΔRs,dtype=dtype
                        ).reshape(-1,1) # Jzzs remain constant
                
                for δ_snapshot in range(num_δs):

                    x_labels_δ = torch.as_tensor(
                            results_dict[realization][δ_snapshot]['hs'], dtype=dtype
                            )
                    x_labels[:,δ_snapshot] = x_labels_δ
                    
                    x_δ = torch.as_tensor(
                        np.array(results_dict[realization][δ_snapshot]['Mg']), 
                        dtype=dtype
                    ) # time-independent
                    x[:,δ_snapshot] = x_δ
                
                graph = Data(
                    x = x, # node features
                    edge_index = edge_index,
                    edge_attr = torch.as_tensor(np.repeat(Rs_nom.reshape(-1,1),num_δs,axis=-1)),
                    x_labels = x_labels,
                    edge_labels = edge_labels
                )

                data_list.append(graph)

                if realization==0:
                    print(f"For size {Lx}x{Ly} and dataset {datasets[it]}, x.shape = {x.shape}, edge_index.shape = {edge_index.shape}")

        if datasets[it] == "training":
            train_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "validation":
            validation_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "test":
            test_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        else:
            raise ValueError("Nonexisting dataset keyword inputted!")
    
    return train_loader, validation_loader, test_loader


def load_datasets_mag_NN_NNN_one_δ(num_realizations: List[int], Ls: List[str], delta: float, *, incl_scnd: bool = False, trgt_diff: bool = True, meas_basis: str = "Z", 
                                   data_folder: str = "./dataset_mps_NNN_OutScope", datasets: List[str] = ["training", "validation", "test"], batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64):
    """
    Prepares the batches of data for training when single Rabi frequencies are considered. It loads the training datasets and properly set up the `Dataloader` objects, 
    which consist in lists of `Data` objects.

        Args:
            num_realizations: total number of `Data` objects that will be split into training, validation and test sets.
            Ls: array of square lattice lengths describing the Rydberg arrays.
            num_δs: length of the array of Rabi frequencies pacted into one graph dataset sample.
            *
            incl_scnd: whether to consider NNN correlators in the loss function during the training.
            trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances.
            meas_basis: basis in which the observables (training input) are measured.
            data_folder: path string to the dataset folder.
            datasets: datasets to prepare.
            batch_sizes: array of batch sizes for the training, validation and test sets.
            dtype: torch data type.

        Returns:
            A three-tuple of training, validation and test sets.
    """
    
    train_loader, validation_loader, test_loader = None, None, None
    for it in range(len(datasets)):
        data_list = []
        
        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            edges = tup_edges(Lx,Ly,is_NNN=True)
            # Load datasets
            with open(data_folder + "/MPS_dict_{Lx}x{Ly}_delta_{delta:.1f}_{d}.pkl".format(Lx=Lx, Ly=Ly, delta=delta, d=datasets[it]), "rb") as tf:
                results_dict = pickle.load(tf)
            
            edge_index = torch.as_tensor(edges, dtype=torch.long).T

            Rs_nom = np.load(data_folder + "/{Lx}x{Ly}/Rs_nom.npy".format(Lx = Lx, Ly = Ly))
            Rps_nom = np.load(data_folder + "/{Lx}x{Ly}/Rps_nom.npy".format(Lx = Lx, Ly = Ly))
            try:
                for realization in range(num_realizations[it]):

                    if trgt_diff:
                        ΔRs = np.array(results_dict[realization]['Rs']) - Rs_nom
                        ΔRps = np.array(results_dict[realization]['Rps']) - Rps_nom
                    else:
                        ΔRs = np.array(results_dict[realization]['Rs'])
                        ΔRps = np.array(results_dict[realization]['Rps'])
                
                    if not incl_scnd: # decide to include the second-nearest neighbor targets in the training or not
                        edge_labels = torch.as_tensor( # the NNN correlation functions are not associated to any couplings of the Hamiltonian, hence the zeros
                                np.concatenate(
                                    (ΔRs,np.zeros_like(ΔRps)), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant
                    else:
                        edge_labels = torch.as_tensor(
                                np.concatenate(
                                    (ΔRs,ΔRps), axis=0 # np.array(results_dict[realization][0]['delta_Rps'])
                                ), 
                                dtype=dtype
                                ).reshape(-1,1) # Jzzs remain constant

                    x_labels = torch.as_tensor(
                            results_dict[realization]['hs'], dtype=dtype
                            ).reshape(-1,1)
                    
                    if meas_basis=="Z":
                        x = torch.as_tensor(
                            np.array(results_dict[realization]['Mg']), 
                            dtype=dtype
                        ).reshape(-1,1) # time-independent

                        edge_attr =  torch.as_tensor(
                            np.concatenate(
                                (np.array(results_dict[realization]['NN_corrs']),np.array(results_dict[realization]['NNN_corrs'])), axis=0
                            ), 
                            dtype=dtype
                        ).reshape(-1,1)
                    elif meas_basis=="X":
                        x = torch.as_tensor(
                            np.array(results_dict[realization]['Mg_X']), 
                            dtype=dtype
                        ).reshape(-1,1) # time-independent

                        edge_attr =  torch.as_tensor(
                            np.concatenate(
                                (np.array(results_dict[realization]['NN_corrs_X']),np.array(results_dict[realization]['NNN_corrs_X'])), axis=0
                            ), 
                            dtype=dtype
                        ).reshape(-1,1)

                    graph = Data(
                        x = x, # node features
                        edge_index = edge_index,
                        edge_attr = edge_attr, # edge features
                        x_labels = x_labels,
                        edge_labels = edge_labels
                    )

                    data_list.append(graph)

                    if realization==0:
                        print(f"For size {L}x{L}, x.shape = {x.shape}, edge_index.shape = {edge_index.shape}, edge_attr = {edge_attr.shape}")
            except KeyError as err:
                raise KeyError("Realization number out of scope! Adapt the range of realizations: {}".format(err))
                
        if datasets[it] == "training":
            train_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "validation":
            validation_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        elif datasets[it] == "test":
            test_loader = DataLoader(data_list, batch_size = batch_sizes[it])
        else:
            raise ValueError("Nonexisting dataset keyword inputted!")
    
    return train_loader, validation_loader, test_loader