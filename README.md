# Hamiltonian_Learning_GNN
Repository containing basic code to perform GNN training and testing over TFIM datasets.


## Installation
The installation is very basic. The source files to train the PNA GNN are contained in the folder `./Equilibrium/HamL`. In the text file `./env_requirements.txt`, the Python packages are listed out and can be installed recursively using the following commands `python3 -m pip install --upgrade pip` and `python3 -m pip install -r ./env_requirements.txt`. The Python version used in this project is 3.9.11.

## Basic usage 
A basic example of the code usage is given in this repository. This example aims to reproduce Fig. (4) in the paper which concerned all training scenarios.

But first, the parameters contained in `params.json` are described for clarity, in order.

1. case_study: an integer defining the training scenario [0,6].
2. num_layers: number of layers in the deep neural network.
3. hidden_channels: number of hidden channels in the message-passing mechanism.
4. out_channels: node feature space output dimensionality of the GNN.
5. hidden_edges: enhancement factor of the edge embedding space.
6. hidden_nodes: enhancement factor of the node embedding space.
7. learning_rate: initial learning rate value.
8. warmup_height: learning rate warm-up height w.r.t initial learning rate value that spans over `warmup_ratio`.
9. warmup_ratio: proportion of the learning rate profile featuring a warm-up of height `warmup_height`.
10. minimum_lr_ratio: determines the prefactor of `learning_rate` to be reached at end of training.
11. weight_decay: weight decay parameter for the AdamW optimizer.
12. num_epochs: number of epochs characterizing the training.
13. batch_sizes: batch sizes for training, validation and testing, respectively.
14. is_early_stopping: use early-stopping or not.
15. Ls: array containing the cluster sizes used during training.
16. total_samples: total number of samples to be used during training. One sample is defined by its array of Rabi frequencies `num_δs`.
17. portions: proportions of the total number of samples dedicated to the training dataset (1st slot), validation dataset (2nd slot) and test dataset (3rd slot).
18. num_δs: number of Rabi frequencies constituting one sample of the dataset.
19. load_former_models: load fromerly trained model with corresponding set of GNN hyperparameters.
20. incl_scnd: whether to consider NNN correlators in the loss function during the training.
21. trgt_diff: whether the targets are expressed in relative NN distances or in absolute distances. Should be kept enabled.
22. delta: specific value of the Rabi frequency if no array is used. `num_δs` needs to be fixed to 1.
23. save_to_one_shots: save the metrics evaluated from testing the trained model on testing sets.
24. save_variances: save the variances of the NN correlators or not.
25. folder_models: path to folder that will contain the trained model.
26. folder_datasets: path to folder containing datasets from with to train/test.

### Procedure

The `params.json` file contains the set of values used throughout the work. Simply run the command `python3 GNN_training.py` to perform training, making sure you have untared the file containing the datasets in `./Equilibrium/Datasets/dataset_X_Mg_NN_NNN_delta_hist/`. Upon completion, this will produce a `*.pt` file in `./Equilibrium/TrainedModels/Mg_NN_NNN_delta_hist/models/dataset_X_Mg_NN_NNN_delta_hist/`. This is the trained model utilized for testing later on. After having performed training, simply change the parameter `total_samples` to 200. The file `GNN_testing.py` is set to test all the sizes considered in this work by default, so no need to change the parameter `Ls`. Just pass the path to the trained model _pathToTrainedModel_ via the command line ```python3 GNN_testing.py pathToTrainedModel```.

