import numpy as np
import pickle
from os import makedirs

data_folder = "./dataset_X_Mg_NN_NNN_delta_hist_SMPL_1000"
#data_folder = "./dataset_mps_NNN_CTN_RYD"
#data_folder = "./dataset_mps_NNN"
#makedirs(data_folder,exist_ok=True)

lengths = ['7x6','7x7','8x7','8x8','9x8','9x9']
total_samples = 200
num_δs = 10 # 10
time_δ_file = num_δs # can be different from num_δs if time dependent calculation
total_samples*=num_δs
portion = 0.1
training_samples = int(total_samples*(1.-2.*portion))
validation_samples = int(total_samples*portion)
test_samples = int(total_samples*portion)
include_Xs = True
has_delta_hs = False
assert np.mod(validation_samples,num_δs)==0 and np.mod(test_samples,num_δs)==0, "The number of validation samples and\
      test samples must be an integer divisible by the number of node and edge features (num_δs)."

# during training, datasets sharing the same Jzz configuration need to be bundled up
# for the testing, this means that by exclusion, bundles are part of the same group

for L in lengths:
    datasets = ["training", "validation", "test"]
    Lx, Ly = L.split('x')
    Lx, Ly = int(Lx), int(Ly)
    for dataset in datasets:

        if dataset == 'training':
            realizations = np.arange(0, training_samples, step=num_δs, dtype=int)
        elif dataset == 'validation':
            realizations = np.arange(training_samples, training_samples+validation_samples, step=num_δs, dtype=int)
        else:
            realizations = np.arange(training_samples+validation_samples, training_samples+validation_samples+test_samples, step=num_δs, dtype=int)
        print(realizations)        
        results_dict = {}
        iterator = 0
        for realization in realizations:
            snapshot_array = []
            for δ_snapshots in range(num_δs):

                Rs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Rps = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Rps_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                if has_delta_hs:
                    delta_hs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/delta_hs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Jzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Jpzzs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Jpzzs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                hs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/hxs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                Mg = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                NN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                NNN_corrs = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                if include_Xs:
                    Mg_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/Mg_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                    NN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))
                    NNN_corrs_X = np.load(data_folder + "/{Lx}x{Ly}/num_deltas_{num_delta}/NNN_corrs_X_{i}.npy".format(Lx = Lx, Ly = Ly, num_delta=time_δ_file, i = realization+δ_snapshots+1))

                if has_delta_hs:
                    tmp_vals_dict = {"Rs": Rs, "Rps": Rps, "Jzzs": Jzzs, "Jpzzs": Jpzzs,
                                     "delta_hs": delta_hs, "hs": hs, "Mg": Mg, "NN_corrs": NN_corrs, "NNN_corrs": NNN_corrs}
                else:
                    if include_Xs:
                        tmp_vals_dict = {"Rs": Rs, "Rps": Rps, "Jzzs": Jzzs, "Jpzzs": Jpzzs,
                                         "hs": hs, "Mg": Mg, "NN_corrs": NN_corrs, "NNN_corrs": NNN_corrs,
                                         "Mg_X": Mg_X, "NN_corrs_X": NN_corrs_X, "NNN_corrs_X": NNN_corrs_X}
                    else:
                        tmp_vals_dict = {"Rs": Rs, "Rps": Rps, "Jzzs": Jzzs, "Jpzzs": Jpzzs,
                                         "hs": hs, "Mg": Mg, "NN_corrs": NN_corrs, "NNN_corrs": NNN_corrs}

                snapshot_array.append(tmp_vals_dict)

            results_dict[iterator] = snapshot_array
            iterator += 1
        with open(data_folder + "/MPS_dict_{Lx}x{Ly}_{d}.pkl".format(Lx=Lx, Ly=Ly, d=dataset), "wb") as tf:
            pickle.dump(results_dict, tf)
