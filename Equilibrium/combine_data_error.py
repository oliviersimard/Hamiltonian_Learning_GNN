import numpy as np
import sys
import h5py
from re import search
from HamL import split_string_around_substring

# This dictionary determines the possible cases of study
dict_cases = {0: "Mg + delta history",
              1: "Mg + NN + delta history",
              2: "Mg + NN + NNN + delta history",
              3: "Mg + NN + NNN + single delta"}

if __name__=='__main__':

    case_study = 2

    assert len(sys.argv) > 1, "Need to provide the path to the hdf5 file containing the metrics."
    paths_to_data = list(map(str,sys.argv))[1:]

    model_path = str(sys.argv[1])

    num_layers = int(search(r'(?<=_NLay_)\d+',model_path).group())
    hidden_channels = int(search(r'(?<=_hidC_)\d+',model_path).group())  # Hidden node feature dimension in which message passing happens
    hidden_edges = int(search(r'(?<=_hidE_)\d+',model_path).group())
    hidden_nodes = int(search(r'(?<=_hidN_)\d+',model_path).group())
    num_δs = int(search(r'(?<=_Ndeltas_)\d+',model_path).group())
    out_channels = int(search(r'(?<=_outC_)\d+',model_path).group()) # Dimension of output per each node
    delta = None
    try:
        delta = float(search(r'(?<=_delta_)\d*\.\d+',model_path).group())
    except:
        pass

    print(f"num_layers = {num_layers}, hidden_channels = {hidden_channels}, hidden_edges = {hidden_edges}, num_deltas = {num_δs}, out_channels = {out_channels}")

    if dict_cases[case_study] == "Mg + NN + NNN + single delta":
        run_name = "_trans_Ising_hidC_{:d}_hidE_{:d}_hidN_{:d}_NLay_{:d}_Ndeltas_{:d}_outC_{:d}_delta_{:.1f}".format(hidden_channels,hidden_edges,hidden_nodes,num_layers,num_δs,out_channels,delta)
    else:
        run_name = "_trans_Ising_hidC_{:d}_hidE_{:d}_hidN_{:d}_NLay_{:d}_Ndeltas_{:d}_outC_{:d}".format(hidden_channels,hidden_edges,hidden_nodes,num_layers,num_δs,out_channels)

    path_to_fig,_ = split_string_around_substring(model_path,'/models/')

    # Regex pattern to extract everything before the last integer
    pattern = r"(.*?)(\d+)(?!.*\d)"

    # Search for the pattern
    front_filename = path_to_fig
    try:
        front_filename = str(search(pattern, path_to_fig).group(1)).rstrip('_')
    except AttributeError:
        pass

    print(f"run_name = {run_name} and path_to_fig = {path_to_fig} and {front_filename}")

    data_dict = {}
    for cc,path_to_data in enumerate(paths_to_data):
        training_sizes = None
        extrapol_sizes = None
        with h5py.File(path_to_data,'r') as ff:
            training_sizes = list(ff.keys())
            for kk in training_sizes: # level of training sizes
                extrapol_sizes = list(ff.get(kk).keys())
                for ll in extrapol_sizes: # level of extrapolation sizes
                    k3 = ff.get(kk).get(ll).keys()
                    for mets in k3: # level of metrics
                        data_dict[kk+'/'+ll+'/'+mets] = data_dict.get(kk+'/'+ll+'/'+mets,[]) 
                        data_dict[kk+'/'+ll+'/'+mets].append(list(ff.get(kk).get(ll).get(mets))[0])

        
    print(f"extrapol sizes = {extrapol_sizes}")

    key = 'R2'
    R2_mean, R2_stderr = {}, {}
    for ii,kk in enumerate(training_sizes):
        for ext in extrapol_sizes:
            dat = data_dict[kk+'/'+ext+'/'+key]
            R2_mean[kk+'/'+ext+'/'+key] = np.mean(dat, axis=0)
            R2_stderr[kk+'/'+ext+'/'+key] = np.std(dat, axis=0, ddof=1) / np.sqrt(len(dat))

    key = 'MAE'
    MAE_mean, MAE_stderr = {}, {}
    for kk in training_sizes:
        for ext in extrapol_sizes:
            dat = data_dict[kk+'/'+ext+'/'+key]
            MAE_mean[kk+'/'+ext+'/'+key] = np.mean(dat, axis=0)
            MAE_stderr[kk+'/'+ext+'/'+key] = np.std(dat, axis=0, ddof=1) / np.sqrt(len(dat))

    key = 'MEDAE'
    MEDAE_mean, MEDAE_stderr = {}, {}
    for kk in training_sizes:
        for ext in extrapol_sizes:
            dat = data_dict[kk+'/'+ext+'/'+key]
            MEDAE_mean[kk+'/'+ext+'/'+key] = np.mean(dat, axis=0)
            MEDAE_stderr[kk+'/'+ext+'/'+key] = np.std(dat, axis=0, ddof=1) / np.sqrt(len(dat))
    
    key = 'STD'
    STD_mean, STD_stderr = {}, {}
    try:
        for kk in training_sizes:
            for ext in extrapol_sizes:
                dat = data_dict[kk+'/'+ext+'/'+key]
                STD_mean[kk+'/'+ext+'/'+key] = np.mean(dat, axis=0)
                STD_stderr[kk+'/'+ext+'/'+key] = np.std(dat, axis=0, ddof=1) / np.sqrt(len(dat))
    except:
        pass

    # training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+[_]\d+[x]\d+[_]\d+[x]\d+',model_path).group())
    # training_sizes = str(search(r'(?<=_sizes_)\d+[x]\d+?[_]\d+[x]\d+?[_]\d+[x]\d+?[_]\d+[x]\d+',model_path).group())
    print(f"training sizes = {training_sizes}")
    filename = front_filename + run_name + '.h5'
    with h5py.File(filename,'a') as ff:
        try:
            for kk in training_sizes:
                for L in extrapol_sizes:
                    gg = ff.require_group(kk)
                    gg = gg.require_group(L)
                    gg.require_dataset('R2_MEAN',shape=(1,),data=R2_mean[kk+'/'+L+'/R2'],dtype=float)
                    gg.require_dataset('R2_STDERR',shape=(1,),data=R2_stderr[kk+'/'+L+'/R2'],dtype=float)
                    gg.require_dataset('MAE_MEAN',shape=(1,),data=MAE_mean[kk+'/'+L+'/MAE'],dtype=float)
                    gg.require_dataset('MAE_STDERR',shape=(1,),data=MAE_stderr[kk+'/'+L+'/MAE'],dtype=float)
                    gg.require_dataset('MEDAE_MEAN',shape=(1,),data=MEDAE_mean[kk+'/'+L+'/MEDAE'],dtype=float)
                    gg.require_dataset('MEDAE_STDERR',shape=(1,),data=MEDAE_stderr[kk+'/'+L+'/MEDAE'],dtype=float)
                    if (len(STD_mean.values()) > 0 and len(STD_stderr.values()) > 0):
                        gg.require_dataset('STD_MEAN',shape=(1,),data=STD_mean[kk+'/'+L+'/STD'],dtype=float)
                        gg.require_dataset('STD_STDERR',shape=(1,),data=STD_stderr[kk+'/'+L+'/STD'],dtype=float)
        except Exception as err:
            raise Exception("Error arisen: {}".format(err))

