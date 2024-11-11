include("DMRG/src/DMRG.jl")

using .DMRG

# nx is num columns, ny is num rows
nx, ny = 4, 4
# number of samples to realize (number of distinct Jzz configurations, given that each of those imply num_δs constant transverse fields)
num_realization = 900 # the effective total number of realizations is num_realization*num_δs
num_δs = 1 # If one uses a single delta, `amp_delta' has be be set to null

# Select the way to calculate correlation functions
is_sampled = true
num_samples = 1000

# Maximum bond dimension for ground state and time evolution 
# (accuracy parameter, χ -> infinity is guaranteed to be exact. Simulation time scales as χ^{3})
χDMRG  = 100

# Initial x-field strength for the ground state simulation
R = 10.0; amp_R = 0.1 
delta = 10.0; amp_delta = 0.0 # 100

path_to_folder = "./dataset_Mg_NN_NNN_delta_one_SMPL" #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL"
if is_sampled
    path_to_folder = path_to_folder * "_SMPL_$(num_samples)"  #"./Mg_NN_NNN_delta_hist/dataset_mps_NNN_CTN_RYD_ALL/"
end

main_Mg_NN_NNN_δ(nx,ny,num_realization,num_δs,χDMRG,R,amp_R,delta,amp_delta,path_to_folder;is_sampled=is_sampled,num_samples=num_samples)
