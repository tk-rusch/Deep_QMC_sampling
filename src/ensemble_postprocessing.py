import numpy as np
import os
import job_id_maps

def best_network(set_size,sampling_method,exp_type,dim):
    cwd = os.getcwd()
    file_path = cwd + '/../data/' + exp_type + '/results/' + sampling_method + \
                '/ensemble/results_dim_' + str(dim) + '_N_' + str(set_size) + '.txt'
    data = np.loadtxt(file_path)
    arg = np.argmin(data[:,5])
    learn_rate, reg_param, width, depth = data[arg,:4]
    model = job_id_maps.Network_architecture(learn_rate, reg_param, width, depth)

    return model