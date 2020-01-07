from job_id_maps import job_id_to_parameters
from torch_methods import train_NN
from sys import argv
import numpy as np

def training_set_sizes(exp_type):
    if(exp_type=="sum_sines"):
        return np.logspace(7,13,7,base=2).tolist()
    elif(exp_type=='airfoil'):
        return np.logspace(2,8,7,base=2).tolist()
    elif(exp_type=='projectile'):
        return np.logspace(4,10,7,base=2).tolist()
    elif(exp_type=='BSPDE'):
        return np.logspace(5,11,7,base=2).tolist()

if __name__ == '__main__':
    script, id, exp_type, train_type, max_epochs = argv

    training_set_sizes = training_set_sizes(exp_type)
    params = job_id_to_parameters(int(id)-1,training_set_sizes,train_type)
    sampling_method = params[-1]
    params = np.array(params[:-1])
    train_NN(*params,train_type,sampling_method,int(max_epochs))
