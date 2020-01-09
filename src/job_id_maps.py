import os
import ensemble_postprocessing

class Network_architecture():
    def __init__(self, learning_rate, regression_param, width, depth):
        self.learning_rate = learning_rate
        self.regression_param = regression_param
        self.width = int(width)
        self.depth = int(depth)


def erase_data(train_type,sampling_method,set_sizes,exp_type,dim):
    cwd = os.getcwd()
    if(train_type == 'ensemble'):
        for N in set_sizes:
            file_path = cwd + '/../data/' + exp_type + '/results/' + sampling_method + \
                        '/ensemble/results_dim_' + str(dim) + '_N_' + str(N) + '.txt'
            open(file_path,'w').close()

    else:
        for N in set_sizes:
            file_path = cwd + '/../data/' + exp_type + '/results/' + sampling_method + \
                        '/retrain/results_dim_' + str(dim) + '_N_' + str(N) + '.txt'
            open(file_path,'w').close()

def input_space_dimensions(exp_type):
    if(exp_type=='airfoil'):
        return [6,10,20]
    elif(exp_type=='sum_sines'):
        return [6]
    elif(exp_type=='projectile'):
        return [7]
    elif(exp_type=='BSPDE'):
        return [5,7,9]

def job_id_to_parameters(id,set_sizes,train_type,experiment_type):
    input_dimensions = input_space_dimensions(experiment_type)

    if(id==0):
        if(train_type == 'ensemble'):
            for dim in input_dimensions:
                erase_data(train_type,'QMC',set_sizes,experiment_type,dim)
                erase_data(train_type,'MC',set_sizes,experiment_type,dim)
        else:
            for dim in input_dimensions:
                erase_data(train_type,'MC',set_sizes,experiment_type,dim)

    params = []
    if (train_type == 'ensemble'):

        widths = [6,12,24]
        depths = [4,8,16]
        learning_rates = [0.01,0.001]
        lambdas = [1.0e-04,1.0e-05,1.0e-06,1e-07]
        sampling = ['QMC','MC']

        for lr in learning_rates:
            for lamb in lambdas:
                for w in widths:
                    for d in depths:
                        model = Network_architecture(lr,lamb,w,d)
                        for N in set_sizes:
                            for dim in input_dimensions:
                                for samp in sampling:
                                    params.append([model, N, dim, samp])
        return params[id]

    else:
        sampling = ['MC']
        num_inits = 100
        for N in set_sizes:
            for i in range(num_inits):
                for dim in input_dimensions:
                    for samp in sampling:
                        model = ensemble_postprocessing.best_network(N,samp,experiment_type,dim)
                        params.append([model, N, dim, samp])

        return params[id]
