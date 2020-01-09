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

def job_id_to_parameters(id,set_sizes,train_type,experiment_type,sampling_method,dim):

    if(id==0):
        if(train_type == 'ensemble'):
            erase_data(train_type,sampling_method,set_sizes,experiment_type,dim)
        else:
            erase_data(train_type,sampling_method,set_sizes,experiment_type,dim)

    params = []
    if (train_type == 'ensemble'):

        widths = [6,12,24]
        depths = [4,8,16]
        learning_rates = [0.01,0.001]
        lambdas = [1.0e-04,1.0e-05,1.0e-06,1e-07]

        for lr in learning_rates:
            for lamb in lambdas:
                for w in widths:
                    for d in depths:
                        model = Network_architecture(lr,lamb,w,d)
                        for N in set_sizes:
                            params.append([model, N])
        return params[id]

    else:
        num_inits = 100
        for N in set_sizes:
            for i in range(num_inits):
                model = ensemble_postprocessing.best_network(N,sampling_method,experiment_type,dim)
                params.append([model, N])

        return params[id]
