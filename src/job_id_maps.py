import os

class network_architecture():
    def __init__(self, learning_rate, regression_param, width, depth):
        self.learning_rate = learning_rate
        self.regression_param = regression_param
        self.width = width
        self.depth = depth


def erase_data(train_type,sampling_method,set_sizes,exp_type):
    cwd = os.getcwd()
    if(train_type == 'ensemble'):
        for N in set_sizes:
            open(cwd + '/../data/'+exp_type+'/results/'+sampling_method+'/ensemble/results_'+str(int(N))+'.txt',
                 'w').close()

    else:
        for N in set_sizes:
            open(cwd + '/../data/' + exp_type + '/results/' + sampling_method + '/retrain/results_' + str(
                int(N)) + '.txt',
                 'w').close()

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

    if(id==0):
        if(train_type == 'ensemble'):
            erase_data(train_type,'QMC',set_sizes,experiment_type)
            erase_data(train_type,'MC',set_sizes,experiment_type)
        else:
            erase_data(train_type,'MC',set_sizes,experiment_type)

    params = []

    widths = [6,12,24]
    depths = [4,8,16]
    learning_rates = [0.01,0.001]
    lambdas = [1.0e-04,1.0e-05,1.0e-06,1e-07]
    input_dimensions = input_space_dimensions(experiment_type)
    sampling = ['QMC','MC']

    for lr in learning_rates:
        for lamb in lambdas:
            for w in widths:
                for d in depths:
                    model = network_architecture(lr,lamb,w,d)
                    for N in set_sizes:
                        for dims in input_dimensions:
                            for samp in sampling:
                                params.append([model, N, dims, samp])
    return params[id][:]
