import os

def erase_data(train_type,sampling_method,set_sizes):

    cwd = os.getcwd()
    if(train_type == 'ensemble'):
        for N in set_sizes:
            open(cwd + '/../data/sum_sines/output/'+sampling_method+
                 '/ensemble/results_' + str(int(N)) + '.txt', 'w').close()
    else:
        for N in set_sizes:
            open(cwd + '/../data/sum_sines/output/'+sampling_method+
                 '/retrain/results_' + str(int(N)) + '.txt', 'w').close()

def job_id_to_parameters(id,set_sizes,train_type):

    if(id==0):
        if(train_type == 'ensemble'):
            erase_data(train_type,'QMC',set_sizes)
            erase_data(train_type,'MC',set_sizes)
        else:
            erase_data(train_type,'MC',set_sizes)

    params = []

    widths = [6,12,24]
    depths = [4,8,16]
    step_sizes = [0.01,0.001]
    lambdas = [1.0e-04,1.0e-05,1.0e-06,1e-07]
    sampling = ['QMC','MC']

    for sz in step_sizes:
        for l in lambdas:
            for w in widths:
                for d in depths:
                    for N in set_sizes:
                        for samp in sampling:
                            params.append([sz,l,w,d,N,samp])

    return params[id]
