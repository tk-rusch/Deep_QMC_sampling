import torch
import os
import pandas as pd
from filelock import Timeout, FileLock

def reader(sampling_method,dim,N,exp_type):
    cwd = os.getcwd()
    if (sampling_method == "QMC"):
        train = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/sobol_train_d' + str(int(dim)) + '.csv',
                            delimiter=',').values
        train_x = torch.from_numpy(train[:int(N), 1:-1])
        train_y = torch.from_numpy(train[:int(N), -1])
        test = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/sobol_test_d' + str(int(dim)) + '.csv',
                           delimiter=',').values
        test_x = torch.from_numpy(test[:, 1:-1])
        test_y = torch.from_numpy(test[:, -1])
    else:
        train = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/rnd_train_d' + str(int(dim)) + '.csv',
                            delimiter=',').values
        train_x = torch.from_numpy(train[:int(N), 1:-1])
        train_y = torch.from_numpy(train[:int(N), -1])
        test = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/rnd_test_d' + str(int(dim)) + '.csv',
                           delimiter=',').values
        test_x = torch.from_numpy(test[:, 1:-1])
        test_y = torch.from_numpy(test[:, -1])

    return train_x, train_y, test_x, test_y


def writer(model,set_size,train_type,exp_type,sampling_method,train_error,gen_error,dim):
    cwd = os.getcwd()

    if (train_type == 'ensemble'):
        file_path = cwd + '/../data/' + exp_type + '/results/' + sampling_method + \
                    '/ensemble/results_dim_' + str(dim) + '_N_' + str(set_size) + '.txt'

        lock = FileLock(file_path + '.lock', timeout=100)
        lock.acquire()
        try:
            open(file_path, "a").write(str(model.learning_rate) + ' ' + str(model.regression_param) + ' ' + str(model.width) + ' ' +
                                       str(model.depth) + ' ' + str(train_error) + ' ' + str(gen_error) + '\n')
        finally:
            lock.release(force=True)

    else:
        file_path = cwd + '/../data/' + exp_type + '/results/' + sampling_method + \
                    '/retrain/results_dim_' + str(dim) + '_N_' + str(set_size) + '.txt'
        lock = FileLock(file_path + '.lock', timeout=100)
        lock.acquire()
        try:
            open(file_path, "a").write(str(model.learning_rate) + ' ' + str(model.regression_param) + ' ' + str(model.width) + ' ' +
                                       str(model.depth) + ' ' + str(train_error) + ' ' + str(gen_error) + '\n')
        finally:
            lock.release(force=True)