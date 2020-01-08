import numpy as np
import torch
import os
import pandas as pd

def reader(sampling_method,dim,N,exp_type):
    cwd = os.getcwd()

    if (sampling_method == "QMC"):
        train = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/sobol_train_d' + str(int(dim)) + '.csv',
                            delimiter=',').values
        train_x = torch.from_numpy(train[:int(N), :-1])
        train_y = torch.from_numpy(train[:int(N), -1])
        test = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/sobol_test_d' + str(int(dim)) + '.csv',
                           delimiter=',').values
        test_x = torch.from_numpy(test[:, :-1])
        test_y = torch.from_numpy(test[:, -1])
    else:
        train = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/rnd_train_d' + str(int(dim)) + '.csv',
                            delimiter=',').values
        train_x = torch.from_numpy(train[:int(N), :-1])
        train_y = torch.from_numpy(train[:int(N), -1])
        test = pd.read_csv(cwd + '/../data/' + exp_type + '/inputs/rnd_test_d' + str(int(dim)) + '.csv',
                           delimiter=',').values
        test_x = torch.from_numpy(test[:, :-1])
        test_y = torch.from_numpy(test[:, -1])

    return train_x, train_y, test_x, test_y