import torch
from torch import nn, optim
import data_handling

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)

def NN_model(layer_sizes, activationFunction, batch_norm=True, Xavier_init=True):
    if(batch_norm):
        model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.BatchNorm1d(num_features=layer_sizes[1]),
            activationFunction())
        for i in range(1, len(layer_sizes) - 2):
            model = nn.Sequential(
                model, nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(num_features=layer_sizes[i + 1]),
                activationFunction())
    else:
        model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            activationFunction())
        for i in range(1, len(layer_sizes) - 2):
            model = nn.Sequential(
                model, nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                activationFunction())
    model = nn.Sequential(
        model, nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    if(Xavier_init):
        model.apply(init_weights)

    return model

def train_NN(model_params,N,dim,train_type,sampling_method,max_epochs,exp_type):

    layer_sizes = [dim]
    for i in range(model_params.depth):
        layer_sizes.append(model_params.width)
    layer_sizes.append(1)

    train_x, train_y, test_x, test_y = data_handling.reader(sampling_method, dim, N, exp_type)

    model = NN_model(layer_sizes, nn.Sigmoid)

    train_objective = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params.learning_rate, weight_decay=model_params.regression_param)

    for e in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x.float())
        loss = train_objective(output, train_y.float())
        loss.backward()
        optimizer.step()

    test_objective = nn.L1Loss()
    output_train = model(train_x.float())
    train_error = test_objective(output_train, train_y.float()).item()

    output_test = model(test_x.float())
    generalization_error = test_objective(output_test, test_y.float()).item()

    data_handling.writer(model_params,N,train_type,exp_type,sampling_method,train_error,generalization_error)