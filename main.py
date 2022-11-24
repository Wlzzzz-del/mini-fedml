import h5py
import matplotlib.pyplot as plt
import numpy as np
import importlib
import random
from model_util import *
import os
import torch
from server_avg import Server_Avg

torch.manual_seed(0)


## ------------在这修改参数-----------
dataset = "Mnist"
model = "cnn"
algorithm = "FedAvg"

batch_size = 20
learning_rate = 0.005
beta = 1.0
lamda = 15
num_global_iters = 800
local_epochs = 20
optimizer = "SGD"
numusers = 20
gpu = 0
times = 5
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

if model == "cnn":
    if(dataset == "Mnist"):
        model = Net().to(device), model
    if(dataset == "Cifar10"):
        model = CNNCifar(10).to(device), model
avg = Server_Avg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_global_iters, local_epochs, optimizer, numusers, times)
avg.train()