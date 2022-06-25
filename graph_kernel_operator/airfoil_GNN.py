#!/usr/bin/env python
# coding: utf-8
import pickle
import time
import torch
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from timeit import default_timer
from torch_geometric.loader import DataLoader
from utilities import *
from nn_conv import NNConv_old

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')
device_cpu = torch.device('cpu')


class GKernelNN(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, depth, width_kernel, dim_in, dim_out):
        super(GKernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(dim_in, dim_node)

        kernel = DenseNet([dim_edge] + width_kernel + [dim_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(dim_node, dim_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(dim_node, dim_out)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)
        x = self.fc2(x)
        return x

start_time = time.time()
dataX_train = torch.load('../data/training-data/GNN_dataX_train_150.pt')
dataY_train = torch.load('../data/training-data/GNN_dataY_train_150.pt')

dataX_valid_5 = torch.load('../data/testing-data/GNN_dataX_test_5_150.pt')
dataY_valid_5 = torch.load('../data/testing-data/GNN_dataY_test_5_150.pt')
dataX_valid_15 = torch.load('../data/testing-data/GNN_dataX_test_15_150.pt')
dataY_valid_15 = torch.load('../data/testing-data/GNN_dataY_test_15_150.pt')
dataX_valid_25 = torch.load('../data/testing-data/GNN_dataX_test_25_150.pt')
dataY_valid_25 = torch.load('../data/testing-data/GNN_dataY_test_25_150.pt')
dataX_valid_35 = torch.load('../data/testing-data/GNN_dataX_test_35_150.pt')
dataY_valid_35 = torch.load('../data/testing-data/GNN_dataY_test_35_150.pt')

dataX_valid_5 = dataX_valid_5.float()
dataY_valid_5 = dataY_valid_5.float()
dataX_valid_15 = dataX_valid_15.float()
dataY_valid_15 = dataY_valid_15.float()
dataX_valid_25 = dataX_valid_25.float()
dataY_valid_25 = dataY_valid_25.float()
dataX_valid_35 = dataX_valid_35.float()
dataY_valid_35 = dataY_valid_35.float()

dataX_valid = torch.cat((dataX_valid_5, dataX_valid_15, dataX_valid_25, dataX_valid_35),0)
dataY_valid = torch.cat((dataY_valid_5, dataY_valid_15, dataY_valid_25, dataY_valid_35),0)

print(dataX_train.shape, dataY_train.shape, dataX_valid.shape, dataY_valid.shape)

dataX_train = dataX_train #[::100]
dataY_train = dataY_train #[::100]
dataX_valid = dataX_valid[::200]
dataY_valid = dataY_valid[::200]

print(dataX_train.shape, dataY_train.shape, dataX_valid.shape, dataY_valid.shape)

# parameters
stencil_size = 150
batches = 800
batch_size_train = int(dataX_train.shape[0]/batches +1)
batch_size_valid = batch_size_train #int((dataX_valid.shape[0]/4) +1)

batch_size_valid_5 = int((dataX_valid_5.shape[0]/140) +1)
batch_size_valid_15 = int((dataX_valid_15.shape[0]/140) +1)
batch_size_valid_25 = int((dataX_valid_25.shape[0]/140) +1)
batch_size_valid_35 = int((dataX_valid_35.shape[0]/140) +1)

dim_node = 16
dim_edge = None
depth = 2
width_kernel = [64,96]
dim_in = None
dim_out = 1

learning_rate = 0.001
scheduler_step = 120
scheduler_gamma = 0.7
epochs = 1400

pwd = np.zeros([stencil_size, stencil_size])
edge_index = np.vstack(np.where(pwd <= 1))
edge_index = torch.tensor(edge_index, dtype=torch.long)

def data_preprocess_full(X, Y, size=None, seed=None):
    size = size or X.shape[0]
    if seed:
        np.random.seed(seed)
        n = X.shape[0]
        idx = np.random.choice(n, size, replace=False)
        X = X[idx, :stencil_size, :]
        Y = Y[idx]
    else:
        X = X[:size, :stencil_size, :]
        Y = Y[:size]
    edge_attr = torch.cat([X[:, edge_index[0]], X[:, edge_index[1]]], dim=2)
    data = []
    for j in range(size):
        data.append(Data(x=X[j], y=Y[j], edge_index=edge_index, edge_attr=edge_attr[j]))
    return data

def data_preprocess_inv(X, Y, size=None, seed=None):
    size = size or X.shape[0]
    if seed:
        np.random.seed(seed)
        n = X.shape[0]
        idx = np.random.choice(n, size, replace=False)
        X = X[idx, :stencil_size, :]
        Y = Y[idx]
    else:
        X = X[:size, :stencil_size, :]
        Y = Y[:size]
    attr1, attr2 = X[:, edge_index[0]], X[:, edge_index[1]]
    xy_prod = torch.sum(attr1[:, :, :2] * attr2[:, :, :2], dim=2, keepdims=True)
    uv_prod = torch.sum(attr1[:, :, 2:4] * attr2[:, :, 2:4], dim=2, keepdims=True)
    edge_attr = torch.cat([xy_prod, uv_prod, attr1[:, :, 4:], attr2[:, :, 4:]], dim=2)
    data = []
    for j in range(size):
        data.append(Data(x=X[j, :, 4:], y=Y[j], edge_index=edge_index, edge_attr=edge_attr[j]))
    return data

num_train = dataX_train.shape[0]
num_valid = dataX_valid.shape[0]

func_data_preprocess = data_preprocess_inv
data_train = func_data_preprocess(dataX_train, dataY_train)
data_valid = func_data_preprocess(dataX_valid, dataY_valid)

train_loader = DataLoader(data_train, batch_size=batch_size_train, drop_last=True, shuffle=True)
valid_loader = DataLoader(data_valid, batch_size=batch_size_valid, shuffle=False)

data_valid_5 = func_data_preprocess(dataX_valid_5, dataY_valid_5)
data_valid_15 = func_data_preprocess(dataX_valid_15, dataY_valid_15)
data_valid_25 = func_data_preprocess(dataX_valid_25, dataY_valid_25)
data_valid_35 = func_data_preprocess(dataX_valid_35, dataY_valid_35)

valid_loader_5 = DataLoader(data_valid_5, batch_size=batch_size_valid_5, shuffle=False)
valid_loader_15 = DataLoader(data_valid_15, batch_size=batch_size_valid_15, shuffle=False)
valid_loader_25 = DataLoader(data_valid_25, batch_size=batch_size_valid_25, shuffle=False)
valid_loader_35 = DataLoader(data_valid_35, batch_size=batch_size_valid_35, shuffle=False)

dim_edge = data_train[0]['edge_attr'].shape[-1]
dim_in = data_train[0]['x'].shape[-1]

model = GKernelNN(dim_node, dim_edge, depth, width_kernel, dim_in, dim_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
elapsed = time.time() - start_time
print('Preprocess time: %.1f s' % (elapsed))

config = dict(
    stencil_size = stencil_size,
    batch_size = batch_size_train,
    num_train = num_train,
    num_valid = num_valid,
    dim_node = dim_node,
    dim_edge = dim_edge,
    depth = depth,
    width_kernel = width_kernel,
    dim_in = dim_in,
    dim_out = dim_out,
    learning_rate = learning_rate,
    scheduler_step = scheduler_step,
    scheduler_gamma = scheduler_gamma,
    epochs = epochs)
print(config)

def evaluate(model, data_loader):
    valid_err_rate_num, valid_err_rate_den = [], []
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch.to(device))
            mse = F.mse_loss(out, batch.y.view(-1,1), reduction='sum').to(device_cpu)
            valid_err_rate_num += [mse.item()]
            valid_err_rate_den += [torch.norm(batch.y.to(device_cpu)).item()**2]
    valid_err_rate_num = np.array(valid_err_rate_num)
    valid_err_rate_den = np.array(valid_err_rate_den)
    valid_loss = np.mean(valid_err_rate_num)
    valid_err_rate = np.sqrt((np.sum(valid_err_rate_num)/np.sum(valid_err_rate_den))) * 100
    return valid_loss, valid_err_rate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total Learnable Parameters: {}'.format(count_parameters(model)))

if not os.path.exists('output_files'):
    os.makedirs('output_files')
    
if not os.path.exists('nonlocal_logs'):
    os.makedirs('nonlocal_logs')

iter_print=10
hist = np.zeros(shape=(int(epochs/iter_print)+1, 5))

start_time = time.time()
model.train()
for ep in range(epochs+1):
    train_err_rate_num, train_err_rate_den = [], []
    for batch in train_loader:
        optimizer.zero_grad()       
        out = model(batch.to(device))        
        mse = F.mse_loss(out, batch.y.view(-1,1), reduction='sum').to(device_cpu)
        mse.backward()
        optimizer.step()
        train_err_rate_num += [mse.item()]
        train_err_rate_den += [torch.norm(batch.y.to(device_cpu)).item()**2]
        
    train_err_rate_num = np.array(train_err_rate_num)
    train_err_rate_den = np.array(train_err_rate_den)
    train_loss = np.mean(train_err_rate_num)
    train_err_rate = np.sqrt((np.sum(train_err_rate_num)/np.sum(train_err_rate_den))) * 100

    scheduler.step()

    verb = True if (ep % iter_print == 0) else False
    if (verb):
        valid_loss, valid_err_rate = evaluate(model, valid_loader)
        hist[int(ep/iter_print),:] = np.array([scheduler.get_last_lr()[0],train_loss,valid_loss,train_err_rate,valid_err_rate])
        print('{:4}   lr: {:.2e}   train_loss: {:.2e}   valid_loss: {:.2e}   train_error:{:7.2f}%   valid_error:{:7.2f}%'                       .format(ep,scheduler.get_last_lr()[0],train_loss,valid_loss,train_err_rate,valid_err_rate))
        
elapsed = time.time() - start_time
print('Training time: %.1f s' % (elapsed))

def plot_history(y1, y2, yname, fname, epochs, iter_print, scale):
    plt.figure()
    plt.plot(np.arange(0, epochs+1)[iter_print::iter_print], y1[1:], color='tab:blue', lw=1.7, label='Training')
    plt.plot(np.arange(0, epochs+1)[iter_print::iter_print], y2[1:], color='tab:red', lw=1.7, label='Validation')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xlabel(r'Epochs',fontsize=15)
    plt.ylabel(yname,fontsize=17)
    plt.yscale(scale)
    plt.grid(True, which='both',axis='both',lw='0.3',color='grey',alpha=0.5)
    plt.legend(shadow=True, loc='upper right', fontsize='x-large')
    plt.savefig('output_files/'+fname+'.pdf',bbox_inches='tight')

yname = 'Loss'
fname = 'loss_gnn_'+str(stencil_size)
plot_history(hist[:,1], hist[:,2], yname, fname, epochs, iter_print, scale = 'log')

yname = 'Error %'
fname = 'error_gnn_'+str(stencil_size)
plot_history(hist[:,3], hist[:,4], yname, fname, epochs, iter_print, scale = 'linear')

valid_loss, valid_err_rate = evaluate(model, valid_loader)
print('valid_loss: {:.2e}   valid_error:{:7.2f}% \n'.format(valid_loss, valid_err_rate))
config["valid_loss"] = valid_loss
config["valid_err_rate"] = valid_err_rate

# Validation data for Angle of Attack = 5
valid_loss_5, valid_err_rate_5 = evaluate(model, valid_loader_5)
print('Angle of Attack = 5 deg')
print('valid_loss: {:.2e}   valid_error:{:7.2f}% \n'.format(valid_loss_5, valid_err_rate_5))

# Validation data for Angle of Attack = 15
valid_loss_15, valid_err_rate_15 = evaluate(model, valid_loader_15)
print('Angle of Attack = 15 deg')
print('valid_loss: {:.2e}   valid_error:{:7.2f}% \n'.format(valid_loss_15, valid_err_rate_15))

# Validation data for Angle of Attack = 25
valid_loss_25, valid_err_rate_25 = evaluate(model, valid_loader_25)
print('Angle of Attack = 25 deg')
print('valid_loss: {:.2e}   valid_error:{:7.2f}% \n'.format(valid_loss_25, valid_err_rate_25))

# Validation data for Angle of Attack = 35
valid_loss_35, valid_err_rate_35 = evaluate(model, valid_loader_35)
print('Angle of Attack = 35 deg')
print('valid_loss: {:.2e}   valid_error:{:7.2f}% \n'.format(valid_loss_35, valid_err_rate_35))

timestr = time.strftime("%Y%m%d_%H%M%S")
with open(f'nonlocal_logs/{timestr}.pkl', 'wb') as f:
    pickle.dump(config, f)
    pickle.dump(hist, f)
print(timestr)

torch.save(model, f'nonlocal_logs/{timestr}_model_gpu.pt')
model.to(device_cpu)
torch.save(model, f'nonlocal_logs/{timestr}_model_cpu.pt')

