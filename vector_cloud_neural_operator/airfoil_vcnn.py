#!/usr/bin/env python
# coding: utf-8

# system modules
import os
import time
from pathlib import Path

# scientific computing
import random
import numpy as np
from numpy import linalg as LA
import pandas as pd
np.random.seed(42)

# plotting
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# pytorch importing
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim import lr_scheduler, Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
torch.manual_seed(42)

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

device_cpu = torch.device('cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,64)
        self.fc6 = nn.Linear(64,64)
        
        self.fc7 = nn.Linear(256,128)
        self.fc8 = nn.Linear(128,1)
        
    def forward(self, X):
        R1 = X           # Batchsize * stencil_size * n_features
        stencil = R1.shape[1]
        R2 = X[:,:,4:11] 
        G1 = self.relu(self.fc1(R2))
        G1 = self.relu(self.fc2(G1))
        G1 = self.relu(self.fc3(G1))
        G1 = self.relu(self.fc4(G1))
        G1 = self.relu(self.fc5(G1))
        G1 = self.fc6(G1)
        G2 = G1[:,:,0:4]
        
        G1t = G1.permute(0,2,1)
        R1t = R1.permute(0,2,1)
        D1 = torch.bmm(G1t,R1)/stencil
        D2 = torch.bmm(R1t,G2)/stencil
        D = torch.bmm(D1,D2)   # D = G1t*R*Rt*G2
        
        Dp = D.reshape(D.shape[0],-1)
        out = self.relu(self.fc7(Dp))
        out = self.fc8(out)
        
        return out


iter_print = 50

def train(train_loader, valid_loader, num_epoch):
    train_err_hist = torch.cuda.FloatTensor(1,1).fill_(0)
    valid_err_hist = torch.cuda.FloatTensor(1,1).fill_(0)
    train_loss_hist = torch.cuda.FloatTensor(1,1).fill_(0)
    valid_loss_hist = torch.cuda.FloatTensor(1,1).fill_(0)

    for epoch in range(num_epoch+1):
        train_loss_array = torch.cuda.FloatTensor(1,1).fill_(0)
        train_err_rate_num = torch.cuda.FloatTensor(1,1).fill_(0)
        train_err_rate_den = torch.cuda.FloatTensor(1,1).fill_(0)
        valid_loss_array = torch.cuda.FloatTensor(1,1).fill_(0)
        valid_err_rate_num = torch.cuda.FloatTensor(1,1).fill_(0)
        valid_err_rate_den = torch.cuda.FloatTensor(1,1).fill_(0)

        for i, data in enumerate(train_loader):
            features, target = data
            optimizer.zero_grad()
            forward = model(features)
            loss = loss_fn(forward, target)
            loss.backward()
            optimizer.step()

            train_loss_array = torch.cat((train_loss_array, torch.cuda.FloatTensor([[loss.item()]])))
            train_err_num, train_err_den = report_err_rate(target, forward)
            train_err_rate_num = torch.cat((train_err_rate_num, (train_err_num.view(1,-1))**2), 0)
            train_err_rate_den = torch.cat((train_err_rate_den, (train_err_den.view(1,-1))**2), 0)

        train_loss = torch.mean(train_loss_array)
        train_err_rate = 100*((torch.sum(train_err_rate_num, 0))**0.5)/((torch.sum(train_err_rate_den, 0))**0.5)

        exp_lr_scheduler.step()

        with torch.no_grad():
            for i, data_valid in enumerate(valid_loader):
                features_valid, target_valid = data_valid
                forward_valid = model(features_valid)
                pred_loss = loss_fn(forward_valid, target_valid)

                valid_loss_array = torch.cat((valid_loss_array, torch.cuda.FloatTensor([[pred_loss.item()]])))
                valid_err_num, valid_err_den = report_err_rate(target_valid, forward_valid)
                valid_err_rate_num = torch.cat((valid_err_rate_num, (valid_err_num.view(1,-1))**2), 0)
                valid_err_rate_den = torch.cat((valid_err_rate_den, (valid_err_den.view(1,-1))**2), 0)

            valid_loss = torch.mean(valid_loss_array)
            valid_err_rate = 100*((torch.sum(valid_err_rate_num, 0))**0.5)/((torch.sum(valid_err_rate_den, 0))**0.5)

        verb = True if epoch % iter_print == 0 else False
        if (verb):
            train_loss_hist = torch.cat((train_loss_hist, torch.cuda.FloatTensor([[train_loss]])))
            train_err_hist = torch.cat((train_err_hist, train_err_rate.view(1,-1)), 0)
            valid_loss_hist = torch.cat((valid_loss_hist, torch.cuda.FloatTensor([[valid_loss]])))
            valid_err_hist = torch.cat((valid_err_hist, valid_err_rate.view(1,-1)), 0)
        verb = True if (epoch % iter_print == 0) else False
        if (verb) :
            print('{:4}   lr: {:.2e}   train_loss: {:.2e}   valid_loss: {:.2e}   train_error:{:7.2f}%   valid_error:{:7.2f}%'                   .format(epoch, exp_lr_scheduler.get_last_lr()[0], train_loss, valid_loss, train_err_rate[0], valid_err_rate[0]))
            
    print('Finished Training')
    return train_loss_hist, train_err_hist, valid_loss_hist, valid_err_hist


def report_err_rate(target, forward):
    errRate_sigma_num = torch.norm(forward - target, dim = 0)
    errRate_sigma_den = torch.norm(target, dim = 0)
    return errRate_sigma_num, errRate_sigma_den


def evaluate(model, data_loader):
    valid_loss_array = torch.FloatTensor(1,1).fill_(0)
    valid_err_rate_num = torch.FloatTensor(1,1).fill_(0)
    valid_err_rate_den = torch.FloatTensor(1,1).fill_(0)
    with torch.no_grad():
        for i, data_valid in enumerate(data_loader):
            features_valid, target_valid = data_valid
            forward_valid = model(features_valid)
            pred_loss = loss_fn(forward_valid, target_valid).to(device_cpu)

            valid_loss_array = torch.cat((valid_loss_array, torch.FloatTensor([[pred_loss.item()]])))
            valid_err_num, valid_err_den = report_err_rate(target_valid.to(device_cpu), forward_valid.to(device_cpu))
            valid_err_rate_num = torch.cat((valid_err_rate_num, (valid_err_num.view(1,-1))**2), 0)
            valid_err_rate_den = torch.cat((valid_err_rate_den, (valid_err_den.view(1,-1))**2), 0)

        valid_loss = torch.mean(valid_loss_array)
        valid_err_rate = 100*((torch.sum(valid_err_rate_num, 0))**0.5)/((torch.sum(valid_err_rate_den, 0))**0.5)
    return valid_loss.detach().numpy(), valid_err_rate[0].detach().numpy()

stencil_size = 150
sample = random.sample(range(0, 150), stencil_size)

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

dataX_valid_5 = dataX_valid_5[:,sample].float().to(device)
dataY_valid_5 = dataY_valid_5.float().to(device)
dataX_valid_15 = dataX_valid_15[:,sample].float().to(device)
dataY_valid_15 = dataY_valid_15.float().to(device)
dataX_valid_25 = dataX_valid_25[:,sample].float().to(device)
dataY_valid_25 = dataY_valid_25.float().to(device)
dataX_valid_35 = dataX_valid_35[:,sample].float().to(device)
dataY_valid_35 = dataY_valid_35.float().to(device)

dataX_valid = torch.cat((dataX_valid_5, dataX_valid_15, dataX_valid_25, dataX_valid_35),0)
dataY_valid = torch.cat((dataY_valid_5, dataY_valid_15, dataY_valid_25, dataY_valid_35),0)

print(dataX_train.shape, dataY_train.shape, dataX_valid.shape, dataY_valid.shape)
print(dataX_train.type(), dataY_train.type(), dataX_valid.type(), dataY_valid.type())

dataX_train = dataX_train[:,sample].to(device) #[::100]
dataY_train = dataY_train.to(device) #[::100]
dataX_valid = dataX_valid.to(device)
dataY_valid = dataY_valid.to(device)
# print(dataX_train.type(), dataY_train.type(), dataX_valid.type(), dataY_valid.type())

batches = 800
batch_size_train = int(dataX_train.shape[0]/batches +1)
batch_size_valid = int((dataX_valid.shape[0]/batches) +1)

batch_size_valid_5 = int((dataX_valid_5.shape[0]/140) +1)
batch_size_valid_15 = int((dataX_valid_15.shape[0]/140) +1)
batch_size_valid_25 = int((dataX_valid_25.shape[0]/140) +1)
batch_size_valid_35 = int((dataX_valid_35.shape[0]/140) +1)

data_train = TensorDataset(dataX_train,dataY_train)
data_valid = TensorDataset(dataX_valid,dataY_valid)

data_valid_5 = TensorDataset(dataX_valid_5,dataY_valid_5)
data_valid_15 = TensorDataset(dataX_valid_15,dataY_valid_15)
data_valid_25 = TensorDataset(dataX_valid_25,dataY_valid_25)
data_valid_35 = TensorDataset(dataX_valid_35,dataY_valid_35)

train_loader = DataLoader(data_train, batch_size=batch_size_train, drop_last=True, shuffle=True)
valid_loader = DataLoader(data_valid, batch_size=batch_size_valid, shuffle=False)

valid_loader_5 = DataLoader(data_valid_5, batch_size=batch_size_valid_5, shuffle=False)
valid_loader_15 = DataLoader(data_valid_15, batch_size=batch_size_valid_15, shuffle=False)
valid_loader_25 = DataLoader(data_valid_25, batch_size=batch_size_valid_25, shuffle=False)
valid_loader_35 = DataLoader(data_valid_35, batch_size=batch_size_valid_35, shuffle=False)

print('stencil size: {} \ntrain: Input:{}  Label:{} \nvalid: Input:{}  Label:{}'       .format(stencil_size,dataX_train.shape,dataY_train.shape,dataX_valid.shape,dataY_valid.shape))

if not os.path.exists('output_files'):
    os.makedirs('output_files')

np.random.seed(7)
model = Net()
model.to(device)
loss_fn = nn.MSELoss(reduction='sum')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
para_count = count_parameters(model)
print('Total Learnable Parameters: {}'.format(para_count))


# training
num_epoch = 14000
learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=900, gamma=0.7)


start_time = time.time()
training_loss_history, training_error_history, valid_loss_history, valid_error_history = train(train_loader, valid_loader, num_epoch)
elapsed = time.time() - start_time                
print('Training time: %.1f s' % (elapsed))


def plot_history(y1, y2, yname, fname, epochs, iter_print, scale):
    plt.figure()
    plt.plot(np.arange(0, epochs+1)[iter_print*2::iter_print], y1[2:], color='tab:blue', lw=1.7, label='Training')
    plt.plot(np.arange(0, epochs+1)[iter_print*2::iter_print], y2[2:], color='tab:red', lw=1.7, label='Validation')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xlabel(r'Epochs',fontsize=15)
    plt.ylabel(yname,fontsize=17)
    plt.yscale(scale)
    plt.grid(True, which='both',axis='both',lw='0.3',color='grey',alpha=0.5)
    plt.legend(shadow=True, loc='upper right', fontsize='x-large')
    plt.savefig('output_files/'+fname+'.pdf',bbox_inches='tight')


yname = 'Loss'
fname = 'loss_vcnn_'+str(stencil_size)
plot_history(training_loss_history[1:,0].to(device_cpu).detach().numpy(), valid_loss_history[1:,0].to(device_cpu).detach().numpy(), yname, fname, num_epoch, iter_print, scale = 'log')

yname = 'Error %'
fname = 'error_vcnn_'+str(stencil_size)
plot_history(training_error_history[1:,0].to(device_cpu).detach().numpy(), valid_error_history[1:,0].to(device_cpu).detach().numpy(), yname, fname, num_epoch, iter_print, scale = 'linear')


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


torch.save(model, 'output_files/airfoil_vcnn_'+str(stencil_size)+'_gpu.pt')
model.to(device_cpu)
torch.save(model, 'output_files/airfoil_vcnn_'+str(stencil_size)+'_cpu.pt')
