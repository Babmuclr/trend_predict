#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
import datetime as dt
year_pattern = r'([1-2][0-9]{3})'
import time
from tqdm import tqdm
import random

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import utils

from scipy.stats import pearsonr
import argparse

parser = argparse.ArgumentParser(description='Parse command line options.')

# parser.add_argument(
#     '-m',
#     '--mode',
#     type = str,
#     help = 'mode',
#     required = True)

parser.add_argument(
    '-d',
    '--data',
    type = str,
    help = 'dataset',
    required = True)

parser.add_argument(
    '-s',
    '--shape',
    type = str,
    help = 'shaping',
    required = True)

options = parser.parse_args()

DATAPATH = "../../data/DBLP/"
OUTPUTPATH = "../../result/aminer/"

VERSION = options.data + "_" + options.shape  # + "_" + options.mode
CUDA = input("0,1,2,3？\n")

device = torch.device("cuda:"+CUDA if torch.cuda.is_available() else 'cpu')
print("GPUを使っているかどうか？    {}".format(torch.cuda.is_available()))

start_time = time.perf_counter()

if options.data == "diff":
    train_y = np.load(DATAPATH + "train_y_diff.npy")
    test_y = np.load(DATAPATH + "test_y_diff.npy")
else:
    train_y = np.load(DATAPATH + "train_y.npy")
    test_y = np.load(DATAPATH + "test_y.npy")
    
train_x = np.load(DATAPATH + "train_x.npy")
test_x = np.load(DATAPATH + "test_x.npy")

N = train_x.shape[0]
M = train_x.shape[2]

train_x, train_y = train_x[np.triu_indices(n=N, k=1)], train_y[np.triu_indices(n=N, k=1)]
test_x, test_y = test_x[np.triu_indices(n=N, k=1)], test_y[np.triu_indices(n=N, k=1)]

P = train_x.shape[0]

# weight = np.where(train_y == 0.0, True, False) # if target == 0 : True else: False
# weight_inv = np.where(weight, False, True) # if target == 0 : False else: True

# 0と1以上のデータを同数にする
# if options.mode == "same":    
#     train_x_0, train_y_0 = train_x[weight], train_y[weight]
#     train_x_1, train_y_1 = train_x[weight_inv], train_y[weight_inv]
#     train_x_0, train_y_0 = utils.shuffle(train_x_0, train_y_0)
#     train_x_0, train_y_0 = train_x_0[:len(train_x_1)], train_y_0[:len(train_y_1)]
#     train_x, train_y = np.concatenate([train_x_0, train_x_1]), np.concatenate([train_y_0, train_y_1])
#     print("0のデータ数: {}, 1以上のデータ数: {}".format(len(train_x_0), len(train_x_1)))
    
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

if options.shape == "norm":
    train_y = (train_y - np.mean(train_y)) / np.std(train_y)
elif options.shape == "max":
    train_y = train_y / np.max(train_y)
else:
    print("error")
    exit()

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=int(P*0.1), random_state=0)

train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)
val_x, val_y = torch.Tensor(val_x), torch.Tensor(val_y)

train_dataset = torch.utils.data.TensorDataset(train_x,train_y)
test_dataset = torch.utils.data.TensorDataset(test_x,test_y)
val_dataset = torch.utils.data.TensorDataset(val_x,val_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(M, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Model().to(device)

epochs = 1000
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
losses = []
scores = []
start_time = time.perf_counter()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    score = 0.0
    if epoch == 0:
        print("開始時間: {}".format(time.perf_counter() - start_time))
    for num, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_function = nn.MSELoss()
        outputs = outputs.reshape(-1)
        loss = loss_function(outputs, labels.view_as(outputs))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if not (loss.item() > 0):
            print(num, inputs, loss.item)
            break
    with open(OUTPUTPATH + "loss/"+VERSION+".txt", mode="a") as f:
        f.write(str(round(total_loss,3)) + "\n")
    losses.append(total_loss)
            
    model.eval()
    actual_labels, pred_labels = np.empty(0), np.empty(0)
    for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)        
        loss_function = nn.MSELoss()
        outputs = outputs.reshape(-1)
        loss_s = loss_function(outputs, labels.view_as(outputs))
        score += loss_s.item()

    print("経過時間: {}, epoch数: {}, loss: {}, score: {}".format(
        round(time.perf_counter() - start_time, 3),
        epoch, 
        round(total_loss,3),
        round(score, 3)))
    
    if epoch >10 and score > np.mean(scores[-10:]):
        scores.append(score)
        break
    scores.append(score)

print("predict test data")
model.eval()
actual_labels, pred_labels = np.empty(0), np.empty(0)
score = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    outputs = outputs.reshape(-1)
    actual_labels = np.concatenate([actual_labels, np.array(labels.cpu().numpy())])
    pred_labels = np.concatenate([pred_labels, np.array(outputs.cpu().detach().numpy())])

print("save model")
torch.save(model.to('cpu').state_dict(), OUTPUTPATH + "model/nn_"+VERSION+".pth")

print(pearsonr(actual_labels, pred_labels))