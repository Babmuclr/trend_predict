#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
import datetime as dt
year_pattern = r'([1-2][0-9]{3})'
import time

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

DATAPATH = "../../data/DBLP/"
START_YEAR = 2000
END_YEAR = 2020
YEAR_STD = END_YEAR - START_YEAR

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("GPUを使っているかどうか？    {}".format(torch.cuda.is_available()))

start_time = time.perf_counter()

train_x = np.load(DATAPATH + "train_x.npy")
train_y = np.load(DATAPATH + "train_y.npy")

test_x = np.load(DATAPATH + "test_x.npy")
test_y = np.load(DATAPATH + "test_y.npy")

N = train_x.shape[0]
print("データサイズ: {}, {}".format(N, N*N))

train_x, train_y = train_x.reshape(N*N,18), train_y.reshape(N*N,1)
test_x, test_y = test_x.reshape(N*N,18), test_y.reshape(N*N,1)

train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)

train_dataset = torch.utils.data.TensorDataset(train_x,train_y)
test_dataset = torch.utils.data.TensorDataset(test_x,test_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(18, 128)
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
start_time = time.perf_counter()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    if epoch == 0:
        print("開始時間: {}".format(time.perf_counter() - start_time))
    for num, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels, = inputs.to(device), labels.to(device)
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
    with open("./loss_checker.txt", mode="a") as f:
        f.write(str(round(total_loss,3)) + "\n")
        
    model.eval()
    actual_labels, pred_labels = np.empty(0), np.empty(0)
    for num, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        labels, outputs = labels.reshape(-1), outputs.reshape(-1)
        actual_labels = np.concatenate([actual_labels, np.array(labels.cpu().numpy())])
        pred_labels = np.concatenate([pred_labels, np.array(outputs.cpu().detach().numpy())])
        if num > 100:
            break
    score = pearsonr(actual_labels, pred_labels)[0]
    
    print("経過時間: {}, epoch数: {}, loss: {}, score: {}".format(
        round(time.perf_counter() - start_time, 3),
        epoch, 
        round(total_loss,3),
        round(score, 3)))
    
    if epoch > 50 and total_loss > np.mean(losses[-10:]):
        losses.append(total_loss)
        break
    losses.append(total_loss)

model.eval()

actual_labels, pred_labels = np.empty(0), np.empty(0)
score = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    labels, outputs = labels.reshape(-1), outputs.reshape(-1)
    actual_labels = np.concatenate([actual_labels, np.array(labels.cpu().numpy())])
    pred_labels = np.concatenate([pred_labels, np.array(outputs.cpu().detach().numpy())])

torch.save(model.to('cpu').state_dict(), './model.pth')

print(pearsonr(actual_labels, pred_labels))