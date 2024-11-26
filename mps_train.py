#!/usr/bin/env python3
import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset,ConcatDataset,SubsetRandomSampler
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,mean_squared_error
from numpy import vstack
from mps_setup import readInput           
import os
import sys
import numpy as np
import random
from scipy.optimize import minimize
from functions import read_mps, normalization, site_normalization, calculate_Sz, calculate_ENE, set_grad_zero, orth_centre, heisenberg_hamiltonian

#####*********************************************************************#####
bond_dim, batch_size, num_epochs, learn_rate, l2_reg, inp_dim, inp_file_pth, inp_file, ref_file, input_list, det = readInput()

n_test = 0.7   #fraction of data for testing

from torchmps import MPS
torch.manual_seed(0)

# Initialize the MPS module
mps = MPS(f_name=ref_file, input_dim=inp_dim, feature_dim = 2, bond_dim=bond_dim,)

#####********************************************************************#####
# Get the training and test sets
class CSVDataset(Dataset):
    def __init__(self, path):
        df  = read_csv(path,usecols=input_list, header=None)
        df_det = read_csv(path,usecols=[det], header=None)

        self.X = df.values[:, :-1]                           # Input descriptor #
        self.y = df.values[:, -1]                            # Output #
        self.det = df_det.values[:,-1]

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.det = self.det.astype('float32')
        
        self.y = self.y.reshape((len(self.y), 1))
        self.det = self.det.reshape((len(self.det), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.det[idx]]

    def get_splits(self, n_test=n_test):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])

def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size, shuffle=True)
    return train_dl, test_dl

#####*******************************************************************************************************#####

ham = heisenberg_hamiltonian()
start = time.time()

path = inp_file_pth+inp_file
train_dl, test_dl = prepare_data(path)

dataset = ConcatDataset([train_dl, test_dl])

loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)
#optimizer = torch.optim.SGD(mps.parameters(), lr=learn_rate, weight_decay=l2_reg, momentum=0.9)

torch.save(mps.state_dict(), inp_file_pth+"model_M"+str(bond_dim)+".pth")
mps.load_state_dict(torch.load(inp_file_pth+"model_M"+str(bond_dim)+".pth",weights_only=True))
f1 = open(inp_file_pth+"train_test_loss_M"+str(bond_dim)+".out", "w")
f1.write("#Epoch" + "\t" + "train_loss" + "\t" + "test_loss" + "\n")

f2 = open(inp_file_pth+"epoch_energy_sz_M"+str(bond_dim)+".out","w")
f2.write("#Epoch"+"\t"+"Energy"+"\t"+"Sz"+"\n")

for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    running_loss_test = 0.0

    for i, (inputs, targets ,dets) in enumerate(train_dl): 
        scores = mps(inputs)
        loss = loss_fun(scores, targets)

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
    epoch_loss = running_loss/len(train_dl)

    mps.eval()
    for i, (inputs, targets, dets) in enumerate(test_dl): 
        scores = mps(inputs)
        loss = loss_fun(scores, targets)
        running_loss_test += loss.item()

    epoch_loss_test = running_loss_test/len(test_dl)
 
    #calculate energy and sz at each epoch
    mps_cut = read_mps(mps)
    mps_cut_1 = mps_cut.copy()
    totalSz = calculate_Sz(mps_cut)
    ene1 = calculate_ENE(mps_cut_1, ham)

    f1.write(str(epoch_num) + "\t" + str(epoch_loss) + "\t" + str(epoch_loss_test) + "\n")
    f2.write(str(epoch_num)+"\t"+str(ene1)+"\t"+str(totalSz)+"\n")

f1.close()
f2.close()

f3 = open(inp_file_pth+"mps_train_output_M"+str(bond_dim)+".out","w")
for i, (inputs, targets ,dets) in enumerate(train_dl):
    scores = mps(inputs)
    scores = scores.detach().numpy()
    scores = np.reshape(scores,(len(scores),1))
    actual = targets.numpy()
    det = dets.numpy()
    for j in range(len(actual)):
        f3.write(str(int(det[j][0]))+"\t"+str(actual[j][0])+"\t"+str(scores[j][0])+"\n")
f3.close()

f4 = open(inp_file_pth+"mps_test_output_M"+str(bond_dim)+".out","w")
for i, (inputs, targets ,dets) in enumerate(test_dl):
    scores = mps(inputs)
    scores = scores.detach().numpy()
    scores = np.reshape(scores,(len(scores),1))
    actual = targets.numpy()
    det = dets.numpy()
    for j in range(len(actual)):
        f4.write(str(int(det[j][0]))+"\t"+str(actual[j][0])+"\t"+str(scores[j][0])+"\n")

f4.close()

torch.save(mps.state_dict(), inp_file_pth+"converged_model_M"+str(bond_dim)+".pth")
os.remove(inp_file_pth+"model_M"+str(bond_dim)+".pth")

