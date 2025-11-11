#!/usr/bin/env python3
import time
import numpy as np
np.bool = np.bool_
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset,ConcatDataset,SubsetRandomSampler
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,mean_squared_error
from numpy import vstack
import warnings
warnings.filterwarnings("ignore")

from mps_setup import readInput           
import os
import sys
import random
from scipy.optimize import minimize
from functions import read_mps, normalization, site_normalization, calculate_Sz, calculate_ENE, orth_centre, heisenberg_hamiltonian

#####*********************************************************************#####
bond_dim, batch_size, num_epochs, learn_rate, l2_reg, inp_dim, inp_file_pth, inp_file, ref_file, input_list, det = readInput()

n_test = 0.7   #fraction of data for testing

from torchmps import MPS
torch.manual_seed(0)

# Initialize the MPS module
mps = MPS(f_name=inp_file_pth+ref_file, input_dim=inp_dim, feature_dim = 2, bond_dim=bond_dim,)

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

# --- Initialize ---
ham = heisenberg_hamiltonian()
path = inp_file_pth + inp_file

train_dl, test_dl = prepare_data(path)
dataset = ConcatDataset([train_dl, test_dl])

loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, weight_decay=l2_reg)

model_file = f"{inp_file_pth}model_M{bond_dim}.pth"
torch.save(mps.state_dict(), model_file)
mps.load_state_dict(torch.load(model_file, weights_only=True))

# --- Output Files ---
train_test_loss_file = f"{inp_file_pth}train_test_loss_M{bond_dim}.out"
train_output_file = f"{inp_file_pth}mps_train_output_M{bond_dim}.out"
test_output_file = f"{inp_file_pth}mps_test_output_M{bond_dim}.out"

with open(train_test_loss_file, "w") as f1:
    f1.write("#Epoch\ttrain_loss\ttest_loss\n")

    for epoch_num in range(1, num_epochs + 1):
        # --- Training Loop ---
        mps.train()
        total_train_loss = 0.0
        for inputs, targets, _ in train_dl:
            scores = mps(inputs)
            loss = loss_fun(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_dl)

        # --- Testing Loop ---
        mps.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in test_dl:
                scores = mps(inputs)
                loss = loss_fun(scores, targets)
                total_test_loss += loss.item()

        epoch_test_loss = total_test_loss / len(test_dl)
        f1.write(f"{epoch_num}\t{epoch_train_loss}\t{epoch_test_loss}\n")

# --- Energy and Sz ---
mps_cut = read_mps(mps)
mps_cut_1 = mps_cut.copy()
totalSz = calculate_Sz(mps_cut)
totalEne = calculate_ENE(mps_cut_1, ham)

print("Total energy:", totalEne)
print("Total Sz:", totalSz)

# --- Helper function to save outputs ---
def save_outputs(dataloader, filename):
    with open(filename, "w") as f:
        with torch.no_grad():
            for inputs, targets, dets in dataloader:
                scores = mps(inputs).detach().cpu().numpy().reshape(-1, 1)
                actual = targets.cpu().numpy()
                det = dets.cpu().numpy()
                for d, a, s in zip(det, actual, scores):
                    f.write(f"{int(d[0])}\t{a[0]}\t{s[0]}\n")

# --- Save train/test outputs ---
save_outputs(train_dl, train_output_file)
save_outputs(test_dl, test_output_file)

# --- Save final model ---
torch.save(mps.state_dict(), f"{inp_file_pth}converged_model_M{bond_dim}.pth")
os.remove(model_file)

