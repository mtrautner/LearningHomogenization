import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pdb
import torch
# Import norm function
from utilities_module import H1Loss

# Load data
data_path = '/groups/astuart/mtrautne/learnHomData/data/smooth_data_tiny.pkl'
with open(data_path, 'rb') as handle:
    A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)
# Combine the two chis along an extra axis
sgc = x_ticks.shape[1] # Ex: 128
N_data = chi1_true.shape[1]

data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes
data_output1 = np.reshape(data_output1,(N_data,sgc, sgc))
data_output2 = np.reshape(data_output2,(N_data,sgc, sgc))

chi_true = np.stack((data_output1,data_output2), axis=3)
chi_true = np.transpose(chi_true, (0,3,1,2))
x = chi_true[:3,:,:,:]
y = chi_true[:3,:,:,:]
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
# Compute norms
loss_func = H1Loss().squared_H1
loss = loss_func(x,y)
print(loss)



# Concatenate chis
