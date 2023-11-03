import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pdb
import torch
# Import utilities
from utilities_module import *

# Load data
data_path = '/groups/astuart/mtrautne/learnHomData/data/vor_data_tiny.pkl'
with open(data_path, 'rb') as handle:
    A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)
# Combine the two chis along an extra axis
sgc = x_ticks.shape[1] # Ex: 128
N_data = chi1_true.shape[1]
data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components

# Input shape (of x): (batch, channels_in, nx_in, ny_in)
data_input = np.transpose(data_input, (0,3,1,2))
data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes
data_output1 = np.reshape(data_output1,(N_data,sgc, sgc))
data_output2 = np.reshape(data_output2,(N_data,sgc, sgc))

chi_true = np.stack((data_output1,data_output2), axis=3)
chi_true = np.transpose(chi_true, (0,3,1,2))

# Plot the first three samples of A and chi_true in 2d
plt.figure()
plt.subplots(2,3)
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(data_input[i,0,:,:])
    plt.colorbar()
    plt.title("A")
    plt.subplot(2,3,i+4)
    plt.imshow(chi_true[i,0,:,:])
    plt.colorbar()
    plt.title("chi_true")
plt.savefig("../Figures/A_chi_true.pdf", format="pdf", bbox_inches="tight")

x = data_input[:3,:,:,:]
y = chi_true[:3,:,:,:]
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
chi_true = torch.from_numpy(chi_true).float()

A = torch.from_numpy(data_input).float()
A = convert_A_to_matrix_shape(A)
output = frob_harm_mean_A(A)
Abar = compute_Abar(A,chi_true)
Aharm = frob_harm_mean_A(A)
Amean = frob_arithmetic_mean_A(A)
Abar_frob = torch.norm(Abar, dim = (1,2))

# Scatter plot norms
plt.figure()
plt.title("Measures of A")
plt.scatter(np.linspace(1,N_data,N_data),Amean,label = "Frob norm of arithmetic mean")
plt.scatter(np.linspace(1,N_data,N_data),Aharm,label = "Frob norm of harmonic mean")
plt.scatter(np.linspace(1,N_data,N_data),Abar_frob,label = r"Frob norm of true $\bar{A}$")
plt.legend()
plt.savefig("../Figures/norm_test.pdf", format="pdf", bbox_inches="tight")


# # Compute norms
# loss_func = H1Loss().squared_H1
# loss = loss_func(x,y)
# print(loss)



# Concatenate chis
