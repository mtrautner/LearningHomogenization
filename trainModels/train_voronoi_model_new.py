import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import gc
import pickle as pkl
import pdb
import numpy as np
import scipy.io
import h5py
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

from models.func_to_func2d import FNO2d
from util.utilities_module import LpLoss, UnitGaussianNormalizer

gc.collect()
GET_ERROR = 1
torch.cuda.empty_cache()

modes_params = [6,12,18,24,36,48]
model_sizes = [144,288,576,1152]
modes, sizes = np.meshgrid(modes_params, model_sizes)
modes = modes.flatten()
sizes = sizes.flatten()

Ndata = 9500
# Ndata = [2000,4000,6000,8000]

USE_CUDA = True
index = int(sys.argv[1])
# print(index)
NMODES = 24 # modes[index]
# SIZE = sizes[index]
WIDTH = 48 #SIZE//NMODES
NDATA = Ndata #[int(index)]

print("Modes = %d" %NMODES)
print("Width = %d" %WIDTH)
# print("Model size = %d" %SIZE)
print("Modes*Width = %d" %(NMODES*WIDTH))
# net_name = "Exp41_FNO_data_" + str(NDATA)

net_names = ["Exp_54_vor_chi1", "Exp_54_vor_chi2"]
net_name = net_names[index]

USE_CUDA = True
data_path = '/groups/astuart/mtrautne/Singularities/data/voronoi_data_54.pkl'

with open(data_path, 'rb') as handle:
    A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)


sgc = x_ticks.shape[1] # Ex: 512


(N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)

Ntotal     = N_data
train_size = 9500
test_start = 9500
test_end = 10000

N_test = test_end - test_start

data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes

data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components

# Input shape (of x): (batch, channels_in, nx_in, ny_in)
data_input = np.transpose(data_input, (0,3,1,2))
data_input = data_input[:,0,:,:] # Just take the first component of A
data_input = np.reshape(data_input, (N_data,1,sgc, sgc))
# Output shape:       (batch, channels_out, nx_out, ny_out)
data_output1 = np.reshape(data_output1,(N_data,sgc, sgc))
data_output2 = np.reshape(data_output2,(N_data,sgc, sgc))

# # If you want to slice the data, change s here. Not super sure what it does relative to the 2d grid so test that for sure.
s = 1


#============================ Morbin Time ====================================#
x_train = data_input[0:train_size,:,:,:]
x_test = data_input[test_start:test_end,:,:]

if index ==0: 
    y_train = data_output1[0:train_size,:,:]
    y_test  = data_output1[test_start:test_end,:,:]
elif index ==1:
    y_train = data_output2[0:train_size,:,:]
    y_test  = data_output2[test_start:test_end,:,:]

loss_func = LpLoss()

# testsize = x_test.shape[0]

# # 
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()



d_in = 1 # 
d_out = 1 # Learning just chi1 

s_outputspace = (sgc//s,sgc//s)

net = FNO2d(modes1 = NMODES, modes2= NMODES, width = WIDTH, d_in = d_in, d_out = d_out, s_outputspace= s_outputspace)
# Number of training epochs
if USE_CUDA:
    net.cuda()

epochs = 300

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)


# # To reload parameters of net and optimizer mid-training
# checkpoint = torch.load("TrainedNetExp50B1_PC2_analytic_data")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# net.load_state_dict(torch.load("TrainedNetExp31_PC2_analytic_data"))

# Batch size
b_size = 50

# Wrap traning data in loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)

# # Train neural net

train_err = np.zeros((epochs,))
test_err = np.zeros((epochs,))
y_test_approx_all = torch.zeros(test_end-test_start,sgc, sgc)

# y_test_approx = torch.zeros(test_end-test_start,N_nodes).cuda()
# y_train_approx = torch.zeros(train_size,N_nodes).cuda()


# for ep in tqdm(range(epochs)):

#     train_loss = 0.0
#     test_loss  = 0.0
#     for x, y in train_loader:
#         optimizer.zero_grad()
#         x = x.cuda()
#         y = y.cuda()

#         # y_approx = torch.zeros(b_size,N_nodes).cuda()
#         y_true  = y

#         y_approx = net(x).squeeze()
    
#         # For forward net: 
#         # Input shape (of x):     (batch, channels_in, nx_in, ny_in)
#         # Output shape:           (batch, channels_out, nx_out, ny_out)
        
#         # The input resolution is determined by x.shape[-2:]
#         # The output resolution is determined by self.s_outputspace
#         loss = loss_func(y_approx,y_true)
#         loss.backward()
#         train_loss = train_loss + loss.item()

#         optimizer.step()
#     scheduler.step()

#     with torch.no_grad():
#         b=0
#         for x,y in test_loader:
#             x = x.cuda()
#             y = y.cuda()
#             y_test_approx = net(x)
#             t_loss = loss_func(y_test_approx,y)
#             test_loss = test_loss + t_loss.item()
#             if ep == epochs - 1:
#                 y_test_approx_all[b*b_size:(b+1)*b_size,:,:] = torch.squeeze(y_test_approx).cpu()
#                 b = b+1

#     train_err[ep] = train_loss/len(train_loader)
#     test_err[ep]  = test_loss/len(test_loader)
#     print(ep, train_err[ep],test_err[ep])

    #if test_err[ep] <= 0.058:
    #    break
# torch.save({'model_state_dict':net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, "TrainedNets/TrainedNet" + net_name)

# print("Model Saved")

# Load model
net.load_state_dict(torch.load("TrainedNets/TrainedNet" + net_name)['model_state_dict'])
# Load train and test error
with open('TrainedNets/'+net_name+'.pkl', 'rb') as handle:
    foo1, foo2, train_err,test_err = pkl.load(handle)

test_loss = 0
with torch.no_grad():
    b=0
    for x,y in test_loader:
        x = x.cuda()
        y = y.cuda()
        y_test_approx = net(x)
        t_loss = loss_func(y_test_approx,y)
        test_loss = test_loss + t_loss.item()
        y_test_approx_all[b*b_size:(b+1)*b_size,:,:] = torch.squeeze(y_test_approx).cpu()
        b = b+1


# grid_edge = sgc
# def standard_rel_H1_err(true,pred):
#     # True and pred have dimensions N_data, grid_edge, grid_edge
#     # Returns vector N_data of relative H1 errors
#     L2_err = np.linalg.norm(pred - true,axis = (1,2))
#     gradient_diff = np.asarray(np.gradient(pred-true,1/grid_edge,axis = (1,2)))
#     norm_grad = np.linalg.norm(gradient_diff,axis = 0) # Size N_data , grid_edge, grid_edge
#     grad_L2_err = np.linalg.norm(norm_grad,axis = (1,2))
#     err_H1_norm = (L2_err**2 + grad_L2_err**2)**(1/2)

#     true_L2_norm = np.linalg.norm(true,axis = (1,2))
#     true_grad_norm = np.linalg.norm(np.asarray(np.gradient(true,1/grid_edge,axis = (1,2))),axis = 0)
#     true_grad_L2_norm = np.linalg.norm(true_grad_norm,axis = (1,2))
#     true_H1_norm = (true_L2_norm**2 + true_grad_L2_norm**2)**(1/2)

#     return err_H1_norm/true_H1_norm

# torch.cuda.empty_cache()
# gc.collect()
# h1_error = 0
# if GET_ERROR == True: 
#     y_test  = y_test.detach().cpu().numpy()
#     y_test_approx_all = y_test_approx_all.detach().cpu().numpy()
#     h1_error = np.mean(standard_rel_H1_err(y_test,y_test_approx_all))

   
#     print("H1_error = %.6f" %h1_error)
#     with open('TrainedNets/h1_'+net_name+'_h1error.pkl', 'wb') as handle:
#         pkl.dump([h1_error], handle, protocol=pkl.HIGHEST_PROTOCOL)
#     print("H1 Error Saved")

# Save only 5 samples
with open('TrainedNets/'+net_name+'.pkl', 'wb') as handle:
     pkl.dump([y_test[:,:,:],y_test_approx_all[:,:,:],train_err,test_err], handle, protocol=pkl.HIGHEST_PROTOCOL)

print("Data Saved")
