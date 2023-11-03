import torch
import torch.utils.data
import gc
import pickle as pkl
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# gc.collect()

# torch.cuda.empty_cache()

# USE_CUDA = True
index = int(sys.argv[1])
A_data_files = ['smooth_data_25_.pkl','star_inc_data_24_.pkl','sq_inc_data_23_.pkl','voronoi_data_54.pkl']
A_file = A_data_files[index]
true_Abar_files = ['smooth_25_Abar.pkl', 'star_24_Abar.pkl', 'sq_23_Abar.pkl', 'voronoi_data_54_Abar.pkl']
true_Abar_file = true_Abar_files[index]
pred_chi1_files = ['TrainedNets/Exp38_FNO_m_24_w_48.pkl','TrainedNets/Exp43A_FNO.pkl','TrainedNets/Exp44A_FNO.pkl','TrainedNets/Exp_54_vor_chi1.pkl']
pred_chi1_file = pred_chi1_files[index]
pred_chi2_files = ['TrainedNets/Exp38B_FNO_m_24_w_48.pkl','TrainedNets/Exp43_FNO.pkl','TrainedNets/Exp44_FNO.pkl','TrainedNets/Exp_54_vor_chi2.pkl']
pred_chi2_file = pred_chi2_files[index]
A_data_path = '/groups/astuart/mtrautne/Singularities/data/'+A_file
true_Abar_path = '/groups/astuart/mtrautne/Singularities/data/' + true_Abar_file
pred_chi1_path = pred_chi1_file
pred_chi2_path = pred_chi2_file

with open(A_data_path, 'rb') as handle:
    A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)

with open(true_Abar_path, 'rb') as handle:
    true_Abars, true_Ameans, true_Aharms = pkl.load(handle)

with open(pred_chi1_path, 'rb') as handle:
    y_test,pred_chi1,train_err,test_err = pkl.load(handle)

with open(pred_chi2_path, 'rb') as handle:
    y_test,pred_chi2,train_err,test_err = pkl.load(handle)

print("Data loaded...")
sgc = 128
# A_input = np.transpose(A_input, (0,3,1,2))
A_input = A_input[9500:,:,:,:]
true_Abars_test = true_Abars[9500:,:,:]
true_Ameans_test = true_Ameans[9500:,:,:]
true_Aharms_test = true_Aharms[9500:,:,:]
# (N_data, dummy1, dummy2, N_nodes) = np.shape(A_input) #18A only
# A_input = np.transpose(A_input, (0,3,1,2)) #18A only
(N_data, N_nodes, dummy1, dummy2) = np.shape(A_input)

if index < 3: 
    pred_chi1= np.squeeze(pred_chi1.detach().cpu().numpy())
    pred_chi2= np.squeeze(pred_chi2.detach().cpu().numpy())

grad_chi1 = np.transpose(np.asarray(np.gradient(pred_chi1,1/sgc,axis = (1,2))),(1,2,3,0)) # N_data, sgc, sgc, 2
grad_chi2 = np.transpose(np.asarray(np.gradient(pred_chi2,1/sgc,axis = (1,2))), (1,2,3,0)) # N_data, sgc, sgc, 2

grad_chi = np.concatenate((np.expand_dims(grad_chi1,axis = 4), np.expand_dims(grad_chi2,axis=4)), axis = 4) # N_data, sgc, sgc, 2,2
print(grad_chi.shape)
grad_chi = np.transpose(np.reshape(grad_chi, (N_data, N_nodes,2,2)),(0,1,3,2)) 

print("Gradients computed...")
integrand = np.asarray([[A_input[n,j, :,:] + np.matmul(A_input[n,j,:,:],grad_chi[n,j,:,:]) for j in range(N_nodes)] for n in range(N_data)])
pred_Abars = 1/N_nodes*np.sum(integrand, axis = 1) # N_data, 2,2

print("Integrated.")

err_norm_Abars = np.asarray([np.linalg.norm(true_Abars_test[n,:,:] - pred_Abars[n,:,:], ord = 'fro') for n in range(N_data)])
mat_norms_Ameans = np.asarray([np.linalg.norm(true_Ameans_test[n,:,:], ord = 'fro') for n in range(N_data)])
mat_norms_Aharms = np.asarray([np.linalg.norm(true_Aharms_test[n,:,:], ord = 'fro') for n in range(N_data)])

denoms = np.abs(mat_norms_Ameans - mat_norms_Aharms)
rel_errs = err_norm_Abars/denoms

with open(pred_chi1_path, 'rb') as handle:
    y_test,y_test_approx,train_err,test_err = pkl.load(handle)

# epochs = len(train_err)
y_test  = y_test.detach().cpu().numpy()
y_test_approx = np.squeeze(y_test_approx.detach().cpu().numpy())
N_test = y_test.shape[0]
h = 1/(np.squeeze(y_test[0]).shape[0])
grid_edge = y_test[0].shape[0]

def standard_rel_H1_err(true,pred):
    '''
    true and pred have dimensions grid_edge, grid_edge
    This is for a single sample only
    '''
    err =  (np.linalg.norm(pred - true)**2 + np.linalg.norm(np.gradient(pred-true,1/grid_edge))**2)**(1/2)
    true_H1_norm = (np.linalg.norm(true)**2 + np.linalg.norm(np.gradient(true,1/grid_edge))**2)**(1/2)
    return err/true_H1_norm

def rel_Wp_err(true,pred,p):
    diff_sols = pred-true
    diff_grad_loc_norms = np.linalg.norm(np.gradient(pred-true,1/grid_edge),axis = 0)
    err =  (np.linalg.norm(diff_sols.flatten(),p)**p + np.linalg.norm(diff_grad_loc_norms.flatten(),p)**p)**(1/p)
    grad_loc_norm = np.linalg.norm(np.gradient(true,1/grid_edge),axis = 0)
    true_H1_norm = (np.linalg.norm(true.flatten(),p)**p + np.linalg.norm(grad_loc_norm.flatten(),p)**p)**(1/p)
    return err/true_H1_norm

all_L2_errors = np.zeros(N_test)
all_grad_L2_errors = np.zeros(N_test)
all_H1_errors = np.zeros(N_test)
all_rel_Lp_errors = np.zeros(N_test)

p = 10
for i in range(N_test):
    all_H1_errors[i] = standard_rel_H1_err(y_test[i], y_test_approx[i])
    all_rel_Lp_errors[i] = rel_Wp_err(y_test[i], y_test_approx[i],p)

median_ind = np.argsort(all_H1_errors)[len(all_H1_errors)//2-1]
print(median_ind)
max_ind = np.argmax(all_H1_errors)
print("Mean Relative Error")
print(np.mean(all_H1_errors))
print("Median Relative Error")
print(np.median(all_H1_errors))
print("Mean Relative W1p Error")
print(np.mean(all_rel_Lp_errors))
print("Median Relative W1p Error")
print(np.median(all_rel_Lp_errors))

# mean_denom = np.mean(denoms)
# min_demon = np.min(denoms)
# med_demon = np.median(denoms)
# mean_error_Abars = np.mean(err_norm_Abars)
# mean_rel_err = np.mean(rel_errs)
med_rel_err_Abar = np.median(rel_errs)

print("Median Relative Abar Error")
print(med_rel_err_Abar)



# print(mean_rel_err)
# print(med_rel_err_Abar)
# print(med_demon)
# print(mean_error_Abars)
# print(min_demon)
# print(mean_denom)
# Scatter plot of frob norms
# plt.figure()
# plt.scatter(range(N_data), frob_norm_Abars, label = 'Abar')
# plt.scatter(range(N_data), frob_norms_Ameans, label = 'Amean')
# plt.scatter(range(N_data), frob_norms_Aharms, label = 'Aharms')
# plt.legend()
# plt.show()