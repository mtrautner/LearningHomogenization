'''
Takes models trained at 128x128 discretization and evaluates them on different discretizations.
Saves the errors in a yaml file.
'''
import numpy as np
import os
import torch
import torch.utils.data
# import path of models dir
import sys
import gc
sys.path.append('../..')
from models.func_to_func2d import FNO2d
from util.utilities_module import *
import pickle as pkl
import matplotlib.pyplot as plt
sys.path.append('../size_compare')

USE_CUDA = torch.cuda.is_available()
print("USE_CUDA = " + str(USE_CUDA))
model_paths = '/groups/astuart/mtrautne/learnHomData/trainedModels/standard_models/'
smooth_modes = 18
vor_modes = 18
samp_count = 5
model_size = 1152
cut_data = 50 # Don't use full data sets
d_in = 3
d_out = 2
b_size = 10
smooth_models_path = model_paths + 'smooth_model_'
vor_models_path = model_paths + 'vor_model_'
err_save_path = 'grid_eval_errors_'
gridsizes = [32]

def plot_sample(x_in,y_out,y_pred,save_name):
    '''
    x_in: (1, channels_in, nx_in, ny_in)
    y_out: (1, channels_out, nx_out, ny_out)
    y_pred: (1, channels_out, nx_out, ny_out)
    '''
    x_in = torch.squeeze(x_in)
    y_out = torch.squeeze(y_out)
    y_pred = torch.squeeze(y_pred)
    plt.figure(figsize = (20,10))
    plt.subplot(241)
    plt.imshow(x_in[0,:,:])
    plt.colorbar()
    plt.title('A11')
    plt.subplot(242)
    plt.imshow(y_out[0,:,:])
    plt.colorbar()
    plt.title('chi1_true')
    plt.subplot(243)
    plt.imshow(y_pred[0,:,:])
    plt.colorbar()
    plt.title('chi1_pred')
    plt.subplot(244)
    plt.imshow(torch.abs(y_pred[0,:,:] - y_out[0,:,:]))
    plt.colorbar()
    plt.title('Abs Error')
    plt.subplot(245)
    plt.imshow(x_in[1,:,:])
    plt.colorbar()
    plt.title('A12')
    plt.subplot(246)
    plt.imshow(y_out[1,:,:])
    plt.colorbar()
    plt.title('chi2_true')
    plt.subplot(247)
    plt.imshow(y_pred[1,:,:])
    plt.colorbar()
    plt.title('chi2_pred')
    plt.subplot(248)
    plt.imshow(x_in[2,:,:])
    plt.colorbar()
    plt.title('A22')
    plt.savefig(save_name)

def format_data(A_input, chi1_true, chi2_true, gridsize):
    # Reshape data
    sgc = gridsize
    (N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)
    data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
    data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes
    data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
    data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components
    # Input shape (of x): (batch, channels_in, nx_in, ny_in)
    data_input = np.transpose(data_input, (0,3,1,2))

    #Output shape:      (batch, channels_out, nx_out, ny_out)
    data_output1 = np.reshape(data_output1, (N_data,sgc, sgc))
    data_output2 = np.reshape(data_output2, (N_data,sgc, sgc))
    # concatenate
    data_output = np.stack((data_output1,data_output2),axis = 3)
    data_output = np.transpose(data_output, (0,3,1,2))

    return data_input, data_output

def eval_net(net,d_out,gridsize,test_loader,b_size,USE_CUDA = USE_CUDA,N_data = 500):
    if USE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()
    net.cuda()
    y_test_approx_all = torch.zeros(N_data,d_out,gridsize,gridsize)
    b = 0
    with torch.no_grad():
        for x,y in test_loader:
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
            y_pred = net(x)
            y_test_approx_all[b*b_size:(b+1)*b_size,:,:,:] = torch.squeeze(y_pred).cpu()
            b += 1
    return y_test_approx_all

if USE_CUDA:
    gc.collect()
    torch.cuda.empty_cache()

for sample in range(samp_count):
    print(sample)
    for gridsize in gridsizes:
        print(gridsize)
        # load model
        smooth_net = FNO2d(modes1 = smooth_modes, modes2=smooth_modes,width = model_size//smooth_modes,d_in = d_in, d_out = d_out,s_outputspace=(gridsize,gridsize))
        vor_net = FNO2d(modes1 = vor_modes, modes2=vor_modes,width = model_size//vor_modes,d_in = d_in, d_out = d_out,s_outputspace=(gridsize,gridsize))
        # load smooth net
        smooth_model_file = smooth_models_path + str(sample)
        smooth_net.load_state_dict(torch.load(smooth_model_file)['model_state_dict'])
        smooth_net.eval()
        # load vor net
        vor_model_file = vor_models_path + str(sample)
        vor_net.load_state_dict(torch.load(vor_model_file)['model_state_dict'])
        vor_net.eval()

        # Load data
        data_path = '/groups/astuart/mtrautne/learnHomData/data/'
        smooth_data_file = data_path +'smooth_res_data_' + str(gridsize) + '.pkl'
        vor_data_file = data_path +'vor_res_data_' + str(gridsize) + '.pkl'
        with open(smooth_data_file, 'rb') as handle:
            A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)
        smooth_data_input, smooth_data_output = format_data(A_input, chi1_true, chi2_true, gridsize)
        with open(vor_data_file, 'rb') as handle:
            A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)
        vor_data_input, vor_data_output = format_data(A_input, chi1_true, chi2_true, gridsize)

        # Convert to torch arrays
        smooth_data_input = torch.from_numpy(smooth_data_input).float()
        smooth_data_output = torch.from_numpy(smooth_data_output).float()
        vor_data_input = torch.from_numpy(vor_data_input).float()
        vor_data_output = torch.from_numpy(vor_data_output).float()

        # Cut data to only cut_data samples
        smooth_data_input = smooth_data_input[:cut_data,:,:,:]
        smooth_data_output = smooth_data_output[:cut_data,:,:,:]
        vor_data_input = vor_data_input[:cut_data,:,:,:]
        vor_data_output = vor_data_output[:cut_data,:,:,:]

        smooth_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(smooth_data_input, smooth_data_output), batch_size=b_size, shuffle=False)
        vor_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vor_data_input, vor_data_output), batch_size=b_size, shuffle=False)
        
        # Eval smooth first
        N_data = vor_data_input.shape[0]
        net = smooth_net
        if USE_CUDA:
            net.cuda()
        net.eval()
        y_test_approx_all = eval_net(net,d_out,gridsize,smooth_test_loader,b_size,USE_CUDA = USE_CUDA, N_data = N_data)
        err_path = 'smooth_res_' + str(gridsize) +'_' + str(sample) + '.pkl'
        loss_report(y_test_approx_all, smooth_data_output, smooth_data_input, err_path)
        # plot_sample(smooth_data_input[0,:,:,:],smooth_data_output[0,:,:,:],y_test_approx_all[0,:,:,:],'smooth_res_' + str(gridsize) +'_' + str(sample) + '.pdf')
        # Eval vor
        net = vor_net
        net.eval()
        if USE_CUDA:
            net.cuda()
        
        y_test_approx_all = eval_net(net,d_out,gridsize,vor_test_loader,b_size,USE_CUDA = USE_CUDA, N_data = N_data)
        err_path = 'vor_res_' + str(gridsize) +'_' + str(sample) + '.pkl'
        loss_report(y_test_approx_all, vor_data_output, vor_data_input, err_path)
        # plot_sample(vor_data_input[0,:,:,:],vor_data_output[0,:,:,:],y_test_approx_all[0,:,:,:],'vor_res_' + str(gridsize) +'_' + str(sample) + '.pdf')

        




