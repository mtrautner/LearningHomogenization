import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import pickle as pkl
import pdb
import numpy as np
import scipy.io
import sys
import torch.utils.data
from tqdm import tqdm
import yaml


from models.func_to_func2d import FNO2d
from util.utilities_module import H1Loss
from util.utilities_module import loss_report

def train_model(data_path, model_name, Ntotal = 10000,N_train = 9500, N_modes = 24, width = 48, epochs = 300, b_size = 50, lr = 0.001, USE_CUDA = True): 
    gc.collect()
    torch.cuda.empty_cache()

    model_info_path = 'Trained_Models/' + model_name + '_info.pkl'
    model_path = 'Trained_Models/' + model_name
    
    with open(data_path, 'rb') as handle:
        A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)



    sgc = x_ticks.shape[1] # Ex: 512


    (N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)
  
    train_size = N_data
    test_start = N_train
    test_end = Ntotal

    N_test = test_end - test_start

    data_output1 = np.transpose(chi1_true[:,:]) # N_data, N_nodes
    data_output2 = np.transpose(chi2_true[:,:]) # N_data, N_nodes

    data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
    data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components

    # Input shape (of x): (batch, channels_in, nx_in, ny_in)
    data_input = np.transpose(data_input, (0,3,1,2))

    # Output shape:       (batch, channels_out, nx_out, ny_out)
    data_output1 = np.reshape(data_output1,(N_data,sgc, sgc))
    data_output2 = np.reshape(data_output2,(N_data,sgc, sgc))

    #================== TRAINING ==================#
    # Concatenate data1_output1 and data_output2
    data_output = np.stack((data_output1,data_output2), axis=3)
    data_output = np.transpose(data_output, (0,3,1,2))

    # Split data into training and testing
    y_train = data_output[:train_size,:,:,:]
    y_test = data_output[test_start:test_end,:,:,:]

    x_train = data_input[:train_size,:,:,:]
    x_test = data_input[test_start:test_end,:,:,:]

    # Set loss function to be H1 loss
    loss_func = H1Loss().squared_H1

    # Convert to torch arrays
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Specify pointwise degrees of freedom
    d_in = 3 # A \in \R^{2 \times 2}_sym
    d_out = 2 # \chi \in \R^2

    s_outputspace = (sgc,sgc)

    # Initialize model
    net = FNO2d(modes1 = N_modes, modes2= N_modes, width = width, d_in = d_in, d_out = d_out, s_outputspace= s_outputspace)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    if USE_CUDA:
        net.cuda()
    
    # Wrap training data in loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)
    
    # Train net
    train_err = np.zeros((epochs,))
    test_err = np.zeros((epochs,))
    y_test_approx_all = torch.zeros(test_end-test_start,d_out,sgc, sgc)


    for ep in tqdm(range(epochs)):
        train_loss = 0.0
        test_loss  = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_approx = net(x).squeeze()
        
            # For forward net: 
            # Input shape (of x):     (batch, channels_in, nx_in, ny_in)
            # Output shape:           (batch, channels_out, nx_out, ny_out)
            
            # The input resolution is determined by x.shape[-2:]
            # The output resolution is determined by self.s_outputspace
            loss = loss_func(y_approx,y)
            loss.backward()
            train_loss = train_loss + loss.item()

            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            b=0
            for x,y in test_loader:
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()
                y_test_approx = net(x)
                t_loss = loss_func(y_test_approx,y)
                test_loss = test_loss + t_loss.item()
                if ep == epochs - 1:
                    y_test_approx_all[b*b_size:(b+1)*b_size,:,:,:] = torch.squeeze(y_test_approx).cpu()
                    b = b+1

        train_err[ep] = train_loss/len(train_loader)
        test_err[ep]  = test_loss/len(test_loader)
        print(ep, train_err[ep],test_err[ep])

    # Save model
    model_path = 'Trained_Models/' + model_name
    torch.save({'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_err,
            'test_loss_history': test_err,
            }, model_path)

    # Save model info
    model_info_path = 'Trained_Models/' + model_name + '_info.pkl'
    with open(model_info_path, 'wb') as handle:
        pkl.dump([Ntotal, N_train, N_modes, width, epochs, b_size, lr, x_ticks, y_ticks,data_path], handle)
    
    # Compute and save errors
    loss_report(y_test_approx_all, y_test, x_test, model_name)
    

if __name__ == "__main__":
    # Take in user arguments 
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_model(config['data_path'], config['model_name'], Ntotal = config['Ntotal'], N_train = config['N_train'], N_modes = config['N_modes'], width = config['width'], epochs = config['epochs'], b_size = config['batch_size'], lr = config['lr'], USE_CUDA = config['USE_CUDA'])
