import os.path
import pickle as pkl
import numpy as np
import sys
import scipy.io
import pdb

'''
This script converts the .mat data to .pkl data
'''

def convert_data(mat_data_dir,pkl_dir,data_name,N_data,small_dataset = False):

    list_string = list(map(str, range(N_data+1)))
    indices = list_string[1:]

    info_path = os.path.join(mat_data_dir,'data_info.mat')
    info_in_mat = scipy.io.loadmat(info_path)
    x_ticks = info_in_mat['x_grid']
    y_ticks = info_in_mat['y_grid']
    grid_edge = x_ticks.shape[1]

    A_input = np.zeros((N_data,grid_edge**2,2,2))
    chi1_data = np.zeros((grid_edge**2,N_data))
    chi2_data = np.zeros((grid_edge**2, N_data))

    for i, i_str in enumerate(indices):
        data_file = os.path.join(mat_data_dir,'data_' + i_str + ".mat")
        data_mat = scipy.io.loadmat(data_file)
        A_mat = data_mat['Avals']
        chi1_interp = data_mat['chi1_interp']
        chi2_interp = data_mat['chi2_interp']

        A_input[i,:,:,:] = np.transpose(A_mat,(2,0,1))
        chi1_data[:,i] = np.squeeze(chi1_interp)
        chi2_data[:,i] = np.squeeze(chi2_interp)

    full_pkl_path = os.path.join(pkl_dir,data_name + '.pkl')
    with open(full_pkl_path, 'wb') as handle:
        pkl.dump([A_input, chi1_data, chi2_data, x_ticks, y_ticks], handle, protocol=pkl.HIGHEST_PROTOCOL)

    if small_dataset == True:
        tiny_pkl_path = os.path.join(pkl_dir,data_name+"_tiny.pkl")
        with open(tiny_pkl_path, 'wb') as handle:
            pkl.dump([A_input[:20,:,:,:], chi1_data[:,:20], chi2_data[:,:20], x_ticks, y_ticks], handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Take in user arguments
    mat_data_dir = sys.argv[1]
    pkl_dir = sys.argv[2]
    data_name = sys.argv[3]
    N_data = int(sys.argv[4])

    small_dataset = True 

    convert_data(mat_data_dir,pkl_dir,data_name,N_data,small_dataset)