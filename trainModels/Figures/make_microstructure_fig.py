import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
import pdb
import os
import torch
# add parent dir to path
import sys
sys.path.append('..')
from models.func_to_func2d import FNO2d
from util.utilities_module import format_data

# Set font default
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['mathtext.rm'] = 'stix'
matplotlib.rcParams['mathtext.it'] = 'stix'
matplotlib.rcParams['mathtext.bf'] = 'stix'

plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' 'serif' 'Times
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
'''

tickfontsize = 50
fontsize = 70

SMALL_SIZE = tickfontsize
MEDIUM_SIZE = tickfontsize
BIGGER_SIZE = fontsize

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Import tiny datasets
smooth_file = '/groups/astuart/mtrautne/learnHomData/data/smooth_data_tiny.pkl'
star_file = '/groups/astuart/mtrautne/learnHomData/data/star_data_tiny.pkl'
sq_file = '/groups/astuart/mtrautne/learnHomData/data/sq_data_tiny.pkl'
vor_file = '/groups/astuart/mtrautne/learnHomData/data/vor_data_tiny.pkl'
smooth_model = '/groups/astuart/mtrautne/learnHomData/trainedModels/standard_models/smooth_model_0'
star_model = '/groups/astuart/mtrautne/learnHomData/trainedModels/standard_models/star_model_0'
sq_model = '/groups/astuart/mtrautne/learnHomData/trainedModels/standard_models/sq_model_0'
vor_model = '/groups/astuart/mtrautne/learnHomData/trainedModels/standard_models/vor_model_0'


i = -1

def read_data(file,i):
    with open(file, 'rb') as f:
        A, chi_1, chi_2, x_ticks, y_ticks = pkl.load(f)
    true_A = A[i,:,:,:]
    true_A = np.expand_dims(true_A, axis = 0)
    true_chi = chi_1[:,i]
    true_chi = np.expand_dims(true_chi, axis = 1)
    return true_A, true_chi

def get_pred_sample(model_path,net, input,USE_CUDA = True):
    net.load_state_dict(torch.load(model_path)['model_state_dict'])
    net.eval()
    input = torch.tensor(input).float()
    if USE_CUDA:
        input = input.cuda()
        net.cuda()
    
    pred = net(input).detach().cpu().numpy()
    return pred

def compute_grad(x):
        x = torch.tensor(x).float()
        grid_edge = x.size()[-1]
        h = 1.0 / (grid_edge - 1.0)

        x_grad = torch.gradient(x, dim = (-2,-1), spacing = h)
        x_grad1 = x_grad[0].unsqueeze(1) # Component 1 of the gradient
        x_grad2 = x_grad[1].unsqueeze(1) # Component 2 of the gradient
        x_grad = torch.cat((x_grad1, x_grad2), 1)
        return x_grad.detach().numpy()


# sgc = 128
# h = 1/sgc
# net = FNO2d(modes1 = 18, modes2 = 18, width = 64, d_in =3, d_out = 2, s_outputspace=(128,128))

# smooth_A, smooth_chi_true = read_data(smooth_file,i)
# star_A, star_chi_true = read_data(star_file,i)
# sq_A, sq_chi_true = read_data(sq_file,i)
# vor_A, vor_chi_true = read_data(vor_file,i)


# smooth_A, smooth_chi_true = format_data(smooth_A, smooth_chi_true,smooth_chi_true, sgc)
# star_A, star_chi_true = format_data(star_A, star_chi_true,star_chi_true, sgc)
# sq_A, sq_chi_true= format_data(sq_A, sq_chi_true,sq_chi_true, sgc)
# vor_A, vor_chi_true = format_data(vor_A, vor_chi_true,vor_chi_true, sgc)

# all_A = (smooth_A, star_A, sq_A, vor_A)
# all_chi_true = (smooth_chi_true, star_chi_true, sq_chi_true, vor_chi_true)

# smooth_chi_pred = get_pred_sample(smooth_model,net,smooth_A)
# star_chi_pred = get_pred_sample(star_model,net,star_A)
# sq_chi_pred = get_pred_sample(sq_model,net,sq_A)
# vor_chi_pred = get_pred_sample(vor_model,net,vor_A)
# all_chi_pred = (smooth_chi_pred, star_chi_pred, sq_chi_pred, vor_chi_pred)

# smooth_grad_chi_true = compute_grad(smooth_chi_true)
# star_grad_chi_true = compute_grad(star_chi_true)
# sq_grad_chi_true = compute_grad(sq_chi_true)
# vor_grad_chi_true = compute_grad(vor_chi_true)
# all_grad_chi_true = (smooth_grad_chi_true, star_grad_chi_true, sq_grad_chi_true, vor_grad_chi_true)

# smooth_grad_chi_pred = compute_grad(smooth_chi_pred)
# star_grad_chi_pred = compute_grad(star_chi_pred)
# sq_grad_chi_pred = compute_grad(sq_chi_pred)
# vor_grad_chi_pred = compute_grad(vor_chi_pred)
# all_grad_chi_pred = (smooth_grad_chi_pred, star_grad_chi_pred, sq_grad_chi_pred, vor_grad_chi_pred)

# smooth_chi_error = np.abs(smooth_chi_pred - smooth_chi_true)
# star_chi_error = np.abs(star_chi_pred - star_chi_true)
# sq_chi_error = np.abs(sq_chi_pred - sq_chi_true)
# vor_chi_error = np.abs(vor_chi_pred - vor_chi_true)
# all_chi_error = (smooth_chi_error, star_chi_error, sq_chi_error, vor_chi_error)

# smooth_grad_chi_error = np.abs(smooth_grad_chi_pred - smooth_grad_chi_true)
# star_grad_chi_error = np.abs(star_grad_chi_pred - star_grad_chi_true)
# sq_grad_chi_error = np.abs(sq_grad_chi_pred - sq_grad_chi_true)
# vor_grad_chi_error = np.abs(vor_grad_chi_pred - vor_grad_chi_true)
# all_grad_chi_error = (smooth_grad_chi_error, star_grad_chi_error, sq_grad_chi_error, vor_grad_chi_error)

def get_A11(A):
    return np.squeeze(A[:,0,:,:])
def get_chi1(chi):
    return np.squeeze(chi[:,0,:,:])
def get_grad_chi1(grad_chi):
    chi1= np.squeeze(grad_chi[:,:,0,:,:])
    norm_grad = np.linalg.norm(chi1, axis = 0)
    return norm_grad

# # Save data to plot
# with open('microstructure_fig_data.pkl', 'wb') as f:
#     pkl.dump([all_A, all_chi_true, all_chi_pred, all_chi_error, all_grad_chi_true, all_grad_chi_pred, all_grad_chi_error], f)

with open('microstructure_fig_data.pkl', 'rb') as f:
    all_A, all_chi_true, all_chi_pred, all_chi_error, all_grad_chi_true, all_grad_chi_pred, all_grad_chi_error = pkl.load(f)

# Plot a grid where each row is a microstructure and has seven columns: true A, true chi, true grad chi, pred chi, pred grad chi, error chi, error grad chi
fig, ax = plt.subplots(4,7, figsize = (70,35))
cbarpad = 0.02
cmap = 'cividis' # 'viridis'
# Labels for subplots are 1a, 1b, etc 2a, 2b, etc
letters = ['1a', '1b', '1c', '1d', '1e', '1f', '1g', '2a', '2b', '2c', '2d', '2e', '2f', '2g', '3a', '3b', '3c', '3d', '3e', '3f', '3g', '4a', '4b', '4c', '4d', '4e', '4f', '4g']
for j in range(4):
    im0 = ax[j,0].imshow(get_A11(all_A[j]), cmap = cmap)
    # Add colorbar
    cbar0 = fig.colorbar(im0, ax = ax[j,0], fraction=0.046, pad=cbarpad)

    ymin = min(get_chi1(all_chi_true[j]).min(), get_chi1(all_chi_pred[j]).min())
    ymax = max(get_chi1(all_chi_true[j]).max(), get_chi1(all_chi_pred[j]).max())
    im1 = ax[j,1].imshow(get_chi1(all_chi_true[j]), cmap = cmap, vmin = ymin, vmax = ymax)
    im2 = ax[j,2].imshow(get_chi1(all_chi_pred[j]), cmap = cmap, vmin = ymin, vmax = ymax)
    # Add colorbars, make them the same limits
    
    cbar1 = fig.colorbar(im1, ax = ax[j,1], fraction=0.046, pad=cbarpad)
    cbar2 = fig.colorbar(im2, ax = ax[j,2], fraction=0.046, pad=cbarpad)
   
    
    

    im3 = ax[j,3].imshow(get_chi1(all_chi_error[j]), cmap = cmap)
    cbar3 = fig.colorbar(im3, ax = ax[j,3], fraction=0.046, pad=cbarpad)

    ymin = min(get_grad_chi1(all_grad_chi_true[j]).min(), get_grad_chi1(all_grad_chi_pred[j]).min())
    ymax = max(get_grad_chi1(all_grad_chi_true[j]).max(), get_grad_chi1(all_grad_chi_pred[j]).max())

    im4 = ax[j,4].imshow(get_grad_chi1(all_grad_chi_true[j]), cmap = cmap, vmin = ymin, vmax = ymax)
    im5 = ax[j,5].imshow(get_grad_chi1(all_grad_chi_pred[j]), cmap = cmap, vmin = ymin, vmax = ymax)
    # Add colorbars, make them the same limits
    cbar4 = fig.colorbar(im4, ax = ax[j,4], fraction=0.046, pad=cbarpad)
    cbar5 = fig.colorbar(im5, ax = ax[j,5], fraction=0.046, pad=cbarpad)

    im6 = ax[j,6].imshow(get_grad_chi1(all_grad_chi_error[j]), cmap = cmap)
    cbar6 = fig.colorbar(im6, ax = ax[j,6], fraction=0.046, pad=cbarpad)

    # Set colorbars tick font size
    cbar0.ax.tick_params(labelsize=tickfontsize)
    cbar1.ax.tick_params(labelsize=tickfontsize)
    cbar1.formatter.set_powerlimits((0, 0))
    cbar2.ax.tick_params(labelsize=tickfontsize)
    cbar2.formatter.set_powerlimits((0, 0))
    cbar3.formatter.set_powerlimits((0, 0))
    cbar3.ax.tick_params(labelsize=tickfontsize)
    cbar4.ax.tick_params(labelsize=tickfontsize)
    cbar4.formatter.set_powerlimits((0,0))
    cbar5.ax.tick_params(labelsize=tickfontsize)
    cbar5.formatter.set_powerlimits((0,0))
    cbar6.ax.tick_params(labelsize=tickfontsize)
    cbar6.formatter.set_powerlimits((0,0))


    for k in range(7):
        ax[j,k].set_xticks([])
        ax[j,k].set_yticks([])
        ax[j,k].text(-0.15, 1.02, '('+letters[7*j+k]+')', transform=ax[j,k].transAxes, 
                size=MEDIUM_SIZE, weight='bold')
# Label rows: Smooth, Star, Square, Voronoi
ax[0,0].set_ylabel('Smooth', fontsize = fontsize)
ax[1,0].set_ylabel('Star', fontsize = fontsize)
ax[2,0].set_ylabel('Square', fontsize = fontsize)
ax[3,0].set_ylabel('Voronoi', fontsize = fontsize)

# Label columns on top: A, chi, grad chi, pred chi, pred grad chi, error chi, error grad chi
ax[0,0].set_title(r'$A_{11}$', fontsize = fontsize)
ax[0,1].set_title(r'True $\chi_1$', fontsize = fontsize)
ax[0,2].set_title(r'FNO $\chi_1$', fontsize = fontsize)
ax[0,3].set_title(r'Error $\chi_1$', fontsize = fontsize)
ax[0,4].set_title(r'True $\nabla \chi_1$', fontsize = fontsize)
ax[0,5].set_title(r'FNO $\nabla \chi_1$', fontsize = fontsize)
ax[0,6].set_title(r'Error $\nabla \chi_1$', fontsize = fontsize)



plt.tight_layout()
# Reduce horizontal space between subplots
plt.subplots_adjust(wspace=0.07, hspace=0.095)
# make left border zero
plt.subplots_adjust(left=0.006,bottom = 0.005,right = 0.99, top = 0.97)
# pdf quality high
plt.savefig('microstructure_fig.pdf')


     



