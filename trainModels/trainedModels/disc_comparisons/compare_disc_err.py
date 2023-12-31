'''
Takes models trained on 128x128 grids and test them on grids of different sizes.
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import yaml


# Set font default
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
shapes = ['o','s','D','v']
fontsize = 30
'''

'''

# Load errors
smooth_err_path = 'smooth_res_'
vor_err_path = 'vor_res_'
gridsizes = [32,64,128,256,512]
samp_count = 5
all_smooth_errors = np.zeros((len(gridsizes),samp_count))
all_vor_errors = np.zeros((len(gridsizes),samp_count))

for gridsize in gridsizes:
    if gridsize == 128:
        for samp in range(samp_count):
            with open('../standard_models/smooth_model' + '_' + str(samp) + '_errors.yml') as file:
                smooth_err = yaml.load(file, Loader=yaml.FullLoader)
                H1_rel_mean_err = smooth_err['H1_rel_mean']
                all_smooth_errors[gridsizes.index(gridsize),samp] = H1_rel_mean_err
            with open('../standard_models/vor_model' + '_' + str(samp) + '_errors.yml') as file:
                vor_err = yaml.load(file, Loader=yaml.FullLoader)
                H1_rel_mean_err = vor_err['H1_rel_mean']
                all_vor_errors[gridsizes.index(gridsize),samp] = H1_rel_mean_err
    else:
        for samp in range(samp_count):
            with open(smooth_err_path + str(gridsize) + '_' + str(samp) + '.pkl_errors.yml') as file:
                smooth_err = yaml.load(file, Loader=yaml.FullLoader)
                H1_rel_mean_err = smooth_err['H1_rel_mean']
                all_smooth_errors[gridsizes.index(gridsize),samp] = H1_rel_mean_err
            with open(vor_err_path + str(gridsize) + '_' + str(samp) + '.pkl_errors.yml') as file:
                vor_err = yaml.load(file, Loader=yaml.FullLoader)
                H1_rel_mean_err = vor_err['H1_rel_mean']
                all_vor_errors[gridsizes.index(gridsize),samp] = H1_rel_mean_err

linewidth = 4
markersize = 15
# Plot errors
smooth_stds = np.std(all_smooth_errors,axis=1)
vor_stds = np.std(all_vor_errors,axis=1)
print(smooth_stds)
print(vor_stds)
# set fontsize
matplotlib.rcParams.update({'font.size': fontsize})
plt.figure(figsize = (10,10))
plt.plot(gridsizes,all_vor_errors.mean(axis=1),label='Voronoi',linewidth=linewidth,marker = shapes[0],color = CB_color_cycle[0],markersize=markersize)
plt.plot(gridsizes,all_smooth_errors.mean(axis=1),label='Smooth',linewidth=linewidth,marker = shapes[1],color = CB_color_cycle[1],markersize=markersize)
# Add error bars
plt.errorbar(gridsizes,all_vor_errors.mean(axis=1),yerr = 2*vor_stds,color = CB_color_cycle[0],linewidth=3)
plt.fill_between(gridsizes,all_vor_errors.mean(axis=1) - 2*vor_stds,all_vor_errors.mean(axis=1) + 2*vor_stds,color = CB_color_cycle[0],alpha = 0.2)
plt.errorbar(gridsizes,all_smooth_errors.mean(axis=1),yerr=2*smooth_stds,color=CB_color_cycle[1],linewidth=3)
plt.fill_between(gridsizes,all_smooth_errors.mean(axis=1) - 2*smooth_stds,all_smooth_errors.mean(axis=1) + 2*smooth_stds,color = CB_color_cycle[1],alpha = 0.2)
plt.xlabel('Grid Size',fontsize=fontsize)
plt.title('QQQ', color = 'white')
plt.ylabel(r'Mean Relative $H^1$ Error',fontsize = fontsize)
# set xticks
plt.yscale('log')
plt.xscale('log')
plt.xticks(gridsizes,fontsize=fontsize)
# Make xticks show on logscale
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# flip x axis
# set ylim
plt.ylim([3e-3,1e0])
# make more space on left
plt.subplots_adjust(left=0.2)
# add vertical dashed line at x = 128
plt.axvline(x=128,linestyle='--',color='k',label = 'Training Resolution')
plt.gca().invert_xaxis()
plt.legend(fontsize=fontsize,loc = 'upper left')
plt.savefig('grid_eval_errors.pdf')
