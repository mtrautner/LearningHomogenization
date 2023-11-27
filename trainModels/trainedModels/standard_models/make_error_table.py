'''
Takes the errors from each set of models and makes a table of their means and stds.
'''
import numpy as np
import os
import prettytable
import yaml

smooth_err_path = 'smooth_model_'
star_err_path = 'star_model_'
sq_err_path = 'sq_model_'
vor_err_path = 'vor_model_'
vor_fg_err_path = 'vor_model_fixed_geom_'
samp_count = 5

err_types = ['H1_rel_mean','W1_10_rel_mean','Abar_rel_error2_med']
err_names = ['H1 rel mean','W1 10 rel mean','Abar rel error med']

smooth_errs = np.zeros((len(err_types),samp_count))
star_errs = np.zeros((len(err_types),samp_count))
sq_errs = np.zeros((len(err_types),samp_count))
vor_errs = np.zeros((len(err_types),samp_count))
vor_fg_errs = np.zeros((len(err_types),samp_count))

for samp in range(samp_count):
    with open(smooth_err_path + str(samp) + '_errors.yml') as file:
        smooth_err = yaml.load(file, Loader=yaml.FullLoader)
        for i in range(len(err_types)):
            err_type = err_types[i]
            smooth_errs[i,samp] = smooth_err[err_type]
    with open(star_err_path + str(samp) + '_errors.yml') as file:
        star_err = yaml.load(file, Loader=yaml.FullLoader)
        for i in range(len(err_types)):
            err_type = err_types[i]
            star_errs[i,samp] = star_err[err_type]
    with open(sq_err_path + str(samp) + '_errors.yml') as file:
        sq_err = yaml.load(file, Loader=yaml.FullLoader)
        for i in range(len(err_types)):
            err_type = err_types[i]
            sq_errs[i,samp] = sq_err[err_type]
    with open(vor_err_path + str(samp) + '_errors.yml') as file:
        vor_err = yaml.load(file, Loader=yaml.FullLoader)
        for i in range(len(err_types)):
            err_type = err_types[i]
            vor_errs[i,samp] = vor_err[err_type]
    with open(vor_fg_err_path + str(samp) + '_errors.yml') as file:
        vor_fg_err = yaml.load(file, Loader=yaml.FullLoader)
        for i in range(len(err_types)):
            err_type = err_types[i]
            vor_fg_errs[i,samp] = vor_fg_err[err_type]

smooth_errs_mean = np.mean(smooth_errs,axis=1)
smooth_errs_std = np.std(smooth_errs,axis=1)
star_errs_mean = np.mean(star_errs,axis=1)
star_errs_std = np.std(star_errs,axis=1)
sq_errs_mean = np.mean(sq_errs,axis=1)
sq_errs_std = np.std(sq_errs,axis=1)
vor_errs_mean = np.mean(vor_errs,axis=1)
vor_errs_std = np.std(vor_errs,axis=1)
vor_fg_errs_mean = np.mean(vor_fg_errs,axis=1)
vor_fg_errs_std = np.std(vor_fg_errs,axis=1)

table = prettytable.PrettyTable()
# rows are Smooth, Star, Square, Voronoi, and Voronoi Fixed Geometry
table.add_row(["Error Type"] + err_names)
table.add_row(["Smooth"] + ["%.5f ± %.5f" % (smooth_errs_mean[i],smooth_errs_std[i]) for i in range(len(err_types))])
table.add_row(["Star"] + ["%.5f ± %.5f" % (star_errs_mean[i],star_errs_std[i]) for i in range(len(err_types))])
table.add_row(["Square"] + ["%.5f ± %.5f" % (sq_errs_mean[i],sq_errs_std[i]) for i in range(len(err_types))])
table.add_row(["Voronoi"] + ["%.5f ± %.5f" % (vor_errs_mean[i],vor_errs_std[i]) for i in range(len(err_types))])
table.add_row(["Voronoi FG"] + ["%.5f ± %.5f" % (vor_fg_errs_mean[i],vor_fg_errs_std[i]) for i in range(len(err_types))])
# table.add_row("Smooth",["%.5f ± %.5f" % (smooth_errs_mean[i],smooth_errs_std[i]) for i in range(len(err_types))])
# table.add_row("Star",["%.5f ± %.5f" % (star_errs_mean[i],star_errs_std[i]) for i in range(len(err_types))])
# table.add_row("Square",["%.5f ± %.5f" % (sq_errs_mean[i],sq_errs_std[i]) for i in range(len(err_types))])
# table.add_row("Voronoi",["%.5f ± %.5f" % (vor_errs_mean[i],vor_errs_std[i]) for i in range(len(err_types))])
# table.add_row("Voronoi FG",["%.5f ± %.5f" % (vor_fg_errs_mean[i],vor_fg_errs_std[i]) for i in range(len(err_types))])
print(table)
