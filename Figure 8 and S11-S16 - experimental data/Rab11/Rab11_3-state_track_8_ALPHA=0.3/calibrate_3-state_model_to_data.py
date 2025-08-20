# importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy
import random
import pandas as pd
import math
import scipy
from scipy.stats import norm, expon
from scipy.linalg import null_space
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid

import os
cwd = os.getcwd()

import sys
sys.path.append('../../')

from functions_MCMC_alpha import *

init_seed = 8 #choose accordingly to dataset used?
type_dataset = 'Rab11'

alpha = 0.3

data_import = pd.read_csv(cwd+"/"+type_dataset+"_delta_y"+str(init_seed)+".csv", delimiter=',', index_col=0)

# define variables
data_seed = 1223
burnin = 10000 #10k
n_after_burnin = 10000 #10k
delta_t = 0.3

n_chains = 4
n_sim = 1

n_param = 9

parameter_names = ['v1', 'v2', 'log(lambda1)', 'log(lambda2)',
                   'log(lambda3)', 'p12', 'p21', 'p31', 'sigma']
parameter_names_tex = [r'$v_1$', r'$v_2$', r'log$(\lambda_1)$',
                       r'log$(\lambda_2)$', r'log$(\lambda_3)$', r'$p_{12}$', r'$p_{21}$', r'$p_{31}$',
                       r'$\sigma$']

#choose initial covariance matrix for resampling
init_cov_matrix = np.array([np.diag(np.array([0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
                            for _ in range(n_chains)])


correlated = True
up_to_switches = 1
track = True

plots = False
save = True
all_plots = False
plot_posteriors_grid = False
plot_fit = False
plot_fit_median = False


#THIS STILL NEEDS TO BE MODIFIED FOR SEVERAL CASES
def get_parameters(theta):
    """Obtaining parameters from theta"""
    V = np.array(list(theta[0:2])+[0.0])
    Lambda = np.exp(theta[2:5])
    P = np.array([[0.0, theta[-4], 1.0-theta[-4]],
                  [theta[-3], 0.0, 1.0-theta[-3]],
                  [theta[-2], 1-theta[-2], 0.0]])
    sigma = 1.0*theta[-1]
    #print(V, Lambda, P, sigma)
    return V, Lambda, P, sigma

def q(theta, cov_matrix=init_cov_matrix):
    """q samples a new theta_star given theta"""
    theta_star = np.random.multivariate_normal(theta, cov_matrix)
    while (theta_star[0]<0 or theta_star[0]>4000 or 
           theta_star[1]>0 or theta_star[1]<-4000 or 
           np.any(theta_star[2:5]<-4) or np.any(theta_star[2:5]>4) or
           np.any(theta_star[5:8]<0) or np.any(theta_star[5:8]>1) or
           theta_star[-1]<0 or np.any(theta_star[-1]>100)):
        theta_star = np.random.multivariate_normal(theta, cov_matrix)
    return theta_star

def theta_init_uniform(seed):
    np.random.seed(seed)
    rand_vec = np.random.uniform(size=n_param)
    theta = ((np.array([4000, -4000, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 100])*rand_vec)
              +np.array([0.0, 0.0, -4.0, -4.0, -4.0, 0.0, 0.0, 0.0, 0.0]))
    #print('theta_init =', get_parameters(theta))
    return theta

delta_Y = np.array(data_import).flatten()

N = delta_Y.shape[0]
T = N*0.3

rebuild_y = np.zeros(N+1)
for i in range(1, N+1):
    rebuild_y[i] += rebuild_y[i-1] + delta_Y[i-1]
plt.figure(figsize=(4,8))
plt.plot(rebuild_y,delta_t*np.arange(0,N+1), '.', color='#d46a7e', alpha=1)
plt.xlabel(r'measured locations', fontsize=14)
plt.ylabel(r'time', fontsize=14)
plt.gca().invert_yaxis()
#plt.legend()
plt.savefig('y_plot_track.png', format="png", dpi=1200, bbox_inches="tight")


all_theta, all_log_pi, _, _, dY = distinct_track_runs_MCMC(theta_true = None,
                                 get_parameters = get_parameters,
                                 parameter_names = parameter_names,
                                 parameter_names_tex = parameter_names_tex,
                                 burnin = burnin, n_after_burnin = n_after_burnin,
                                 delta_t = delta_t, T = T,
                                 n_chains = n_chains, n_sim = n_sim,
                                 plots = plots, save = save, all_plots = all_plots,
                                 plot_posteriors_grid = plot_posteriors_grid,
                                 plot_fit = plot_fit,
                                 plot_fit_median = plot_fit_median, track = track,
                                 up_to_switches = up_to_switches,
                                 init_cov_matrix = init_cov_matrix, q = q,
                                 delta_Y = delta_Y,
                                 theta_init = None,
                                 theta_init_distribution = theta_init_uniform,
                                 correlated = correlated, show_time = True, init_seed = init_seed, alpha=alpha)
#init seed to remember track number

flat_all_theta = np.array([list(all_theta[i,:,:,:].flatten()) for i in range(n_param)])

log_pi = all_log_pi.flatten()
theta = flat_all_theta
log_pi = log_pi
j1 = 0
#plots
fig, ax = plt.subplots(n_param, n_param, figsize=(n_param*2+5,n_param*2+2))
plt.subplots_adjust(wspace=0.35, hspace=0.2)
for i in range(n_param):
    ax[i,i].set_ylabel(parameter_names_tex[i], fontsize=14)
    ax[i,i].set_xlabel(parameter_names_tex[i], fontsize=14)

for i in range(n_param):
    ax[i,i].hist(theta[i,:], bins=25,
                 label=r'$\theta$ posterior', color='lightsteelblue')
    best_theta_comp_i = theta[i, np.nanargmax(log_pi)]
    ax[i,i].axvline(best_theta_comp_i, linestyle='-.', color='blue',
                    label=r'$\hat\theta$')
for i in range(n_param):
    for j in range(0,i):
        ax[i,j].axis('off')
    for j in range(i+1,n_param):
        #plt.title('Checking correlations')
        ax[i,j].hist2d(theta[j,:], theta[i,:], density=True, bins=30, alpha=0.7,
                       norm=colors.LogNorm(), cmap='Blues')
        ax[i,j].axvline(theta[j, np.nanargmax(log_pi)], linestyle='-.', color='blue',
                        label=r'$\hat\theta$')
        ax[i,j].axhline(theta[i, np.nanargmax(log_pi)], linestyle='-.', color='blue',
                        label=r'$\hat\theta$')
plt.subplots_adjust(right=0.85)
plt.legend(bbox_to_anchor=(-0.50, 0.7), fontsize=14, loc='upper right', borderaxespad=0)
plt.savefig("best_parameters_posteriors.png", format="png", dpi=1200, bbox_inches="tight")


#plot data vs best fit
eval_points = np.linspace(np.min(delta_Y), np.max(delta_Y), 100)

nbins = 10

plt.figure(figsize=(5,4))
plt.hist(delta_Y, bins=nbins, density=True, color='#d46a7e', alpha=0.5,
         label=r'$P^\mathcal{D}(\Delta y | \theta_t)$')
    
best_theta_comp = theta[:, np.nanargmax(log_pi)]
approx_pdf_comp = approx_pdf_theta(best_theta_comp, get_parameters, delta_t, eval_points,
                                   up_to_switches = up_to_switches, track = False)

plt.plot(eval_points, approx_pdf_comp, '-.', color='blue', label=str(r'$P_1(\Delta y |\hat\theta)$'))
plt.legend(fontsize=14)
plt.xlabel(r'$\Delta y$', fontsize=14)
plt.ylabel(r'$\mathbb{P}(\Delta y | \theta)$', fontsize=14)
plt.savefig("best_fit_compared_to_data.png", format="png", dpi=1200, bbox_inches="tight")


#define points for which we need the PDF approximation  
minmax = np.linspace(np.min(delta_Y), np.max(delta_Y), 400)
X, Y = np.meshgrid(minmax, minmax)

theta_best = theta[:, np.nanargmax(log_pi)]
V_best, Lambda_best, P_best, sigma_best = get_parameters(theta_best)

#computing the approximate PDF P1(Delta y_1, Delta y_2)
approx_pdf_track_res_best = np.array([[np.prod(approx_pdf_track_up_to_1_switch(V_best,
                                                                                 Lambda_best,
                                                                          P_best, sigma_best,
                                                                          delta_t, 
                                                           np.array([X[i,j], Y[i,j]])))
                                  for i in range(X.shape[0])]
                                 for j in range(X.shape[0])])


fig = plt.figure(figsize=(6,3))
ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15)
ax[0].hist2d(delta_Y[1:], delta_Y[0:-1], density=True, bins=10,
             norm=colors.LogNorm(vmin=10**(-8), vmax=10**(-4.5)), cmap='PuBu')

ax[0].set_title(r'$P^\mathcal{D}(\Delta y_1, \Delta y_2)$', fontsize=14)
ax[0].set_xlabel(r'$\Delta y_2$', fontsize=12)
ax[0].set_ylabel(r'$\Delta y_1$', fontsize=12)

cmap = mpl.colormaps['PuBu']
# Take colors at regular intervals spanning the colormap.
colors_vec = cmap(np.linspace(0, 1, 400))
ax[0].set_facecolor(colors_vec[0])

pcm = ax[1].pcolor(X, Y, approx_pdf_track_res_best.T,
                   norm=colors.LogNorm(vmin=10**(-8), vmax=10**(-4.5)), cmap='PuBu')
ax[1].set_title(r'$P_1(\Delta y_1, \Delta y_2 | \hat\theta)$', fontsize=14)
ax[1].set_xlabel(r'$\Delta y_2$', fontsize=12)


ax[1].cax.colorbar(pcm)
ax[1].cax.toggle_label(True)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("P_Deltay_1,Deltay_2_best.png",
            format="png", bbox_inches="tight", dpi=1200)