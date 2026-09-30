import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datetime import datetime
import numpy as np
import sys
from model import Model, ModelParams, SimParams
import glob
import pickle
import pandas as pd
import re
import shutil
from scipy.optimize import rosen, differential_evolution
from scipy.stats import ks_2samp
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from sarcasm import Motion, Utils
from model import *

# Show detailed simulation logs
verbose = False

# result folder
result_folder = './diff_evo_distr/'
os.makedirs(result_folder, exist_ok=True)

# Create base simulation parameters using dataclass
sim_params = SimParams(
    y0=None,
    t1=10,
    dt=0.002,
    a_mode='sine',
    a=[1, 0.0, 1],
    p=0,
)

# load experiment data
data_folder = '/Users/daniel/Documents/model_paper/2023_results/experimental_data/'

# 5kPa
file_5kPa = data_folder + '5kPa/Series017.tif'
roi_5kPa = '202_46_873_13_12_roi.json'

# 20kPa
file_20kPa = data_folder + '20kPa/Series079.tif'
roi_20kPa = '168_58_923_54_12_roi.json'

# 85kPa
file_85kPa = data_folder + '85kPa/Series052.tif'
roi_85kPa = '200_22_873_86_12_roi.json'

# Optimization bounds
bounds = [(0.001, 0.05),  # mu
          (0.5, 5),         # eta0
          (1, 50),          # eta1
          (0.5, 50),         # k_s0
          (0.5, 50),         # k_s1
          (0.01, 0.5),      # u_s
          (-5, 5),          # f_s0
          (-5, 5),          # f_s1
          (-5, 5),          # f_s2
          (-5, 5),          # f_s3
        ]

tlim = (50, -1)
sampling = 8
n_intervals = 3
k_l = 2
test = 'KS'

# Initialize Motion objects globally
sarc_5kPa = Motion(file_5kPa, roi_5kPa)
sarc_20kPa = Motion(file_20kPa, roi_20kPa)
sarc_85kPa = Motion(file_85kPa, roi_85kPa)

# Create a global dictionary to store pre-processed experimental data
global_exp_data = {}


def preprocess_exp_data(sarc_exp, tlim, n_intervals):
    """
    Preprocess experimental data for comparison with model.

    Parameters
    ----------
    sarc_exp : Motion
        Experimental sarcomere motion data.
    tlim : tuple
        Time limits (start, end) for data extraction.
    n_intervals : int
        Number of intervals to split contraction cycle into.

    Returns
    -------
    dict
        Dictionary containing processed experimental data arrays.
    """
    length_exp = sarc_exp.loi_data['delta_slen'][:, tlim[0]: tlim[1]]
    vel_exp = sarc_exp.loi_data['vel'][:, tlim[0]: tlim[1]]
    vdot_exp = Utils.custom_diff(vel_exp.T, dt=sarc_exp.metadata.frametime).T
    length_avg_exp = np.nanmean(length_exp, axis=0)
    vel_avg_exp = np.nanmean(vel_exp, axis=0)
    contr_exp = sarc_exp.loi_data['contr'][tlim[0]: tlim[1]]
    contr_intervals_exp = split_contraction_intervals(contr_exp, n_intervals=n_intervals)

    length_exp_intervals = []
    vel_exp_intervals = []
    vdot_exp_intervals = []
    for i in range(n_intervals + 1):
        length_exp_i = remove_nans(length_exp[:, contr_intervals_exp == i].flatten())
        length_exp_intervals.append(length_exp_i)
        vel_exp_i = remove_nans(vel_exp[:, contr_intervals_exp == i].flatten())
        vel_exp_intervals.append(vel_exp_i)
        vdot_exp_i = remove_nans(vdot_exp[:, contr_intervals_exp == i].flatten())
        vdot_exp_intervals.append(vdot_exp_i)

    length_avg_exp_contr = remove_nans(length_avg_exp[contr_exp == 1].flatten())
    length_avg_exp_quiet = remove_nans(length_avg_exp[contr_exp == 0].flatten())
    vel_avg_exp_contr = remove_nans(vel_avg_exp[contr_exp == 1].flatten())
    vel_avg_exp_quiet = remove_nans(vel_avg_exp[contr_exp == 0].flatten())

    return {
        'length_exp_intervals': length_exp_intervals,
        'vel_exp_intervals': vel_exp_intervals,
        'vdot_exp_intervals': vdot_exp_intervals,
        'length_avg_exp_contr': length_avg_exp_contr,
        'length_avg_exp_quiet': length_avg_exp_quiet,
        'vel_avg_exp_contr': vel_avg_exp_contr,
        'vel_avg_exp_quiet': vel_avg_exp_quiet
    }


# Preprocess experimental data
global_exp_data = preprocess_exp_data(sarc_20kPa, tlim, n_intervals)


def sim(paras):
    """
    Simulate sarcomere dynamics and calculate loss function.

    Parameters
    ----------
    paras : array-like
        Array of parameter values to optimize.

    Returns
    -------
    float
        Loss value (mean KS statistic across distributions).
    """
    mu, eta0, eta1, k_s0, k_s1, u_s, f_s0, f_s1, f_s2, f_s3 = paras

    # Create model parameters using dataclass
    model_params = ModelParams(
        N=20,
        mu=mu,
        eta=[eta0, eta1],
        k_l=k_l,
        k_s=[k_s0, k_s1],
        u_s=u_s,
        o_s=None,
        f_s=[f_s0, f_s1, f_s2, f_s3],
        h=0.0,
    )

    # Create and run model
    model = Model(model_params, sim_params, autosave=False, verbose=True)
    model.integrate()

    # # Check for NaN in results
    if np.any(np.isnan(model.data['length'])) or np.any(np.isnan(model.data['vel'])):
        if verbose:
            print("NaN values found in model data, returning high loss.")
        return 1e6  # Penalize heavily

    model.analyze()

    # Model samples
    length_model = model.data['length'][:, ::sampling]
    vel_model = model.data['vel'][:, ::sampling]
    if length_model.size == 0 or vel_model.size == 0:
        if verbose:
            print("Empty model data arrays, returning high loss.")
        return 1e6
    vdot_model = Utils.custom_diff(vel_model.T, dt=model.sim_params.dt * sampling).T
    length_avg_model = np.nanmean(length_model, axis=0)
    vel_avg_model = np.nanmean(vel_model, axis=0)
    contr_model = model.data['contr'][::sampling]
    contr_intervals_model = split_contraction_intervals(contr_model, n_intervals=n_intervals)

    # Split into contraction intervals
    length_model_intervals = []
    vel_model_intervals = []
    vdot_model_intervals = []
    for i in range(n_intervals + 1):
        length_model_i = remove_nans(length_model[:, contr_intervals_model == i].flatten())
        length_model_intervals.append(length_model_i)
        vel_model_i = remove_nans(vel_model[:, contr_intervals_model == i].flatten())
        vel_model_intervals.append(vel_model_i)
        vdot_model_i = remove_nans(vdot_model[:, contr_intervals_model == i].flatten())
        vdot_model_intervals.append(vdot_model_i)

    length_avg_model_contr = remove_nans(length_avg_model[contr_model == 1].flatten())
    length_avg_model_quiet = remove_nans(length_avg_model[contr_model == 0].flatten())
    vel_avg_model_contr = remove_nans(vel_avg_model[contr_model == 1].flatten())
    vel_avg_model_quiet = remove_nans(vel_avg_model[contr_model == 0].flatten())

    # Calculate Kolmogorov-Smirnov statistics
    ks_length = np.mean([compare_distributions(d_exp, d_model, test=test) for d_exp, d_model in
                         zip(global_exp_data['length_exp_intervals'], length_model_intervals)])
    ks_vel = np.mean([compare_distributions(d_exp, d_model, test=test) for d_exp, d_model in
                      zip(global_exp_data['vel_exp_intervals'], vel_model_intervals)])
    ks_vdot = np.mean([compare_distributions(d_exp, d_model, test=test) for d_exp, d_model in
                       zip(global_exp_data['vdot_exp_intervals'], vdot_model_intervals)])
    ks_length_avg = np.mean(
        [compare_distributions(global_exp_data['length_avg_exp_contr'], length_avg_model_contr, test=test) +
         compare_distributions(global_exp_data['length_avg_exp_quiet'], length_avg_model_quiet, test=test)])
    ks_vel_avg = np.mean([compare_distributions(global_exp_data['vel_avg_exp_contr'], vel_avg_model_contr, test=test) +
                          compare_distributions(global_exp_data['vel_avg_exp_quiet'], vel_avg_model_quiet, test=test)])

    loss = (ks_length + ks_vel + ks_vdot) / 3

    if np.isnan(loss) or np.isinf(loss):
        if verbose:
            print("Loss is NaN or Inf, returning high loss.")
        return 1e6

    if verbose:
        print(f"Params: {paras}, Loss: {loss}")

    return loss


def callback(xk, convergence):
    """
    Callback function for differential evolution optimization.

    Parameters
    ----------
    xk : array-like
        Current best solution parameters.
    convergence : float
        Current convergence value.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diff = sim(xk)
    with open(result_folder + 'callbacks.txt', 'a') as f:
        sys.stdout = f
        print(f"[{timestamp}] Loss: {diff}, Best Solution: {xk}, Convergence: {convergence}")
        sys.stdout = sys.__stdout__
    print(f"[{timestamp}] Loss: {diff}, Best Solution: {xk}, Convergence: {convergence}")


def compare_distributions(d_exp, d_model, test='KS'):
    """
    Compare experimental and model distributions.

    Parameters
    ----------
    d_exp : ndarray
        Experimental data distribution.
    d_model : ndarray
        Model data distribution.
    test : str, optional
        Statistical test to use ('KS' for Kolmogorov-Smirnov).

    Returns
    -------
    float
        Test statistic value.
    """
    if test == 'KS':
        stat, _ = ks_2samp(d_exp, d_model)
        return stat
    else:
        raise ValueError(f"Test {test} not implemented")


if __name__ == '__main__':
    result = differential_evolution(
        sim, 
        bounds, 
        workers=10,
        maxiter=2000,
        strategy='randtobest1bin',         
        mutation=(0.1, 1.0),
        tol=0.001,
        atol=0.0001,
        polish=True,  # Local refinement at the end
        init='sobol',
        seed=42,        
        callback=callback,
        updating='deferred'
    )


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(result_folder + 'callbacks.txt', 'a') as f:
        sys.stdout = f
        print(f"[{timestamp}] Final Result: {result}")
        sys.stdout = sys.__stdout__
    print(result)
    
    # Save optimal parameters as dataclass
    mu, eta0, eta1, k_s0, k_s1, u_s, f_s0, f_s1, f_s2, f_s3 = result.x
    
    optimal_params = ModelParams(
        N=20,
        mu=mu,
        eta=[eta0, eta1],
        k_l=k_l,
        k_s=[k_s0, k_s1],
        u_s=u_s,
        o_s=None,
        f_s=[f_s0, f_s1, f_s2, f_s3],
        h=0.0,
    )
    
    # Save optimal parameters
    optimal_params.save_json(result_folder + 'optimal_params.json')
    
    print(f"\nOptimal parameters saved to {result_folder}")
