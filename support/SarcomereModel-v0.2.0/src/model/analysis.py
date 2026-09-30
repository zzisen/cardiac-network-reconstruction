# Copyright (c) 2025-2026 Daniel Härtter
# Licensed under the MIT License.

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chisquare


def analyze_trajs(model):
    """ Analyze sarcomere trajectories (extrema of each sarcomeres contraction and velocity) """
    # initialize arrays
    # maximal contraction
    contr_max = np.zeros((len(model.data['length']), len(model.data['start_contr']))) * np.nan
    contr_max_avg = np.zeros(len(model.data['start_contr'])) * np.nan
    # maximal elongation
    elong_max = np.zeros_like(contr_max) * np.nan
    elong_max_avg = np.zeros_like(contr_max_avg)
    # maximal velocity in both directions
    vel_contr_max = np.zeros_like(contr_max) * np.nan
    vel_elong_max = np.zeros_like(contr_max) * np.nan
    vel_contr_max_avg = np.zeros_like(contr_max_avg)
    vel_elong_max_avg = np.zeros_like(contr_max_avg)

    # iterate single sarcomeres
    start_contr = model.data['start_contr']
    for j, length_j in enumerate(model.data['length']):
        vel_j = model.data['vel'][j]
        for i, contr in enumerate(start_contr):
            if i < len(start_contr) - 1:  # ignore last contraction cycle
                # get time-series of one contraction cycle (start to start)
                length_i = length_j[contr:start_contr[i + 1]]
                vel_i = vel_j[contr:start_contr[i + 1]]
                # find extrema
                contr_max[j][i] = np.nanmin(length_i)
                elong_max[j][i] = np.nanmax(length_i)
                vel_contr_max[j][i] = np.nanmin(vel_i)
                vel_elong_max[j][i] = np.nanmax(vel_i)

    # average contraction
    for i, contr in enumerate(start_contr):
        if i < len(start_contr) - 1:  # ignore last contraction cycle
            # get time-series of one contraction cycle (start to start)
            length_i = model.data['length_avg'][contr:start_contr[i + 1]]
            vel_i = model.data['vel_avg'][contr:start_contr[i + 1]]
            # find extrema
            contr_max_avg[i] = np.nanmin(length_i)
            elong_max_avg[i] = np.nanmax(length_i)
            vel_contr_max_avg[i] = np.nanmin(vel_i)
            vel_elong_max_avg[i] = np.nanmax(vel_i)

    # save data
    model.data.update({'contr_max': contr_max, 'elong_max': elong_max, 'vel_contr_max': vel_contr_max,
                       'vel_elong_max': vel_elong_max, 'contr_max_avg': contr_max_avg,
                       'elong_max_avg': elong_max_avg, 'vel_contr_max_avg': vel_contr_max_avg,
                       'vel_elong_max_avg': vel_elong_max_avg})
    if model.autosave:
        model.save_model()


def analyze_popping(model, thres_popping=0.3, plot=False):
    """Analyze sarcomere popping - popping if elongation larger than thres_popping"""
    # identify popping events
    elong_max = model.data['elong_max']
    popping = np.zeros_like(elong_max, dtype='bool')
    popping[elong_max > thres_popping] = 1

    # calculate frequencies
    freq_time = np.mean(popping, axis=0)
    freq_sarcomeres = np.mean(popping, axis=1)
    prop = np.mean(popping)

    # dictionary
    dict_popping = {'popping_freq_time': freq_time, 'popping_freq_sarcomeres': freq_sarcomeres,
                    'popping_freq': prop, 'popping_events': popping}
    model.data.update(dict_popping)
    if model.autosave:
        model.save_model()

    if plot:
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax.pcolorfast(popping, cmap='Greys')
        ax_histx.bar(np.arange(len(freq_time)) + 0.5, freq_time, color='k', alpha=0.4)
        ax_histy.barh(np.arange(len(freq_sarcomeres)) + 0.5, freq_sarcomeres, color='k', alpha=0.4)

        ax.set_xlabel('Contraction cycle [#]')
        ax.set_ylabel('Sarcomere [#]')
        yticks = np.arange(len(freq_sarcomeres))
        ax.set_yticks(yticks + 0.5)
        ax.set_yticklabels(yticks)
        ax_histx.set_ylabel('Frequency')
        ax_histy.set_xlabel('Frequency')
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.grid()

        plt.tight_layout()
        fig.savefig(model.folder + 'popping.png', dpi=300)
        plt.show()


def analyze_popping_tau_dist(model):
    """Analyze time gap between popping events in same sarcomere and distance of popping events in
    each contraction cycle"""

    popping_events = model.data['popping_events']
    idxs_popping_s, idxs_popping_c = np.where(popping_events == 1)

    # inter sarcomere distance of popping events in each contraction cycle
    cycles = np.unique(idxs_popping_c)
    dist_popping = []
    for t in cycles:
        popping_t = idxs_popping_s[idxs_popping_c == t]
        dist_t = np.diff(popping_t)
        dist_popping.append(dist_t)
    try:
        dist_popping = np.concatenate(dist_popping)
    except:
        dist_popping = []

    # time gap between popping events of same sarcomere
    sarcomeres = np.unique(idxs_popping_s)
    tau_popping = []
    for s in sarcomeres:
        popping_s = idxs_popping_c[idxs_popping_s == s]
        tau_s = np.diff(popping_s)
        tau_popping.append(tau_s)
    try:
        tau_popping = np.concatenate(tau_popping)
    except:
        tau_popping = []

    geometric_distr = lambda k, p: (1 - p) ** (k - 1) * p
    p = model.data['popping_freq']
    if p > 0.05:
        bins = 400
        range_ = np.arange(1, bins + 1)
        geometric_distr_p = geometric_distr(range_, p)
        # tau
        tau_histogram = np.histogram(tau_popping, density=True, range=(1, bins + 1), bins=bins)[0]
        # dist
        dist_histogram = np.histogram(dist_popping, density=True, range=(1, bins + 1), bins=bins)[0]
        # chi-square test
        chisq_dist, p_dist = chisquare(dist_histogram, geometric_distr_p, ddof=np.count_nonzero(dist_histogram) - 2)
        chisq_tau, p_tau = chisquare(tau_histogram, geometric_distr_p, ddof=np.count_nonzero(tau_histogram) - 2)
    else:
        chisq_dist, chisq_tau = np.nan, np.nan

    # store in dictionary
    model.data['popping_dist'] = dist_popping
    model.data['popping_tau'] = tau_popping
    model.data['popping_chisq_dist'] = chisq_dist
    model.data['popping_chisq_tau'] = chisq_tau
    if model.autosave:
        model.save_model()


def uncentered_corr(x, y):
    """
    Compute uncentered correlation (cosine similarity) between two vectors.
    This matches equation (1) in the manuscript.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return np.nan if denom == 0 else float(np.dot(x, y) / denom)


def correlation_cycles_mutual_serial(model):
    """
    Compute mutual and serial correlation coefficients for sarcomere dynamics.
    
    Mutual correlation: synchrony between different sarcomeres within the same contraction cycle
    Serial correlation: consistency of individual sarcomeres across different contraction cycles
    
    Uses uncentered correlation (cosine similarity) as defined in equation (1).
    """
    if model.data['n_contr'] > 0:
        time_contr = model.data['time_contr_tps']  # time points per contraction cycle
        
        # Initialize correlation matrices - use dataclass attribute access
        corr_length_cycle = np.zeros((model.model_params.N, model.model_params.N,
                                     model.data['n_contr'], model.data['n_contr'])) * np.nan
        corr_vel_cycle = np.zeros((model.model_params.N, model.model_params.N,
                                 model.data['n_contr'], model.data['n_contr'])) * np.nan
        
        # Compute all pairwise correlations
        for i in range(model.model_params.N):
            for j in range(model.model_params.N):
                length_i = model.data['length'][i]
                vel_i = model.data['vel'][i]
                length_j = model.data['length'][j]
                vel_j = model.data['vel'][j]
                
                for k, contr_k in enumerate(model.data['start_contr'][:-1]):
                    for l, contr_l in enumerate(model.data['start_contr'][:-1]):
                        if k != l or i != j:  # Skip identical sarcomere-cycle pairs
                            # Extract motion patterns for this cycle
                            length_i_cycle = length_i[contr_k:contr_k + time_contr]
                            length_j_cycle = length_j[contr_l:contr_l + time_contr]
                            vel_i_cycle = vel_i[contr_k:contr_k + time_contr]
                            vel_j_cycle = vel_j[contr_l:contr_l + time_contr]
                            
                            # Compute uncentered correlations (cosine similarity)
                            corr_length_cycle[i, j, k, l] = uncentered_corr(length_i_cycle, length_j_cycle)
                            corr_vel_cycle[i, j, k, l] = uncentered_corr(vel_i_cycle, vel_j_cycle)
        
        # Serial correlation: r_s = <r(i,i,k,l)>_{k≠l}
        # Average over all pairs of different contraction cycles for the same sarcomere
        serial_length_values = []
        serial_vel_values = []
        for i in range(model.model_params.N):
            for k in range(model.data['n_contr']):
                for l in range(model.data['n_contr']):
                    if k != l and not np.isnan(corr_length_cycle[i, i, k, l]):
                        serial_length_values.append(corr_length_cycle[i, i, k, l])
                        serial_vel_values.append(corr_vel_cycle[i, i, k, l])
        
        corr_length_serial = np.nanmean(serial_length_values) if serial_length_values else np.nan
        corr_vel_serial = np.nanmean(serial_vel_values) if serial_vel_values else np.nan
        
        # Mutual correlation: r_m = <r(i,j,k,k)>_{i≠j}  
        # Average over all pairs of different sarcomeres within the same contraction cycle
        mutual_length_values = []
        mutual_vel_values = []
        for k in range(model.data['n_contr']):
            for i in range(model.model_params.N):
                for j in range(model.model_params.N):
                    if i != j and not np.isnan(corr_length_cycle[i, j, k, k]):
                        mutual_length_values.append(corr_length_cycle[i, j, k, k])
                        mutual_vel_values.append(corr_vel_cycle[i, j, k, k])
        
        corr_length_mutual = np.nanmean(mutual_length_values) if mutual_length_values else np.nan
        corr_vel_mutual = np.nanmean(mutual_vel_values) if mutual_vel_values else np.nan
        
    else:
        corr_length_serial = np.nan
        corr_vel_serial = np.nan
        corr_length_mutual = np.nan
        corr_vel_mutual = np.nan
    
    # Store as individual dictionary entries (not a tuple)
    model.data['corr_length_serial'] = corr_length_serial
    model.data['corr_vel_serial'] = corr_vel_serial
    model.data['corr_length_mutual'] = corr_length_mutual
    model.data['corr_vel_mutual'] = corr_vel_mutual
    
    if model.autosave:
        model.save_model()
