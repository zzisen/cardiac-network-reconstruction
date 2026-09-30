# Copyright (c) 2025-2026 Daniel Härtter
# Licensed under the MIT License.

import hashlib

import random
import string

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.ndimage import label
from scipy.stats import skewnorm, ks_2samp, anderson_ksamp, wasserstein_distance


def compare_distributions(d1, d2, test, large_value=1e3, normalize_wasserstein=True):
    """
    Compare two distributions using specified statistical test.

    Parameters
    ----------
    d1 : array_like
        First sample of observations.
    d2 : array_like
        Second sample of observations.
    test : {'KS', 'Anderson', 'Wasserstein'}
        The statistical test to use for comparison:
        - 'KS': Kolmogorov-Smirnov two-sample test
        - 'Anderson': Anderson-Darling test
        - 'Wasserstein': Wasserstein distance (Earth Mover's Distance)
    large_value : float, optional
        Value to return if either input is empty or if the result is NaN (default is 1e3).
    normalize_wasserstein : bool, optional
        Whether to normalize the Wasserstein distance (default is True).

    Returns
    -------
    float
        The test statistic. Returns `large_value` if either input is empty or if the
        result is NaN. For Wasserstein distance, returns normalized value if specified.

    Raises
    ------
    ValueError
        If an undefined test is specified.

    Notes
    -----
    The function handles empty inputs by returning `large_value`.
    Wasserstein distance is normalized by the range of the combined data when specified.

    Examples
    --------
    >>> d1 = np.random.normal(0, 1, 1000)
    >>> d2 = np.random.normal(0.5, 1.2, 1000)
    >>> ks_stat = compare_distributions(d1, d2, 'KS')
    >>> wasserstein_stat = compare_distributions(d1, d2, 'Wasserstein')
    """
    if d1.size > 0 and d2.size > 0:
        if test == 'KS':
            stat = ks_2samp(d1, d2).statistic
        elif test == 'Anderson':
            stat = anderson_ksamp([d1, d2]).statistic
        elif test == 'Wasserstein':
            stat = wasserstein_distance(d1, d2)
            if normalize_wasserstein:
                min_value = np.min(d1)
                max_value = np.max(d2)
                stat = stat / (max_value - min_value)
                stat = np.abs(stat)
        else:
            raise ValueError(f'statistic {test} not defined.')
        if np.isnan(stat):
            stat = large_value
    else:
        stat = large_value
    return stat


def remove_nans(arr):
    return arr[~np.isnan(arr)]


def get_id_from_string(input_string, length=6):
    # Create a hash object
    hash_object = hashlib.sha1()

    # Encode the input string and update the hash object with this encoded string
    hash_object.update(input_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()

    # Return the first 10 characters of the hexadecimal hash as the short ID
    return hex_dig[:length]


def get_random_id(length):
    # Define the characters that can be used in the ID
    characters = string.ascii_letters + string.digits  # All uppercase and lowercase letters and digits

    # Generate a 6 character long random ID
    random_id = ''.join(random.choice(characters) for _ in range(length))

    return random_id


def split_contraction_intervals(contr, n_intervals=5):
    """
    Split each positive interval in N equal intervals.

    Parameters
    ----------
    contr : ndarray
        A 1D binary time-series array where positive intervals (sequences of 1s)
        are to be split into two halves (contraction and relaxation).
    n_intervals : int
        Number of equally long contraction intervals
    Returns
    -------
    contr_intervals : ndarray
        A 1D time-series array where the different contraction intervals are labels (1, 2, ..., n_intervals)

    """
    labels_contr, n_labels_contr = label(contr)
    contr_intervals = np.zeros_like(contr, dtype='uint8')

    # Iterate over each labeled region
    for label_idx in range(1, n_labels_contr + 1):
        # Find the indices for the current label
        indices = np.where(labels_contr == label_idx)[0]

        contr_intervals[indices] = np.ceil((indices - indices[0] + 1) / len(indices) * n_intervals)

    return contr_intervals
