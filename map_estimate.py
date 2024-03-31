"""!@file map_estimate.py
@brief Module containing code to calculate the MAP estimate
@details This module contains code to calculate the MAP estimate for parameters for a specific parameter
indexed by i in the MCMC chain
@author Created by C. Factor on 31/03/2024
"""
import numpy as np


def map_estimate(samples, i, num_bins=1000):
    """
    @brief Calculate the MAP estimate for the parameters
    @param samples: The samples from the MCMC chain
    @param i: The index of the parameter for which to calculate the MAP estimate
    @return The MAP estimate for the chosen parameter"""
    # Get the data for the parameter
    data = samples[:, i]
    # Calculate the histogram of the data
    hist_counts, bin_edges = np.histogram(data, bins=num_bins)
    # Find the index of the maximum count
    max_count_index = np.argmax(hist_counts)
    # Find the bin edges of the maximum count and return the average
    map = (bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2
    return map
