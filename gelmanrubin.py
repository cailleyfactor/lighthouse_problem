"""!@file gelmanrubin.py
@brief Module containing code to calculate the Gelman-Rubin statistic
@details This module contains code to calculate the Gelman-Rubin statistic for the parameters alpha, beta and I_0
@author Created by C. Factor on 30/03/2024
"""

import numpy as np


# The Gelman-Rubin statistic
def gelman_rubin(samples, parameter_index):
    """
    @brief The Gelman-Rubin statistic
    @param samples: The samples from the MCMC chain
    @param parameter_index: The index of the parameter for which to calculate the Gelman-Rubin statistic
    @return The Gelman-Rubin statistic for the chosen parameter
    """
    samples = samples[:, :, parameter_index]
    n_chains, n_samples = samples.shape

    # Compute mean of rows, i.e. the mean of each chain - shape (n_sequences,)
    chain_means = np.mean(samples, axis=1)  # Take mean along row i.e. of each chain

    grand_mean = np.mean(chain_means)  # Mean of the means of each chain

    # Compute between-chain variance
    B = (n_samples / (n_chains - 1)) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Compute chain variances
    chain_variances = np.var(samples, axis=1, ddof=1)

    # Mean of within chain variances, i.e., average sample variance
    W = np.mean(chain_variances, axis=0)

    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1) * W) / n_samples) + (B / n_samples)

    # GR for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR
