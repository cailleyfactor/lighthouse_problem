"""!@file gelmanrubin.py
@brief Module containing code to calculate the Gelman-Rubin statistic
@details This module contains code to calculate the Gelman-Rubin statistic for the parameters alpha, beta and I_0
@author Created by C. Factor on 30/03/2024
"""

import numpy as np


# The Gelman-Rubin statistic
def gelman_rubin_alpha(samples):
    """
    @brief The Gelman-Rubin statistic for the alpha parameter
    @param samples: The samples from the MCMC chain
    @return The Gelman-Rubin statistic for the alpha parameter
    """
    samples = samples[:, :, 0]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)

    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)

    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)

    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1) * W) / n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR


# # The Gelman-Rubin statistic
def gelman_rubin_beta(samples):
    """
    @brief The Gelman-Rubin statistic for the beta parameter
    @param samples: The samples from the MCMC chain
    @return The Gelman-Rubin statistic for the beta parameter
    """
    samples = samples[:, :, 1]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)

    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)

    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)

    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1) * W) / n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR


# The Gelman-Rubin statistic
def gelman_rubin_I(samples):
    """
    @brief The Gelman-Rubin statistic for the intensity parameter
    @param samples: The samples from the MCMC chain
    @return The Gelman-Rubin statistic for the intensity parameter"""
    samples = samples[:, :, 2]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)

    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)

    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)

    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1) * W) / n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR
