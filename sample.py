"""!@file sample.py
@brief Module containing code to run the MCMC sampler
@details This module contains code to initialise random starting positions for the walkers in the parameter space,
run the MCMC sampler, and return the sampler object. The module also contains code to run the MCMC
sampler with intensity as an additional parameter.
@author Created by C. Factor on 30/03/2024
"""

import emcee
from posterior import log_posterior, log_posterior_intensity
import numpy as np


def sample(x, alpha_max, alpha_min, beta_max, beta_min, nwalkers, ndim, nsteps):
    """
    @brief Run the MCMC sampler
    @param x: The location data where flashes are recieved by detectors along the coastline
    @param alpha_max: The maximum value of alpha
    @param alpha_min: The minimum value of alpha
    @param beta_max: The maximum value of beta
    @param beta_min: The minimum value of beta
    @param nwalkers: The number of walkers in the MCMC sampler
    @param ndim: The number of dimensions of the parameter space
    @param nsteps: The number of steps in the MCMC sampler
    @return The MCMC sampler object"""
    # Intialise random starting positions for the walkers in the parameter space
    start_positions = np.random.rand(nwalkers, ndim)
    # Ensure that the start positions are in the sample space
    start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min
    start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min

    # Initialise & run sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=[x, alpha_max, alpha_min, beta_max, beta_min],
    )
    sampler.run_mcmc(start_positions, nsteps, progress=True)

    return sampler


def sample_with_intensity(
    x,
    Intensity,
    alpha_max,
    alpha_min,
    beta_max,
    beta_min,
    I_0max,
    I_0min,
    nwalkers,
    ndim,
    nsteps,
):
    """
    @brief Run the MCMC sampler with intensity as an additional parameter
    @param x: The location data where flashes are recieved by detectors along the coastline
    @param Intensity: The intensity of the flashes from the lighthouse registered by the detectors
    @param alpha_max: The maximum value of alpha
    @param alpha_min: The minimum value of alpha
    @param beta_max: The maximum value of beta
    @param beta_min: The minimum value of beta
    @param I_0max: The maximum intensity of the lighthouse
    @param I_0min: The minimum intensity of the lighthouse
    @param nwalkers: The number of walkers in the MCMC sampler
    @param ndim: The number of dimensions of the parameter space
    @param nsteps: The number of steps in the MCMC sampler
    @return The MCMC sampler object"""
    # # Intialise random starting positions for the walkers in the parameter space
    start_positions = np.random.rand(nwalkers, ndim)

    # Ensure that the start positions are in the sample space
    start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min
    start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min
    start_positions[:, 2] = start_positions[:, 2] * (I_0max - I_0min) + I_0min

    # Initialise & run sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_intensity,
        args=[x, Intensity, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min],
    )
    sampler.run_mcmc(start_positions, nsteps, progress=True)
    return sampler
