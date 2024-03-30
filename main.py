"""!@file main.py
@brief Module containing code to run the coursework
@details This module contains tools for using an MCMC sampler to sample from the joint posterior on alpha and beta,
and then to sample from the joint posterior on alpha, beta and I_0. The module also contains code to calculate the
Gelman-Rubin statistic for the parameters alpha, beta and I_0.
@author Created by C. Factor on 01/03/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from posterior import log_posterior, log_posterior_intensity
from gelmanrubin import gelman_rubin_alpha, gelman_rubin_beta, gelman_rubin_I

# Load the data from the .txt file
data = np.loadtxt("lighthouse_flash_data.txt")

# Print out the min and max values of the data to determine the prior ranges
x = data[:, 0]
print("min x", x.min(), "max x", x.max())
Intensity = data[:, 1]
print("min I", Intensity.min(), "max I", Intensity.max())

# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 100, 2
nsteps = 10000

# Define args for the sampler and start positions
x = data[:, 0]
alpha_max = 100
alpha_min = -100
beta_max = 100
beta_min = 0

# Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min
start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min

# Initialise & run sampler
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, beta_max, beta_min]
)
sampler.run_mcmc(start_positions, nsteps, progress=True)

# Get the chain
chain = sampler.get_chain(flat=True)

# Trace plot
chain_labels = ["Alpha", "Beta"]
for i in range(chain.shape[1]):
    plt.plot(chain[:, i], alpha=0.5, label=chain_labels[i])
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.show()

# Print out the acceptance rate
print("The acceptance fraction is", np.mean(sampler.acceptance_fraction))

# Get autocorrelation times
taus = sampler.get_autocorr_time(tol=0)
print("Autocorrelation:", taus)

# Get the maximum autocorrelation time
tau = max(taus)
print(f"{tau = }")

# To get iid samples, thin the chain, and discard the burn-in
iid_samples = sampler.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(iid_samples)

# Printing the number of iid samples
print("The number of samples after thinning is", num_samples)

# Estimate the expectation with axis=0 to average over all samples for each parameter
mean_alpha = np.mean(iid_samples[:, 0], axis=0)
mean_beta = np.mean(iid_samples[:, 1], axis=0)

# Stdevs as specified
stdev_alpha = np.std(iid_samples[:, 0], ddof=1)
stdev_beta = np.std(iid_samples[:, 1], ddof=1)

print("mean alpha", mean_alpha, "stdev alpha", stdev_alpha)
print("mean alpha", mean_beta, "stdev alpha", stdev_beta)

# Corner plot - plotting 1D marginalised histograms
fig = corner.corner(
    iid_samples,
    labels=["alpha", "beta"],
    bins=50,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    show_titles=True,
    hist2d_kwargs={"cmap": "viridis", "density": True},
)
plt.show()

# Get the chain
chain = sampler.get_chain(flat=True, discard=1000)

# Repeat running the MCMC chain to obtain the GR statistic
# Intialise random starting positions for the walkers in the parameter space
start_positions2 = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions2[:, 0] = start_positions2[:, 0] * (alpha_max - alpha_min) + alpha_min
start_positions2[:, 1] = start_positions2[:, 1] * (beta_max - beta_min) + beta_min

# Initialise & run sampler
sampler2 = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, beta_max, beta_min]
)
sampler2.run_mcmc(start_positions2, nsteps)

# Get chain from second sampler
chain2 = sampler2.get_chain(flat=True, discard=1000)

# Combine chains into a single chain
chains = np.array([chain, chain2])

GR_alpha = gelman_rubin_alpha(chains)
GR_beta = gelman_rubin_beta(chains)
print(
    "Gelman Rubin statistic for alpha:",
    GR_alpha,
    "Gelman Rubin statistic for beta:",
    GR_beta,
)

# vi
# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 100, 3
nsteps = 10000

# Define args for the sampler and start positions
x = data[:, 0]
Intensity = data[:, 1]
alpha_max = 100
alpha_min = -100
beta_max = 100
beta_min = 0
I_0max = 100
I_0min = 0.001

# # Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)

# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min
start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min
start_positions[:, 2] = start_positions[:, 2] * (I_0max - I_0min) + I_0min

# Initialise & run sampler
sampler3 = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_posterior_intensity,
    args=[x, Intensity, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min],
)
sampler3.run_mcmc(start_positions, nsteps, progress=True)

# Get the chain
chain3 = sampler3.get_chain(flat=True)

# Trace plot
chain_labels = ["Alpha", "Beta", "Intensity_0"]
for i in range(chain3.shape[1]):
    plt.plot(chain3[:, i], alpha=0.5, label=chain_labels[i])
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.show()

# Acceptance rate plot
print("The acceptance fraction is", np.mean(sampler3.acceptance_fraction))

# Get autocorrelation times
taus = sampler3.get_autocorr_time(tol=2)
print("Autocorrelation:", taus)

# Get the maximum autocorrelation time
tau = max(taus)
print(f"{tau = }")

# To get iid samples, thin the chain, and discard the burn-in - question of dealing with burn-in and thinning
iid_samples = sampler3.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(iid_samples)

# Printing the number of iid samples and time per iid sample
print("The number of i.i.d. samples =", num_samples)

# Estimate the expectation with axis=0 to average over all samples for each parameter
mean_alpha = np.mean(iid_samples[:, 0], axis=0)
mean_beta = np.mean(iid_samples[:, 1], axis=0)
mean_intensity_0 = np.mean(iid_samples[:, 2], axis=0)

# Think that SEM is better than stdev
stdev_alpha = np.std(iid_samples[:, 0], ddof=1)
stdev_beta = np.std(iid_samples[:, 1], ddof=1)
stdev_intensity_0 = np.std(iid_samples[:, 2], ddof=1)

print("mean alpha", mean_alpha, "stdev alpha", stdev_alpha)
print("mean beta", mean_beta, "stdev beta", stdev_beta)
print("mean intensity_0", mean_intensity_0, "stdev intensity_0", stdev_intensity_0)

# Corner plot - plotting 1D marginalised histograms
fig = corner.corner(
    iid_samples,
    labels=["alpha", "beta", "intensity_0"],
    bins=50,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    show_titles=True,
)
plt.show()

# Repeat running the MCMC chain to obtain the GR statistic
# Intialise random starting positions for the walkers in the parameter space
start_positions2 = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions2[:, 0] = start_positions2[:, 0] * (alpha_max - alpha_min) + alpha_min
start_positions2[:, 1] = start_positions2[:, 1] * (beta_max - beta_min) + beta_min
start_positions2[:, 2] = start_positions2[:, 2] * (I_0max - I_0min) + I_0min

# Initialise & run sampler
sampler4 = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_posterior_intensity,
    args=[x, Intensity, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min],
)
sampler4.run_mcmc(start_positions2, nsteps)

# Get chain from second sampler
chain3 = sampler3.get_chain(flat=True, discard=1000)
chain4 = sampler4.get_chain(flat=True, discard=1000)

# Combine chains into a single chain
chains_with_I = np.array([chain3, chain4])

GR_alpha = gelman_rubin_alpha(chains_with_I)
GR_beta = gelman_rubin_beta(chains_with_I)
GR_I = gelman_rubin_I(chains_with_I)
print(
    "Gelman Rubin for alpha",
    GR_alpha,
    "Gelman Rubin for beta",
    GR_beta,
    "Gelman Rubin for Intensity_0",
    GR_I,
)
