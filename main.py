"""!@file main.py
@brief Module containing code to run the coursework
@details This module contains tools for using an MCMC sampler to sample from the joint posterior on alpha and beta,
and then to sample from the joint posterior on alpha, beta and I_0. The module also contains code to calculate the
Gelman-Rubin statistic for the parameters alpha, beta and I_0.
@author Created by C. Factor on 01/03/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
from gelmanrubin import gelman_rubin
from sample import sample, sample_with_intensity
from map_estimate import map_estimate

# Load the data from the .txt file
data = np.loadtxt("lighthouse_flash_data.txt")

# Define the arguments for the sampler, including the prior bounds
x = data[:, 0]
Intensity = data[:, 1]
alpha_max = 100
alpha_min = -100
beta_max = 100
beta_min = 0

# Print out the min and max values of the data to determine the prior ranges
print("The minimum value of x is:", x.min(), "and the max of x is:", x.max())
print(
    "The minimum value of I is:",
    Intensity.min(),
    "and the max of I is:",
    Intensity.max(),
)

# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 100, 2
nsteps = 10000

# Get the chain
sampler = sample(x, alpha_max, alpha_min, beta_max, beta_min, nwalkers, ndim, nsteps)
chain = sampler.get_chain(flat=True)

# Trace plot
chain_labels = [r"$\alpha$", r"$\beta$"]
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
print(
    "Autocorrelation time for alpha is:",
    taus[0],
    "Autocorrelation time for beta is:",
    taus[1],
)

# Get the maximum autocorrelation time
tau = max(taus)
print("Max autocorrelation time is", tau)

# To get iid samples, thin the chain, and discard the burn-in
iid_samples = sampler.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(iid_samples)

# Printing the number of iid samples
print("The number of samples after thinning is", num_samples)

# Means of alpha and beta
mean_alpha = np.mean(iid_samples[:, 0], axis=0)
mean_beta = np.mean(iid_samples[:, 1], axis=0)

# Standard deviations
stdev_alpha = np.std(iid_samples[:, 0], ddof=1)
stdev_beta = np.std(iid_samples[:, 1], ddof=1)

print(
    "The MAP of alpha is:",
    map_estimate(iid_samples, 0),
    "The MAP of beta is:",
    map_estimate(iid_samples, 1),
)
print(
    "The mean of alpha is:",
    mean_alpha,
    "The standard deviation of alpha is:",
    stdev_alpha,
)
print(
    "The mean of beta is:", mean_beta, "The standard deviation of beta is:", stdev_beta
)

# Corner plot - plotting 1D marginalised histograms
labels = [r"$\alpha$", r"$\beta$"]
fig = corner.corner(
    iid_samples,
    bins=50,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    show_titles=True,
)

# Corner plot has a bug and many of its arguments are not working
# Define hist2d kwargs
hist2d_kwargs = {"bins": 50, "cmap": "viridis", "density": True, "alpha": 0.75}


# Create a figure with 2x2 subplots
axes = np.array(fig.axes).reshape(2, 2)

# Iterate over the axes to create custom 2D histograms to replace the 2D histograms in corner
for i in range(2):
    for j in range(i):
        ax = axes[i, j]
        # Clear the axis to draw a new 2D histogram
        ax.clear()

        # Choose the data for the x and y axis
        x_data = iid_samples[:, j]
        y_data = iid_samples[:, i]

        # Create the 2D histogram for the current pair of parameters.
        ax.hist2d(x_data, y_data, **hist2d_kwargs)

        # y-axis labels for the first column
        if j == 0:
            ax.set_ylabel(labels[i])
        # x-axis labels for the bottom row
        if i == 1:
            ax.set_xlabel(labels[j])

# Set the labels on the edges
for i in range(2):
    # Set y-axis labels for the first column
    axes[i, 0].set_ylabel(labels[i])
    # Set x-axis labels for the bottom row
    axes[1, i].set_xlabel(labels[i])

# Hide x-axis labels and ticks for all but the bottom row
for ax in axes[:-1, :].flatten():
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position("none")

# Hide y-axis labels and ticks for all but the first column
for ax in axes[:, 1:].flatten():
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position("none")
plt.show()

# Get a second chain for use in calculating the Gelman-Rubin statistic
sampler2 = sample(x, alpha_max, alpha_min, beta_max, beta_min, nwalkers, ndim, nsteps)

# Get chain from second sampler (and previous chain with just burn-in)
chain = sampler.get_chain(flat=True, discard=1000)
chain2 = sampler2.get_chain(flat=True, discard=1000)

# Combine chains into a single chain
chains = np.array([chain, chain2])

# Calculate the Gelman-Rubin statistic for alpha and beta
GR_alpha = gelman_rubin(chains, 0)
GR_beta = gelman_rubin(chains, 1)
print(
    "The Gelman-Rubin statistic for alpha is:",
    GR_alpha,
    "The Gelman-Rubin statistic for beta is:",
    GR_beta,
)

# vi
# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 150, 3
nsteps = 10000

# Define additional args for the sampler and start positions
Intensity = data[:, 1]
I_0max = 100
I_0min = 0.001

# Run MCMC sampler for the 3D posterior including intensity
sampler3 = sample_with_intensity(
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
)

# Get the chain
chain3 = sampler3.get_chain(flat=True)

# Trace plot
chain_labels = [r"$\alpha$", r"$\beta$", r"$I_0$"]
for i in range(chain3.shape[1]):
    plt.plot(chain3[:, i], alpha=0.5, label=chain_labels[i])
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.show()

# Acceptance rate
print("The acceptance fraction is", np.mean(sampler3.acceptance_fraction))

# Get autocorrelation times
taus = sampler3.get_autocorr_time(tol=2)
print(
    "The autocorrelation time for alpha is",
    taus[0],
    "The autocorrelation time for beta is",
    taus[1],
    "The autocorrelation time for intensity_0 is",
    taus[2],
)

# Get the maximum autocorrelation time
tau = max(taus)
print("The maximum autocorrelation time is", tau)

# To get iid samples, thin the chain, and discard the burn-in
iid_samples = sampler3.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(iid_samples)

# Printing the number of iid samples and time per iid sample
print("The number of i.i.d. samples =", num_samples)

# Means of alpha, beta and intensity_0
mean_alpha = np.mean(iid_samples[:, 0], axis=0)
mean_beta = np.mean(iid_samples[:, 1], axis=0)
mean_intensity_0 = np.mean(iid_samples[:, 2], axis=0)

# Standard deviations
stdev_alpha = np.std(iid_samples[:, 0], ddof=1)
stdev_beta = np.std(iid_samples[:, 1], ddof=1)
stdev_intensity_0 = np.std(iid_samples[:, 2], ddof=1)

# Print the MAP estimates
print(
    "The MAP of alpha is:",
    map_estimate(iid_samples, 0),
    "The MAP of beta is:",
    map_estimate(iid_samples, 1),
    "The MAP of intensity_0 is:",
    map_estimate(iid_samples, 2),
)

# Print the mean estimates and stdev
print(
    "The mean of alpha is:",
    mean_alpha,
    "The standard deviation of alpha is:",
    stdev_alpha,
)
print(
    "The mean of beta is:", mean_beta, "The standard deviation of beta is:", stdev_beta
)
print(
    "The mean of intensity_0 is:",
    mean_intensity_0,
    "The standard deviation of intensity_0 is:",
    stdev_intensity_0,
)

# Corner plot - plotting 1D marginalised histograms
labels = [r"$\alpha$", r"$\beta$", r"$I_0$"]
fig = corner.corner(
    iid_samples,
    bins=50,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    show_titles=True,
)

# Corner plot has a bug and many of its arguments are not working
# Define hist2d kwargs
hist2d_kwargs = {"bins": 50, "cmap": "viridis", "density": True, "alpha": 0.75}

# Create a figure with 3x3 subplots
axes = np.array(fig.axes).reshape(3, 3)

# Iterate over the axes to create custom 2D histograms to replace the 2D histograms in corner
for i in range(3):
    for j in range(i):
        ax = axes[i, j]
        # Clear the axis to draw a new 2D histogram
        ax.clear()

        # Choose the data for the x and y axis
        x_data = iid_samples[:, j]
        y_data = iid_samples[:, i]

        # Create the 2D histogram for the current pair of parameters.
        ax.hist2d(x_data, y_data, **hist2d_kwargs)

# Set the labels on the edges for a 3x3 corner plot
for i in range(3):
    # Set y-axis labels for the first column
    axes[i, 0].set_ylabel(labels[i])
    # Set x-axis labels for the bottom row
    axes[2, i].set_xlabel(labels[i])

# Hide x-axis labels and ticks for all but the bottom row for a 3x3 grid
# Only iterate over the first two rows
for i in range(2):
    for ax in axes[i, :]:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")

# Hide y-axis labels and ticks for all but the first column for a 3x3 grid
# Start from the second column
for j in range(1, 3):
    for ax in axes[:, j]:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
plt.show()

# Repeat running the MCMC chain to obtain the GR statistic
sampler4 = sample_with_intensity(
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
)

# Get chain from second sampler (and previous chain with just burn-in) - (nsteps * nwalkers, ndim)
chain3 = sampler3.get_chain(flat=True, discard=1000)
chain4 = sampler4.get_chain(flat=True, discard=1000)

# Combine chains into a single chain - (2, nsteps * nwalkers, ndim)
chains_with_I = np.array([chain3, chain4])

# Calculate the Gelman-Rubin statistic for alpha, beta and intensity_0 and print
GR_alpha = gelman_rubin(chains_with_I, 0)
GR_beta = gelman_rubin(chains_with_I, 1)
GR_I_0 = gelman_rubin(chains_with_I, 2)
print(
    "The Gelman-Rubin statistic for alpha is:",
    GR_alpha,
    "The Gelman-Rubin statistic for beta is:",
    GR_beta,
    "The Gelman-Rubin statistic for intensity_0 is:",
    GR_I_0,
)
