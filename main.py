import numpy as np
import pandas as pd
from scipy.stats import expon
import matplotlib.pyplot as plt
import emcee

# Load the data from the .txt file
data = np.loadtxt('lighthouse_flash_data.txt')
print(data)

# Def the cauchy PDF
def cauchy_pdf(x, alpha, beta):
    numerator = beta
    # Broadcast by enabling x to have two more dimensions
    denominator = np.pi * (beta**2 + ((x[:, None, None] - alpha)**2))
    return numerator / denominator

# Def the log posterior
def log_posterior(params, x, alpha_max, alpha_min, scale):
    alpha, beta = params
    if alpha_min < alpha < alpha_max and beta > 0:
        prior_alpha = 1/(alpha_max-alpha_min)
        prior_beta = expon(scale=scale).pdf(beta)
        cauchy_pdf_val = cauchy_pdf(x, alpha, beta)
        # Multiply the cauchy_pdf_val for each value of x
        likelihood = np.prod(cauchy_pdf_val, axis=0)
        return np.log(likelihood) + np.log(prior_alpha) + np.log(prior_beta)
    else:  
        # return a very small log posterior if alpha or beta are out of bounds
        return -np.inf


# Using zeus package for stochastic sampling
# Initialise parameters for zeus, nwalkers at least two times dimension
nwalkers, ndim = 10,2
nsteps = 2000

# Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)

# Define args for the sampler and start positions
x = data[:,0]
alpha_max = 10
alpha_min = -10
scale = 1

# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions[:, 1] = start_positions[:, 1] * 10 # Ensuring that beta is not intialised too large

# Initialise sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, scale])

# Run the MCMC for 1000 steps starting from the initial positions
sampler.run_mcmc(start_positions, nsteps)

# Retrieve the samples
burn_in = int(0.25 *nsteps)
chain = sampler.get_chain(discard = burn_in, flat=True)
print(chain)

# Retrieve the autocorrelation time
tau = sampler.get_autocorr_time(tol=0)

# Plot a 2D histogram showing the joint posterior on alpha and beta
# alpha_values = np.linspace(-10, 10, 100)
# beta_values = np.linspace(0.01, 10, 100)
# alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)

# # Calculate the posterior for alpha_max = 10, alpha_min = 10, scale = 1
# posterior_values = posterior(data[:,0], alpha_grid, beta_grid, -10, 10, 1)
# print(posterior_values.shape)
# plt.figure(figsize=(10, 8))
# plt.contourf(alpha_grid, beta_grid, posterior_values, cmap='viridis')
# plt.colorbar(label='Posterior Probability')
# plt.xlabel('Alpha')
# plt.ylabel('Beta')
# plt.title('Joint Posterior Distribution on Alpha and Beta')
# plt.show()

# 