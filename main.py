import numpy as np
import pandas as pd
from scipy.stats import expon
import matplotlib.pyplot as plt
import emcee
import corner
import time

# Load the data from the .txt file
data = np.loadtxt('lighthouse_flash_data.txt')
# print(data)

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

# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 10,2
nsteps = 2000

# Define args for the sampler and start positions
x = data[:,0]
alpha_max = 10
alpha_min = -10
scale = 1

# Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions[:, 1] = start_positions[:, 1] * 10 

# Initialise & run sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, scale])
sampler.run_mcmc(start_positions, nsteps)

# Get autocorrelation times 
taus = sampler.get_autocorr_time(tol=2)
print('Autocorrelation:', taus)

# Get the maximum autocorrelation time
tau = max(taus)
print(f"{tau = }")

# Get the chain
chain = sampler.get_chain(flat=True)

# Trace plot 
for i in range(chain.shape[1]):  
    plt.plot(chain[:, i], alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()

# Effective sample size
ess = len(chain)/tau
print("Effective sample size:", ess)

# # To get iid samples, thin the chain and discard the burn-in - question of dealing with burn-in and thinning
# iid_samples = sampler.get_chain(flat=True, thin=2*int(tau), discard=int(0.05*nsteps))
# num_samples = len(iid_samples)

# # Printing the number of iid samples and time per iid sample
# print("I.i.d samples = {}".format(num_samples))

# Estimate the expectation with axis=0 to average over all samples for each parameter
mean_alpha = np.mean(chain[:,0], axis=0)
mean_beta = np.mean(chain[:,1], axis=0)

# Would think that SEM is better?
sem_alpha = np.std(chain[:,0])/np.sqrt(len(chain))
sem_beta = np.std(chain[:,1])/np.sqrt(len(chain))

# Think that SEM is better than stdev
stdev_alpha = np.std(chain[:,0], ddof=1)
stdev_beta = np.std(chain[:,1], ddof=1)

print(mean_alpha, stdev_alpha)
print(mean_beta, stdev_beta)

# Plot 2D histogram
plt.hist2d(chain[:,0], chain[:,1], bins=50, cmap='Blues')
plt.colorbar()
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.show()

# Corner plot - plotting 1D marginalised histograms
fig = corner.corner(chain, labels=["alpha", "beta"], plot_datapoints=True, plot_density=True, plot_contours=True, hist_kwargs={"density": True}, show_titles=True)
plt.show()

# # Repeat running the MCMC chain to obtain the GR statistic
# # Intialise random starting positions for the walkers in the parameter space
# start_positions = np.random.rand(nwalkers, ndim)
# # Ensure that the start positions are in the sample space
# start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min 
# start_positions[:, 1] = start_positions[:, 1] * 10 

# # Initialise & run sampler
# sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, scale])
# sampler2.run_mcmc(start_positions, nsteps)

# # Get chain from second sampler
# chain2 = sampler2.get_chain(flat=True)

# # Combine chains into a single chain
# chains = np.array([chain,chain2])

# # # The Gelman-Rubin statistic
# def gelman_rubin(samples):
#     n_chains, n_samples, n_dim = samples.shape
#     chain_means = np.mean(samples, axis=1)
#     chain_variances = np.var(samples, axis=1, ddof=1)
    
#     # Between-chain variance
#     B = n_samples * np.var(chain_means, ddof=1)
    
#     # Mean of empirical variance between each chain
#     W = np.mean(chain_variances)
    
#     # Estimated variance of the target distribution
#     sigma_sq = (((n_samples - 1)*W)/ n_samples) + (B / n_samples)
    
#     # T-distribution variance, m=2 chains
#     sqrt_t_variance = np.sqrt(sigma_sq + (B/(n_samples * 2)))

#     # DOF
#     if np.var(sqrt_t_variance) == 0:
#         print("Variance of sqrt_t_variance is zero - divide by zero error.")
#     else:
#         DOF= 2*(sqrt_t_variance**2)/np.var(sqrt_t_variance)

#     # Potential Scale Reduction Factor
#     GR = np.sqrt((sqrt_t_variance * DOF) /((DOF -2) * W))
    
#     return GR

# GR = gelman_rubin(chains)
