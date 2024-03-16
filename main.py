import numpy as np
import pandas as pd
from scipy.stats import expon
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.stats import lognorm

# Load the data from the .txt file
data = np.loadtxt('lighthouse_flash_data.txt')

# # print(data)
# I = data[:,1]
# print(I)
# print(I.min(), I.max())

# Def the cauchy PDF
def cauchy_pdf(x, alpha, beta):
    numerator = beta
    # Broadcast by enabling x to have two more dimensions
    denominator = np.pi * (beta**2 + ((x[:, None, None] - alpha)**2))
    return numerator / denominator

# Def the log posterior
def log_posterior(params, x, alpha_max, alpha_min, beta_max, beta_min):
    alpha, beta = params
    if alpha_min < alpha < alpha_max and beta_min < beta < beta_max :
        prior_alpha = 1/(alpha_max-alpha_min)
        prior_beta = 1/(beta_max-beta_min)
        cauchy_pdf_val = cauchy_pdf(x, alpha, beta)
        # Multiply the cauchy_pdf_val for each value of x
        likelihood = np.prod(cauchy_pdf_val, axis=0)
        return np.log(likelihood) + np.log(prior_alpha) + np.log(prior_beta)
    else:  
        # return a very small log posterior if alpha or beta are out of bounds
        return -np.inf

# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 100,2
nsteps = 10000

# Define args for the sampler and start positions
x = data[:,0]
alpha_max = 10
alpha_min = -10
beta_max = 10
beta_min = 0

# Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min 

# Initialise & run sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, beta_max, beta_min])
sampler.run_mcmc(start_positions, nsteps)

# Get the chain
chain = sampler.get_chain(flat=True)

# Trace plot 
for i in range(chain.shape[1]):  
    plt.plot(chain[:, i], alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()

# Should this be doen after burn in though?
# Get autocorrelation times 
taus = sampler.get_autocorr_time(tol=2)
print('Autocorrelation:', taus)

# Get the maximum autocorrelation time
tau = max(taus)
print(f"{tau = }")

# Effective sample size
ess = len(chain)/tau
print("Effective sample size:", ess)

# To get iid samples, thin the chain, and discard the burn-in - question of dealing with burn-in and thinning
iid_samples = sampler.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(chain)

# Printing the number of iid samples and time per iid sample
print("I.i.d samples = {}".format(num_samples))

# Estimate the expectation with axis=0 to average over all samples for each parameter
mean_alpha = np.mean(iid_samples[:,0], axis=0)
mean_beta = np.mean(iid_samples[:,1], axis=0)

# Would think that SEM is better?
sem_alpha = np.std(iid_samples[:,0])/np.sqrt(len(iid_samples))
sem_beta = np.std(iid_samples[:,1])/np.sqrt(len(iid_samples))

# Think that SEM is better than stdev
stdev_alpha = np.std(iid_samples[:,0], ddof=1)
stdev_beta = np.std(iid_samples[:,1], ddof=1)

print(mean_alpha, stdev_alpha)
print(mean_beta, stdev_beta)

# Plot 2D histogram
plt.hist2d(iid_samples[:,0], iid_samples[:,1], bins=50, cmap='Blues')
plt.colorbar()
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.show()

# Corner plot - plotting 1D marginalised histograms
fig = corner.corner(iid_samples, labels=["alpha", "beta"], bins=50, plot_datapoints=True, plot_density=True, plot_contours=True, show_titles=True)
plt.show()

# Get the chain
chain = sampler.get_chain(flat=True, discard=1000)

# Repeat running the MCMC chain to obtain the GR statistic
# Intialise random starting positions for the walkers in the parameter space
start_positions2 = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions2[:, 0] = start_positions2[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions2[:, 1] = start_positions2[:, 1] * (beta_max - beta_min) + beta_min 
# # Ensure that the start positions are in the sample space
# start_positions[:, 0] = 1 + 0.1 * np.random.randn(nwalkers)
# start_positions[:, 1] = 1 + 0.1 * np.random.randn(nwalkers)

# Initialise & run sampler
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, alpha_max, alpha_min, beta_max, beta_min])
sampler2.run_mcmc(start_positions2, nsteps)

# Get chain from second sampler
chain2 = sampler2.get_chain(flat=True, discard=1000)

# Combine chains into a single chain
chains = np.array([chain,chain2])
print(chain.shape, chain2.shape, chains.shape)

# The Gelman-Rubin statistic
def gelman_rubin_alpha(samples):
    samples = samples[:,:,0]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)
    
    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1)*W)/ n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR

# # The Gelman-Rubin statistic
def gelman_rubin_beta(samples):
    samples = samples[:,:,1]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)
    
    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1)*W)/ n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR

GR_alpha = gelman_rubin_alpha(chains)
GR_beta = gelman_rubin_beta(chains)
print("Gelman Rubin statistic:", GR_alpha, "beta", GR_beta)



# vi
# Prior for intensity
def prior_I_0(I_0, I_0max, I_0min):
    return 1 / (I_0 * np.log(I_0max / I_0min))

# vii
# Log-norm likelihood
def log_normal_pdf(x, I, alpha, beta, I_0):
    d_squared = (beta**2) + ((x - alpha)**2)
    mu = np.log(I_0/d_squared)
    sigma = 1
    numerator = np.exp(-(np.log(I)-mu)**2/(2*(sigma**2)))
    denominator = I * np.sqrt(2 * np.pi *(sigma**2))
    return numerator/denominator

# Def the log posterior
def log_posterior_intensity(params, x, I, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min):
    alpha, beta, I_0 = params
    if alpha_min < alpha < alpha_max and beta_min < beta < beta_max and I_0min < I_0 < I_0max:
        prior_alpha = 1/(alpha_max-alpha_min)
        prior_beta = 1/(beta_max-beta_min)
        prior_intensity_0 = prior_I_0(I_0, I_0max, I_0min)
        cauchy_pdf_val = cauchy_pdf(x, alpha, beta)
        log_normal_pdf_val = log_normal_pdf(x, I, alpha, beta, I_0)

        # Calculate the overal log likelihood
        log_likelihood = np.sum(np.log(cauchy_pdf_val)) + np.sum(np.log(log_normal_pdf_val))
        return log_likelihood + np.log(prior_alpha) + np.log(prior_beta) + np.log(prior_intensity_0)
    else:  
        # return a very small log posterior if alpha or beta are out of bounds
        return -np.inf

# Initialise parameters for emcee, nwalkers at least two times dimension
nwalkers, ndim = 100,3
nsteps = 10000

# Define args for the sampler and start positions
x = data[:,0]
I = data[:,1]
alpha_max = 10
alpha_min = -10
beta_max = 10
beta_min = 0
I_0max = 100
I_0min = 0.001

# # Intialise random starting positions for the walkers in the parameter space
start_positions = np.random.rand(nwalkers, ndim)

print(start_positions.shape)

# Ensure that the start positions are in the sample space
start_positions[:, 0] = start_positions[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions[:, 1] = start_positions[:, 1] * (beta_max - beta_min) + beta_min 
start_positions[:, 2] = start_positions[:, 2] * (I_0max - I_0min) + I_0min 

# Initialise & run sampler
sampler3 = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_intensity, args=[x, I, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min])
sampler3.run_mcmc(start_positions, nsteps)

# Get the chain
chain3 = sampler3.get_chain(flat=True)
print(chain3)

# Trace plot 
for i in range(chain3.shape[1]):  
    plt.plot(chain3[:, i], alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()

# Should this be doen after burn in though?
# Get autocorrelation times 
taus = sampler3.get_autocorr_time(tol=2)
print('Autocorrelation:', taus)

# Get the maximum autocorrelation time
tau = max(taus)
print(f"{tau = }")

# Effective sample size
ess = len(chain3)/tau
print("Effective sample size:", ess)

# To get iid samples, thin the chain, and discard the burn-in - question of dealing with burn-in and thinning
iid_samples = sampler3.get_chain(flat=True, thin=int(tau), discard=1000)
num_samples = len(iid_samples)
print(iid_samples.shape)

# Printing the number of iid samples and time per iid sample
print("I.i.d samples = {}".format(num_samples))

# Estimate the expectation with axis=0 to average over all samples for each parameter
mean_alpha = np.mean(iid_samples[:,0], axis=0)
mean_beta = np.mean(iid_samples[:,1], axis=0)
mean_intensity_0 = np.mean(iid_samples[:,2], axis=0)

# # Would think that SEM is better?
# sem_alpha = np.std(iid_samples[:,0])/np.sqrt(len(iid_samples))
# sem_beta = np.std(iid_samples[:,1])/np.sqrt(len(iid_samples))

# Think that SEM is better than stdev
stdev_alpha = np.std(iid_samples[:,0], ddof=1)
stdev_beta = np.std(iid_samples[:,1], ddof=1)
stdev_intensity_0 = np.std(iid_samples[:,2], ddof=1)

print(mean_alpha, stdev_alpha)
print(mean_beta, stdev_beta)
print(mean_intensity_0, stdev_intensity_0)

# Corner plot - plotting 1D marginalised histograms
fig = corner.corner(iid_samples, labels=["alpha", "beta", "intensity_0"], bins=50, plot_datapoints=True, plot_density=True, plot_contours=True, show_titles=True)
plt.show()

# Repeat running the MCMC chain to obtain the GR statistic
# Intialise random starting positions for the walkers in the parameter space
start_positions2 = np.random.rand(nwalkers, ndim)
# Ensure that the start positions are in the sample space
start_positions2[:, 0] = start_positions2[:, 0] * (alpha_max - alpha_min) + alpha_min 
start_positions2[:, 1] = start_positions2[:, 1] * (beta_max - beta_min) + beta_min 
start_positions2[:, 2] = start_positions2[:, 2] * (I_0max - I_0min) + I_0min 

# # Ensure that the start positions are in the sample space
# start_positions[:, 0] = 1 + 0.1 * np.random.randn(nwalkers)
# start_positions[:, 1] = 1 + 0.1 * np.random.randn(nwalkers)

# Initialise & run sampler
sampler4 = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_intensity, args=[x, I, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min])
sampler4.run_mcmc(start_positions2, nsteps)

# Get chain from second sampler
chain3 = sampler3.get_chain(flat=True, discard=1000)
chain4 = sampler4.get_chain(flat=True, discard=1000)

# Combine chains into a single chain
chains_with_I = np.array([chain3,chain4])
print(chain3.shape, chain4.shape, chains_with_I.shape)

# The Gelman-Rubin statistic
def gelman_rubin_I(samples):
    samples = samples[:,:,2]
    n_chains, n_samples = samples.shape
    # Compute column mean
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Mean of empirical variance between each chain
    W = np.mean(chain_variances)
    
    # Estimated variance of the target distribution
    sigma_sq = (((n_samples - 1)*W)/ n_samples) + (B / n_samples)

    # Potential Scale Reduction Factor for each parameter
    GR = np.sqrt(sigma_sq / W)
    return GR

GR_alpha = gelman_rubin_alpha(chains_with_I)
GR_beta = gelman_rubin_beta(chains_with_I)
GR_I = gelman_rubin_I(chains_with_I)
print(GR_alpha, GR_beta, GR_I)