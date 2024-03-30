import numpy as np


# Def the cauchy PDF
def cauchy_pdf(x, alpha, beta):
    """
    @brief Cauchy PDF function
    @param x: The location data where flashes are recieved by detectors
    @param alpha: The lighthouse position along the coastline
    @param beta: The distance of the lighthouse out to sea
    @return The Cauchy PDF
    """
    numerator = beta
    # Broadcast by enabling x to have two more dimensions
    denominator = np.pi * (beta**2 + ((x[:, None, None] - alpha) ** 2))
    return numerator / denominator


# Def the log posterior
def log_posterior(params, x, alpha_max, alpha_min, beta_max, beta_min):
    """
    @brief Log posterior function for the Cauchy distribution
    @param params: The parameters of the Cauchy distribution
    @param x: The location data where flashes are recieved by detectors along the coastline
    @param alpha_max: The maximum value of alpha
    @param alpha_min: The minimum value of alpha
    @param beta_max: The maximum value of beta
    @param beta_min: The minimum value of beta
    """
    alpha, beta = params
    if alpha_min < alpha < alpha_max and beta_min < beta < beta_max:
        prior_alpha = 1 / (alpha_max - alpha_min)
        prior_beta = 1 / (beta_max - beta_min)
        cauchy_pdf_val = cauchy_pdf(x, alpha, beta)
        # Multiply the cauchy_pdf_val for each value of x
        likelihood = np.prod(cauchy_pdf_val, axis=0)
        return np.log(likelihood) + np.log(prior_alpha) + np.log(prior_beta)
    else:
        # return a very small log posterior if alpha or beta are out of bounds
        return -np.inf


# Prior for intensity
def prior_I_0(I_0, I_0max, I_0min):
    """
    @brief Prior for intensity
    @param I_0: The intensity of the lighthouse
    @param I_0max: The maximum intensity of the lighthouse
    @param I_0min: The minimum intensity of the lighthouse
    @return The prior for the intensity of the lighthouse"""
    return 1 / (I_0 * np.log(I_0max / I_0min))


# vii
# Log-norm likelihood
def log_normal_pdf(x, Intensity, alpha, beta, I_0):
    """
    @brief Log normal PDF function
    @param x: The location data where flashes are recieved by detectors along the coastline
    @param I: The intensity of the lighthouse
    @param alpha: The lighthouse position along the coastline
    @param beta: The distance of the lighthouse out to sea
    @param I_0: The intensity of the lighthouse
    @return The log normal PDF"""
    d_squared = (beta**2) + ((x - alpha) ** 2)
    mu = np.log(I_0 / d_squared)
    sigma = 1
    numerator = np.exp(-((np.log(Intensity) - mu) ** 2) / (2 * (sigma**2)))
    denominator = Intensity * np.sqrt(2 * np.pi * (sigma**2))
    return numerator / denominator


# Def the log posterior
def log_posterior_intensity(
    params, x, Intensity, alpha_max, alpha_min, beta_max, beta_min, I_0max, I_0min
):
    """
    @brief Log posterior function for the Cauchy distribution
    @param params: The parameters of the Cauchy distribution
    @param x: The location data where flashes are recieved by detectors along the coastline
    @param I: The intensity of the lighthouse
    @param alpha_max: The maximum value of alpha
    @param alpha_min: The minimum value of alpha
    @param beta_max: The maximum value of beta
    @param beta_min: The minimum value of beta
    @param I_0max: The maximum intensity of the lighthouse
    @param I_0min: The minimum intensity of the lighthouse
    """
    alpha, beta, I_0 = params
    if (
        alpha_min < alpha < alpha_max
        and beta_min < beta < beta_max
        and I_0min < I_0 < I_0max
    ):
        prior_alpha = 1 / (alpha_max - alpha_min)
        prior_beta = 1 / (beta_max - beta_min)
        prior_intensity_0 = prior_I_0(I_0, I_0max, I_0min)
        cauchy_pdf_val = cauchy_pdf(x, alpha, beta)
        log_normal_pdf_val = log_normal_pdf(x, Intensity, alpha, beta, I_0)

        # Calculate the overal log likelihood
        log_likelihood = np.sum(np.log(cauchy_pdf_val)) + np.sum(
            np.log(log_normal_pdf_val)
        )
        return (
            log_likelihood
            + np.log(prior_alpha)
            + np.log(prior_beta)
            + np.log(prior_intensity_0)
        )
    else:
        # return a very small log posterior if alpha or beta are out of bounds
        return -np.inf
