from scipy.stats import norm
import numpy as np

def analytic_gaussian(epsilon, mu): return norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
def delta_bw(epsilon, mu): return  norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
def tau_diff_func(delta, c_u, sigma): return norm.ppf((1 - delta) ** (1 / c_u)) * sigma