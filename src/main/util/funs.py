from scipy.stats import norm
import numpy as np

# Globally used functions.
analytic_gaussian = lambda epsilon, mu :  norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
delta_bw = lambda epsilon, mu: norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
tau_diff_func = lambda delta, c_u, sigma: norm.ppf((1 - delta) ** (1 / c_u)) * sigma