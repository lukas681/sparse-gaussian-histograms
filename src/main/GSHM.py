import numpy as np
from scipy.stats import norm

# global funcs
delta_analytic_gaussian = \
    lambda epsilon, mu :  norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
delta_bw = \
    lambda epsilon, mu: norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
tau_diff_func = \
    lambda delta, c_u, sigma: norm.ppf((1 - delta) ** (1 / c_u)) * sigma

def gshm_delta(c_u, sigma, tau_diff, epsilon):
    """
    Implements the exact analysis,
    :param c_u:
    :param sigma:
    :param tau_diff:
    :param epsilon:
    :return:
    """
    part_two = np.zeros(c_u)
    part_three = np.zeros(c_u)

    for i in range(1, c_u + 1):
        a_eq = i - 1
        mu = np.sqrt(c_u - a_eq) / sigma # p. 12
        epsilon2 = \
            epsilon - a_eq * np.log(norm.cdf(tau_diff / sigma)) # page 9
        epsilon3 = \
            epsilon + a_eq * np.log(norm.cdf(tau_diff / sigma)) # page 9

        part_two[i-1] = \
            1 - norm.cdf(tau_diff / sigma)**a_eq + norm.cdf(tau_diff / sigma)**a_eq * delta_analytic_gaussian(epsilon2, mu)
        part_three[i - 1] = delta_analytic_gaussian(epsilon3, mu)
    return [np.repeat(1 - norm.cdf(tau_diff / sigma)**c_u,c_u),
            part_two,
            part_three]