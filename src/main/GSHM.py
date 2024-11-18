import math
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
    Implements the exact analysis.
    :param c_u:
    :param sigma:
    :param tau_diff:
    :param epsilon:
    :return:
    """

    part_one, part_two, part_three, maximum = np.zeros(c_u), np.zeros(c_u), np.zeros(c_u), np.zeros(c_u)

    for i in range(1, c_u + 1):
        a_eq = i - 1
        mu = np.sqrt(c_u - a_eq) / sigma # p. 12
        epsilon2 = epsilon - a_eq * np.log(norm.cdf(tau_diff / sigma)) # page 9
        epsilon3 = epsilon + a_eq * np.log(norm.cdf(tau_diff / sigma)) # page 9

        part_two[i-1] = \
            1 - norm.cdf(tau_diff / sigma)**a_eq + norm.cdf(tau_diff / sigma)**a_eq * delta_analytic_gaussian(epsilon2, mu)
        part_three[i - 1] = delta_analytic_gaussian(epsilon3, mu)
        part_one[i-1] = 1 - norm.cdf(tau_diff / sigma)**c_u
        maximum[i-1] = np.max([part_one[i-1], part_two[i-1], part_three[i-1]])
        #logg.debug("k={}, 1: {}, 2: {} 3: {} MAX: {}".format(i, part_one[i-1], part_two[i-1], part_three[i-1], maximum[i-1]))
    return [
            part_one,
            part_two,
            part_three,
            maximum,
    ]
def compute_tau_add_deltas_standard(total_delta_budget, epsilon, k, sigma):
    """
    Computes the add-the-Deltas as in the paper
    "EXACT PRIVACY ANALYSIS OF THE GAUSSIAN SPARSE HISTOGRAM MECHANISM"
    :param total_delta_budget:
    :param epsilon:
    :param k:
    :param sigma:
    :return:
    """

    mu = math.sqrt(k) / sigma ## = l2-sensitivity + ???
    return sigma * norm.ppf(
        (1- total_delta_budget + delta_analytic_gaussian(epsilon, mu))**(1/k)
    )
def compute_tau_add_deltas_correlated(total_delta_budget, epsilon, k, sigma):
    """
    Given a target (eps, delta) guarantee, returns the threshold tau required for add-the-delta
    The function basically numerical solves the $delta = delta_inf + delta_gauss$ approach for $delta_inf$

    Although sigma is the amount of noise we add

    :param total_delta_budget: The total delta budget we can split between inf and gauss.
    :param epsilon
    :param delta:
    :param k: How many bad events can happen at the same time?
    :param sigma: noise level.
    :return:
    """

    # I guess mu must contain sigma as well. Check why.
    # I guess we have another sigma here as well.
    mu = math.sqrt(k + math.sqrt(k))/2/sigma ## TODO: Verify the root k here.
    delta_gauss  = delta_analytic_gaussian(epsilon, mu)
    tau = (1+k**(-1/4)) * sigma * norm.ppf(
        (1 - total_delta_budget + delta_analytic_gaussian(epsilon, mu))**(1/(k+1))
    )
    return tau