import math
import numpy as np
from scipy.stats import norm
from src.main.util.funs import *

def gshm_exact(c_u, sigma, tau_diff, epsilon):
    """
    Implements the exact analysis due to [Wilkins et al.]
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

        part_two[i-1] = 1 - norm.cdf(tau_diff / sigma) ** a_eq + norm.cdf(tau_diff / sigma) ** a_eq * analytic_gaussian(epsilon2, mu)
        part_three[i - 1] = analytic_gaussian(epsilon3, mu)
        part_one[i-1] = 1 - norm.cdf(tau_diff / sigma)**c_u
        maximum[i-1] = np.max([part_one[i-1], part_two[i-1], part_three[i-1]])
    return [
            part_one,
            part_two,
            part_three,
            maximum,
    ]

def gshm_add_the_deltas(total_delta_budget, epsilon, k, sigma):
    """
    TODO is this complete?
    Computes the add-the-Deltas as in the paper
    "EXACT PRIVACY ANALYSIS OF THE GAUSSIAN SPARSE HISTOGRAM MECHANISM"
    :param total_delta_budget:
    :param epsilon:
    :param k:
    :param sigma:
    :return:
    """
    mu = math.sqrt(k) / sigma
    return sigma * norm.ppf(
        (1 - total_delta_budget + analytic_gaussian(epsilon, mu)) ** (1 / k)
    )
