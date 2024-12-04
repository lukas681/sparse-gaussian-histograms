import math
from scipy.stats import norm
from main.util.funs import *

def threshold_add_the_delta(total_delta_budget, epsilon, k, sigma):
    """
    Given a target (eps, delta) guarantee, returns the threshold tau required for add-the-delta
    The function basically numerical solves the $delta = delta_inf + delta_gauss$ approach for $delta_inf$
    :param total_delta_budget: The total delta budget we can split between inf and gauss.
    :param epsilon
    :param delta:
    :param k: How many bad events can happen at the same time?
    :param sigma: noise level.
    :return:
    """
    mu = math.sqrt(k + math.sqrt(k))/(2*sigma) # Sensitivity + Scaling in gaussian part.
    tau = (1+k**(-1/4)) * sigma * norm.ppf(
        (1 - total_delta_budget + analytic_gaussian(epsilon, mu)) ** (1 / (k + 1))
    )
    return tau

def cgshm_tighter(k, sigma, tau, epsilon):
    """
    our approach as described in the paper.
    :param k: How many non-zeroes can change?
    :param sigma:
    :param tau_diff:
    :param epsilon:
    :return:
    """
    psi = lambda m:norm.cdf(tau/(1+k**(-1/4)*sigma))**(m+1)
    gamma = lambda j: min(math.sqrt(j), math.sqrt(j+math.sqrt(k))/2)

    case_one = 1 - psi(k)
    case_two = analytic_gaussian(epsilon, math.sqrt(k + math.sqrt(k))/2)
    case_three = max([1 - psi(k-j) +analytic_gaussian(epsilon, gamma(j)/sigma)                         for j in range(1, k+1)])
    case_four =  max([1 - psi(k-j) + analytic_gaussian(epsilon + math.log(1- psi(j+1)), gamma(j)/sigma) for j in range(1, k+1)])

    return max(case_one, case_two, case_three, case_four)

def check_validity(k, sigma, tau, epsilon, delta):
    """
    Checks whether the given parameters satisfy (eps, delta)-dp guarantees for our CGSHM
    :param k:
    :param sigma:
    :param tau:
    :param epsilon:
    :param delta:
    :return:
    """
    return cgshm_tighter(k, sigma, tau, epsilon) <= delta

def minimum_amount_of_noise(candidate_mu, epsilon, delta, number_iterations=500):
    """
    As the previous work, we use newtons method to find a minimum amount of noise required such that the part of the Gaussian mechanism fulfills our privacy guarantees
    :param k: number of counts to be changed
    :param epsilon:
    :param delta: The delta we would like to get.
    :return: the minimum amount of noise required to satisfy the (eps, delta) guarantee.
    """
    mu = candidate_mu
    for i in range(number_iterations):
        f = norm.cdf(mu/2 - epsilon/mu) - np.e**epsilon * norm.cdf(-mu/2 - epsilon/mu) - delta
        f_derivative = ((norm.pdf(mu/2 - epsilon/mu)) * (1/2 + epsilon/(mu**2))
                        - np.exp(epsilon) * norm.pdf(-mu/2 - epsilon/mu) * (-1/2 + epsilon / (mu**2)))
        mu = mu - f/f_derivative
    return mu

def compute_tau(k, sigma, delta):
    """
    Computing the tau value for our mechanism is a bit more intricate.

    :param k:
    :param sigma:
    :param delta:
    :return:
    """
    return norm.ppf(1 - delta)**(1/k) * sigma
def compute_threshold_tighter(delta, epsilon, k, datapoints = 10):
    """
    Returns the treshold for the infinite privacy loss event part.
    We skip the mixed case here as it should not make any difference.
    :param k:
    :param delta:
    :param sigma:
    :return:
    """
    return None # TODO
    # return norm.ppf((1-delta)**(1/k)) * sigma
