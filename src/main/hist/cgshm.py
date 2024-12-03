import math
from scipy.stats import norm
import main.util.funs as fk

def cgshm_add_the_delta(total_delta_budget, epsilon, k, sigma):
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
        (1 - total_delta_budget + fk.analytic_gaussian(epsilon, mu)) ** (1 / (k + 1))
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
    y = lambda j: min(math.sqrt(j), math.sqrt(j+math.sqrt(k))/2)
    mu = math.sqrt(k + math.sqrt(k))/2

    # Case 1
    case_one = 1 - psi(k)
    case_two = fk.analytic_gaussian(epsilon, mu)
    case_three = max([1 - psi(k-j) + fk.analytic_gaussian(epsilon, y(j)/sigma) for j in range(1, k)])
    case_four =  max([1 - psi(k-j) + fk.analytic_gaussian(epsilon + math.log(1- psi(j+1)), y(j)/sigma) for j in range(1, k)])
    return [
        case_one, case_two, case_three, case_four
    ]