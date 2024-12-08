import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot(sigmas_correlated, sigmas_uncorrelated,
         tau_add_deltas_correlated, tau_tighter_correlated, tau_exact_uncorrelated, tau_add_deltas_uncorrelated, meta_params):
    """
    Sets up plot for experiment 1
    :param sigmas_correlated:
    :param sigmas_uncorrelated:
    :param tau_add_deltas_correlated:
    :param tau_tighter_correlated:
    :param tau_exact_uncorrelated:
    :param tau_add_deltas_uncorrelated:
    :return:
    """
    max_sigma = sigmas_uncorrelated.max()

    sns.set_style("ticks")
    sns.color_palette("tab10")

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plots
    plt.plot(sigmas_correlated, tau_add_deltas_correlated, label='\\textbf{add-the-deltas \\emph{correlated} (this work)}')
    plt.plot(tau_tighter_correlated[0], tau_tighter_correlated[1], label='\\textbf{tighter analyis (this work)}')
    plt.plot(tau_exact_uncorrelated[0], tau_exact_uncorrelated[1], label='exact analysis [WKZK 24]')
    plt.plot(sigmas_uncorrelated, tau_add_deltas_uncorrelated, label='add-the-deltas \\emph{uncorrelated} [Google 20]')

    # Vertical lines
    plt.axvline(x=tau_exact_uncorrelated[2], linestyle="--")
    plt.axvline(x=tau_tighter_correlated[2], linestyle="--")

    # Marking Minimums
    points = []
    points  += [get_minimum(sigmas_correlated, tau_add_deltas_correlated)]
    points  += [get_minimum(tau_tighter_correlated[0], tau_tighter_correlated[1])]
    points  += [get_minimum(sigmas_uncorrelated, tau_add_deltas_uncorrelated)]
    points  += [get_minimum(tau_exact_uncorrelated[0], tau_exact_uncorrelated[1])]

    # Mark the minimum point
    for x,y in points:
        plt.scatter(x, y, color='gray', zorder=5, s=15)
        # plt.annotate(f'({y:.1f})', xy=(x,y), fontsize=10, color='gray', xycoords='figure pixels')

    # Axes
    plt.xlim((None, max_sigma))
    plt.xlabel('$\\sigma$')
    plt.xscale("linear")
    plt.ylabel('$1 + \\tau$')
    plt.title(f'Minimal $\\tau$ for $k={meta_params["k"]}$ for parameters $\delta=10^{{-5}}$ and $\\epsilon={meta_params["eps"]}$')
    plt.legend(fontsize='x-small', title_fontsize='40')
    plt.savefig(f'save/experiment1-k{meta_params["k"]}.pdf', format='pdf', bbox_inches='tight')

def get_minimum(xser, yser):
    min_idx = yser.index(np.nanmin(yser))
    return xser[min_idx], yser[min_idx]
