import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from performative_game.predicted_population.population import FullyConnectedPopulation, ScaleFreePopulation
from performative_game.game import CRD


def compute_anarchic_success(alpha=.5, n=5, t=.5, graph="full", seed=None, m=2):
    """
    compute % of successful groups overcoming threshold for a given alpha, in an anarchic population
    """
    crd = CRD(e=1, r=.6, c=.1, t=t)
    if graph == "full":
        pop = FullyConnectedPopulation(n=n, prior_trust=0, alpha=alpha)
    elif graph == "scale-free":
        pop = ScaleFreePopulation(n=n, prior_trust=0, alpha=alpha, seed=seed, m=m)
    else:
        raise ValueError(f'graph {graph} not recognized')
    for node in pop.G.nodes:
        node.last_prediction = torch.tensor(.5)  # initialize whichever predictions, which will be ignored
    pop.play_all_rational(crd)
    successes = pop.get_group_successes(crd)
    return sum(successes).item() / len(successes)

def plot_heatmap_alpha(line_granularity=100, max_n=25, t_frac=0.5, graph="scale-free"):
    alpha_values = np.linspace(0, 1, line_granularity)
    rows = []
    n_values = np.unique(np.linspace(3, max_n, line_granularity).astype(int))

    for n in n_values:

        y = np.zeros_like(alpha_values)
        for i in range(len(alpha_values)):
            if graph == "full":
                y[i] = compute_anarchic_success(alpha=alpha_values[i], n=n, t=t_frac)
            elif graph == "scale-free":
                trials = []
                for seed in range(5):
                    trials.append(compute_anarchic_success(alpha=alpha_values[i],
                                                           n=n,
                                                           t=t_frac,
                                                           graph="scale-free",
                                                           seed=seed,
                                                           m=2))
                y[i] = sum(trials)/len(trials)
            else:
                raise ValueError(f'graph {graph} not recognized')

        rows.append(y)
    data = np.array(rows)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, aspect='auto', cmap='viridis', origin='lower',
                    extent=[alpha_values[0], alpha_values[-1], n_values[0], n_values[-1]])
    if graph == "scale-free":
        cbar = plt.colorbar(im)
        cbar.set_label('% Successful Groups', fontsize=30)
        cbar.ax.tick_params(labelsize=23)

    # Select a subset of ticks (e.g., every nth element)
    skip = max(1, len(n_values) // 7)
    selected_n_values = n_values[::skip]
    plt.yticks(np.linspace(n_values[0], n_values[-1], len(selected_n_values)), selected_n_values, fontsize=23)
    if graph == "full":
        plt.ylabel('Population Size', fontsize=40)
    if graph == "scale-free":
        plt.tick_params(axis='y', which='major', left=False, right=False, labelleft=False, labelright=False)
    plt.xticks(fontsize=23)
    plt.xlabel(r'$\alpha$', fontsize=50)


    # plt.title(r'% of successful groups in fully-connected anarchic population, by N and $\alpha$, for T=.5')
    # plt.title(r'% of successful groups in a scale-free (m=2) anarchic population, by N and $\alpha$, for T=.5')
    plt.savefig(f'alpha_{graph}-graph.pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default="scale-free", choices={"full", "scale-free"})
    args = parser.parse_args()

    plot_heatmap_alpha(graph=args.graph)
