import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import read_precomputed


def plot_bars(data, maximize='group'):
    architectures = sorted(data.keys())
    welfare_w_means = []
    welfare_w_errors = []
    welfare_a_means = []
    welfare_a_errors = []
    for arch in architectures:
        welfare_w_means.append(np.mean([d[0] for d in data[arch][maximize]]))
        welfare_w_errors.append(np.std([d[0] for d in data[arch][maximize]]))
        welfare_a_means.append(np.mean([d[1] for d in data[arch][maximize]]))
        welfare_a_errors.append(np.std([d[1] for d in data[arch][maximize]]))

    n_architectures = len(architectures)
    bar_width = 0.35

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6.4, 3.))  # default values (6.4, 4.8); possible to reduce img vertically if needed

    # Positions of the bars on the x-axis
    x = np.arange(n_architectures)

    # Plot accuracy bars
    bars1 = ax.bar(x - bar_width / 2, welfare_w_means, bar_width, yerr=welfare_w_errors,
                   capsize=5, label='% of successful groups', hatch='//', color='c', alpha=0.7)

    # Plot welfare bars
    bars2 = ax.bar(x + bar_width / 2, welfare_a_means, bar_width, yerr=welfare_a_errors,
                   capsize=5, label='Accuracy', hatch='\\', color='y', alpha=0.7)

    # Add labels, title, and legend
    # ax.set_xlabel('Architectures')
    # ax.set_ylabel('Values')
    # ax.set_title('Optimizing social welfare with different architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend(loc='lower right', framealpha=1)

    # Show plot
    plt.savefig('architectures.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    thres = 0.5
    # losses = {'group', 'accuracy'}
    loss = args.loss
    seeds = range(0, 4)
    lr = 0.0001
    epochs = 200

    all_stats = {
        'gnn+linear': {'group': [], 'accuracy': []},
        'gnn+mlp': {'group': [], 'accuracy': []},
        'gnn': {'group': [], 'accuracy': []},
        'mlp': {'group': [], 'accuracy': []},
    }
    for seed in seeds:
        for arch in all_stats.keys():
            stats = read_precomputed(thres, loss, seed, architecture=arch, lr=lr, epochs=epochs, path=args.read_path)
            assert stats is not None, f"haven't trained yet for {thres, loss, seed, arch, lr, epochs}, please use 'train_all_archs.py' first."

            if loss == 'group':
                s = stats.get_best_group()
            elif loss == 'accuracy':
                s = stats.get_best_acc()
            else:
                raise ValueError(f'Loss {loss} not recognized')
            all_stats[arch][loss].append(s)
    plot_bars(all_stats, maximize=loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default="group", choices=['group', 'accuracy'])
    parser.add_argument("--save_fig", action='store_true', help="store figure as pdf")
    parser.add_argument("--read_path", type=str, help="path to precomputed data", default="crd_stats")
    args = parser.parse_args()

    main()
