import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import read_precomputed

def plot_ablations(data):
    # Names for the experiments
    experiment_names = ['Full Gradient', 'Block Trust', 'Block Previous Trust']
    colors = ['blue', 'green', 'red']  # Define different colors for each experiment

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Iterate through each experiment
    for i, experiment in enumerate(data):
        experiment_array = np.array(experiment)
        mean_values = np.mean(experiment_array, axis=0)
        std_values = np.std(experiment_array, axis=0)

        plt.plot(range(200), mean_values, label=experiment_names[i], color=colors[i])

        plt.fill_between(range(200), mean_values - std_values, mean_values + std_values,
                         color=colors[i], alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Proportion of Cooperators')
    # plt.title('Comparison of Experiment Results with Standard Deviation')
    plt.xlim([0, 200])
    plt.ylim([0, .8])

    plt.legend()

    plt.savefig('ablation.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main(args):
    thres = 0.5
    arch = "gnn+mlp"
    loss = "individual"
    epochs = 200
    lr = 1e-4
    # expects folders with these names
    ablations = ["crd_stats_grad-full", "crd_stats_grad-blocktrust", "crd_stats_grad-blockprevtrust"]

    all_runs = []
    for ablation in ablations:
        all_runs.append([])
        ablation_path = Path(ablation)
        for seed in range(args.n_seeds):
            stats = read_precomputed(thres, loss, seed, architecture=arch, lr=lr, epochs=epochs, path=ablation_path)
            assert stats is not None, f"haven't trained yet for {thres, loss, seed, arch, lr, epochs}, please use 'train.py' first."
            run = stats.get_track_agents()
            all_runs[-1].append(run)
    plot_ablations(all_runs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5, help="number of seeds to use")

    args = parser.parse_args()
    main(args)