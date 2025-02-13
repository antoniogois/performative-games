import os
import argparse
import torch
from torch import nn
from utils import get_torch_device, read_precomputed, store_precomputed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
from performative_game.game import CRD
from train import train
from performative_game.predicted_population.population import LatticePopulation, ScaleFreePopulation
from performative_game.coop_predictor import Predictor, ProxyGroupLoss


def plot_labeled_pairs(points_list, plot_best_only=True, multiobj_pareto_thres5=None):
    """
    Plot pairs of 2-D points with lines connecting the points with the highest values for each 'T' value.

    Parameters:
    - points_list: List of tuples, where each tuple contains two 2-D points and a corresponding 'T' value.
    """
    t_values = [pair[2] for pair in points_list]

    # Create a colormap and normalize the 'T' values
    norm = Normalize(vmin=min(t_values), vmax=max(t_values))
    cmap = plt.get_cmap('viridis')
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)

    legend_added = {}  # Dictionary to track added legends
    plt.figure(figsize=(6.4, 3.3))  # default: figsize=(6.4, 4.8)

    for t_value in set(t_values):
        filtered_points = [pair for pair in points_list if pair[2] == t_value]

        # Find the points with the highest values for each 'T' value
        best_welfare = max(filtered_points, key=lambda pair: pair[0][0])[0]  # best social-welfare
        best_acc = max(filtered_points, key=lambda pair: pair[1][1])[1]  # best accuracy

        # x_values = [point[0][0] for point in filtered_points] + [point[1][0] for point in filtered_points]
        # y_values = [point[0][1] for point in filtered_points] + [point[1][1] for point in filtered_points]

        color = scalar_map.to_rgba(t_value)  # Assign color based on 'T' value

        if not plot_best_only:
            # Plot points_list[i][0] as squares  (all runs optimizing welfare)
            plt.scatter([point[0][0] for point in filtered_points], [point[0][1] for point in filtered_points],
                        marker='s', s=50, label=None, color=color, alpha=0.7)
            # Plot points_list[i][1] as circles  (all runs optimizing accuracy)
            plt.scatter([point[1][0] for point in filtered_points], [point[1][1] for point in filtered_points],
                        marker='o', s=50, label=None, color=color, alpha=0.7)
        else:
            # Plot (best run optimizing welfare)
            plt.scatter(best_welfare[0], best_welfare[1],
                        marker='s', s=50, label=None, color=color, alpha=0.7)
            # Plot (best run optimizing accuracy)
            plt.scatter(best_acc[0], best_acc[1],
                        marker='o', s=50, label=None, color=color, alpha=0.7)


        # Connect the points with the highest values
        if t_value==.5 and multiobj_pareto_thres5 is not None:
            # Plot points for current task
            plt.plot(multiobj_pareto_thres5[:, 0], multiobj_pareto_thres5[:, 1], 'X', markersize=10, color=color)
            # Connect points with lines along the x-axis
            all_points = np.concatenate(([best_welfare, best_acc], multiobj_pareto_thres5))
            sorted_all = all_points[all_points[:, 0].argsort()]
            plt.plot(sorted_all[:, 0], sorted_all[:, 1], color=color, linestyle='-')
        else:
            plt.plot([best_welfare[0], best_acc[0]], [best_welfare[1], best_acc[1]], linestyle='-', color=color, linewidth=1)

        # # Label points as 'C' and 'A'
        # plt.text(best_welfare[0], best_welfare[1], 'C', ha='right', va='bottom', fontweight='bold')
        # plt.text(best_acc[0], best_acc[1], 'A', ha='left', va='top', fontweight='bold')

        # Add legend only if not already added for this 'T' value
        if t_value not in legend_added:
            legend_added[t_value] = True
            scalar_map.set_array([])  # Dummy data for the colorbar

    # # Place the legend outside the loop for points
    # plt.legend(loc='upper right', title='T', bbox_to_anchor=(1.25, 1))

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Welfare maximization',
               markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Accuracy maximization',
               markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='X', color='w', label='Multi-objective loss',
               markerfacecolor='gray', markersize=12)
    ]

    # Add the custom legend to the plot
    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(.75, 0))

    # Add colorbar legend
    plt.colorbar(scalar_map, label='Threshold T')

    plt.xlabel('% of successful groups')  # Updated xlabel
    plt.ylabel('Accuracy')  # Updated ylabel
    plt.title('Cooperation-accuracy tradeoff',  # \neach model was initialized with previous one by accident, in the order:\nthres 0.2: {seed 0: [group success, accuracy]; seed 1: [...]}; thres 0.3: {...}',
              size='medium')  # Updated title

    # Set axis limits
    plt.xlim(0, 1.1)
    plt.ylim(0.39, 1.02)

    plt.grid(True)
    plt.savefig('plot_tradeoff_thresholds.pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # # Example usage:
    # points_list = [((0.2, 0.3), (0.7, 0.8), 1),
    #                ((0.1, 0.5), (0.9, 0.4), 2),
    #                ((0.3, 0.2), (0.6, 0.6), 3)]
    #
    # plot_labeled_pairs(points_list)


def pick_best_pareto(paretos):
    """
    :param paretos: shape = (seed_id, 2Dpoint_id, axis_id), welfare_value is axis_id=0, accuracy is axis_id=1
    :return: id of seed with pareto containing best point, for criterion welfare*accuracy
    """
    # "scores" contains best welfare*accuracy, per seed
    scores = [max([p[0] * p[1]
                   for p in paretos[seed][np.argsort(paretos[seed][:, 0])]])
              for seed in range(len(paretos))]
    best_seed = np.argmax(scores)
    return best_seed

def get_pareto_thres5(args, pop, architectures, device, thres=0.5):
    paretos = []
    for s in range(args.seed, args.seed + args.n_trials):
        print(f"seed: {s}")
        loss = "multitask"
        print(f"loss: {loss}")
        crd = CRD(e=args.endowment,
                  r=args.risk,
                  c=args.coop_cost,
                  t=thres)
        loss_fn = {
            "group": ProxyGroupLoss(crd),
            "accuracy": nn.BCELoss()
        }
        predictor = Predictor(*architectures[args.architecture], device=device).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
        full_stats = read_precomputed(thres, loss, s, args)
        if full_stats is None:
            args.loss = loss
            predictor, full_stats = train(args,
                                          pop,
                                          predictor,
                                          crd,
                                          loss_fn,
                                          optimizer)
            store_precomputed(full_stats, thres, loss, s, args)
        paretos.append(full_stats.get_pareto())
    best_seed = pick_best_pareto(paretos)
    best_pareto = paretos[best_seed]
    return best_pareto

def main(args):
    torch.manual_seed(args.seed)
    device = get_torch_device(ignore_mps=True)
    torch.use_deterministic_algorithms(True)  # may compromise speed

    if args.topology == "lattice":
        pop = LatticePopulation(side=4)
    elif args.topology == "scale-free":
        pop = ScaleFreePopulation(seed=args.seed)
    else:
        raise NotImplementedError

    architectures = {
        "mlp": (pop, "mlp"),
        "gnn": (pop, "gnn", False, None),  # pop, architecture, random_gnn_graph, gnn_add_layer
        "gnn+linear": (pop, "gnn", False, "linear"),
        # "gnn+linear+rand_graph": (pop, "gnn", True, "linear")
        "gnn+mlp": (pop, "gnn", False, "mlp")
    }

    points = []
    for thres in args.thresholds:
        print(f"threshold: {thres}")
        crd = CRD(e=args.endowment,
                  r=args.risk,
                  c=args.coop_cost,
                  t=thres)
        for s in range(args.seed, args.seed + args.n_trials):
            print(f"seed: {s}")
            points.append([])
            for loss in ["group", "accuracy"]:
                print(f"loss: {loss}")
                if loss == "group":
                    loss_fn = ProxyGroupLoss(crd)
                elif loss == "accuracy":
                    loss_fn = nn.BCELoss()
                else:
                    raise NotImplementedError
                predictor = Predictor(*architectures[args.architecture], device=device).to(device)
                optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

                full_stats = read_precomputed(thres, loss, s, args)
                if full_stats is None:
                    torch.manual_seed(s)
                    args.loss = loss
                    predictor, full_stats = train(args,
                                                  pop,
                                                  predictor,
                                                  crd,
                                                  loss_fn,
                                                  optimizer)
                    store_precomputed(full_stats, thres, loss, s, args)
                points[-1].append(full_stats.get_best())
            points[-1].append(thres)
    best_pareto_curve = get_pareto_thres5(args, pop, architectures, device)
    plot_labeled_pairs(points, plot_best_only=True, multiobj_pareto_thres5=best_pareto_curve)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="maximize accuracy or coop, for multiple thresholds")

    game = parser.add_argument_group("game parameters")
    game.add_argument("-e", "--endowment", type=float, default=1, help="initial endowment per agent")
    game.add_argument("-r", "--risk", type=float, default=.4, help="risk of losing everything,"
                                                                   "if threshold not achieved")
    game.add_argument("-c", "--coop_cost", type=float, default=.2, help="individual cost of cooperating")
    game.add_argument("-t", "--thresholds", nargs='+', type=float, default=[.2, .3, .4, .5, .6, .7, .8],
                      help="different values for fraction of agents required in group, in order to achieve success")

    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument("--architecture", type=str, choices={"mlp", "gnn", "gnn+linear", "gnn+mlp"}, default="mlp")
    misc.add_argument("--topology", type=str, choices={"lattice", "scale-free"}, default="scale-free")

    opt = parser.add_argument_group("optimization")
    opt.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    opt.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    opt.add_argument("--n_rounds", type=int, default=20, help="rounds per epoch")
    opt.add_argument("--seed", type=int, default=0)
    # opt.add_argument("--loss", type=str, choices={"individual", "group", "accuracy"}, default="group",
    #                  help="when training predictor, group loss maximizes successful groups "
    #                              "whereas individual loss maximizes each agent's prob of cooperating ignoring "
    #                              "topology")

    plot = parser.add_argument_group("plot", description="plot tradeoff between accuracy and group success, "
                                                               "for various threshold values")
    plot.add_argument("--n_trials", type=int, default=3, help="get average of n_trials per threshold value")
    plot.add_argument("--plots_path", type=str, default="figs", help="name of folder to store visualizations")
    plot.add_argument("--stats_path", type=str, default="crd_stats", help="name of folder to store visualizations")

    args = parser.parse_args()

    if not os.path.exists(args.plots_path):
        os.makedirs(args.plots_path)
    if not os.path.exists(args.stats_path):
        os.makedirs(args.stats_path)
    args.log_wandb = False

    main(args)
