import os
import argparse
import torch
from torch import nn
from utils import get_torch_device, read_precomputed, store_precomputed
import matplotlib.pyplot as plt
from performative_game.game import CRD
from train import train
from performative_game.predicted_population.population import LatticePopulation, ScaleFreePopulation
from performative_game.coop_predictor import Predictor, ProxyGroupLoss


def plot_lines(data, y_axis, loss, seed):
    fig, ax = plt.subplots()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # colors for lines

    for i, (key, value) in enumerate(data.items()):
        track_groups = value.get_track_groups()
        track_acc = value.get_track_acc()
        track_loss = value.get_track_loss()

        if y_axis == "group":
            points = track_groups
            plt.ylabel('% successful groups')
        elif y_axis == "accuracy":
            points = track_acc
            plt.ylabel('avg accuracy per round')
        elif y_axis == "loss":
            points = [i*-1 for i in track_loss]
            plt.ylabel('loss')
        else:
            raise NotImplementedError

        color = colors[i % len(colors)]  # pick a unique color for each entry

        ax.plot(points, color=color, linestyle='-', label=f'{key}')
        # ax.plot(track_groups, color=color, linestyle='-', label=f'{key}')  # - Track Groups')
        # ax.plot(track_acc, color=color, linestyle='--', label=f'{key}')  # - Track Acc')

    ax.legend(loc='lower right')
    plt.xlabel('training step')

    # plt.ylabel('avg accuracy per round')
    if loss == "group":
        plt.title(f'Maximizing group success (seed={seed})')
    elif loss == "accuracy":
        plt.title('Maximizing accuracy')
    else:
        raise NotImplementedError
    plt.grid(True)
    if y_axis != "loss":
        ax.set_ylim([None, 1.01])  # set upper limit of the vertical axis
    plt.show()


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

    crd = CRD(e=args.endowment,
              r=args.risk,
              c=args.coop_cost,
              t=args.threshold)

    if args.loss == "group":
        loss_fn = ProxyGroupLoss(crd)
    elif args.loss == "accuracy":
        loss_fn = nn.BCELoss()
    elif args.loss == "multitask":
        loss_fn = {
            "group": ProxyGroupLoss(crd),
            "accuracy": nn.BCELoss()
        }
    else:
        raise NotImplementedError

    all_stats = {}


    architectures = {
        "mlp": (pop, "mlp"),
        "gnn": (pop, "gnn", False, None),  # pop, architecture, random_gnn_graph, gnn_add_layer
        "gnn+linear": (pop, "gnn", False, "linear"),
        # "gnn+linear+rand_graph": (pop, "gnn", True, "linear")
        "gnn+mlp": (pop, "gnn", False, "mlp")
    }
    for arch in architectures.keys():
        print(arch, flush=True)
        full_stats = read_precomputed(args.threshold, args.loss, args.seed, args, architecture=arch)
        if full_stats is None:
            predictor = Predictor(*architectures[arch], device=device).to(device)
            optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
            predictor, full_stats = train(args,
                                          pop,
                                          predictor,
                                          crd,
                                          loss_fn,
                                          optimizer)
            store_precomputed(full_stats, args.threshold, args.loss, args.seed, args, architecture=arch)
        all_stats[arch] = full_stats
    plot_lines(all_stats, args.y_axis, args.loss, args.seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    game = parser.add_argument_group("game parameters")
    game.add_argument("-e", "--endowment", type=float, default=1, help="initial endowment per agent")
    game.add_argument("-r", "--risk", type=float, default=.4, help="risk of losing everything,"
                                                                   "if threshold not achieved")
    game.add_argument("-c", "--coop_cost", type=float, default=.2, help="individual cost of cooperating")
    game.add_argument("-t", "--threshold", type=float, default=.5,
                      help="fraction of agents required in group, in order to achieve success")

    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument("--topology", type=str, choices={"lattice", "scale-free"}, default="scale-free")

    opt = parser.add_argument_group("optimization")
    opt.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    opt.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    opt.add_argument("--n_rounds", type=int, default=20, help="rounds per epoch")
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--loss", type=str, choices={"individual", "group", "accuracy"}, default="group",
                     help="when training predictor, group loss maximizes successful groups "
                                 "whereas individual loss maximizes each agent's prob of cooperating ignoring "
                                 "topology")

    plot = parser.add_argument_group("plot", description="plot tradeoff between accuracy and group success, "
                                                               "for various threshold values")
    plot.add_argument("--n_trials", type=int, default=1, help="get average of n_trials per threshold value")
    plot.add_argument("--plots_path", type=str, default="figs", help="name of folder to store visualizations")
    plot.add_argument("--stats_path", type=str, default="crd_stats", help="name of folder to store visualizations")
    plot.add_argument("--y_axis", type=str, default="group", choices={"group", "accuracy", "loss"})

    args = parser.parse_args()

    if not os.path.exists(args.plots_path):
        os.makedirs(args.plots_path)
    if not os.path.exists(args.stats_path):
        os.makedirs(args.stats_path)
    args.log_wandb = False

    main(args)