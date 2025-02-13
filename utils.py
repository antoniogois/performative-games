import os
from operator import itemgetter
from pathlib import Path
import pickle
import torch
from PIL import Image
import numpy as np


def crd_parser(parser):
    game = parser.add_argument_group("game parameters")
    game.add_argument("-e", "--endowment", type=float, default=1, help="initial endowment per agent")
    game.add_argument("-r", "--risk", type=float, default=.4, help="risk of losing everything,"
                                                                   "if threshold not achieved")
    game.add_argument("-c", "--coop_cost", type=float, default=.2, help="individual cost of cooperating")
    game.add_argument("-t", "--threshold", type=float, default=.5, help="fraction of agents required in group,"
                                                                        "in order to achieve success")


def get_torch_device(ignore_mps=False):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if ignore_mps and device=="mps":
        device = "cpu"
    return device


def run_simulation(crd, pop, predictor, figs_path=None, plot_figs=False, n_rounds=10, dummy_preds=False, inner_colour="trust"):
    figs_list = []
    figs_list.append(pop.draw_network(timestep=0, update_inner_colour=True, path=figs_path, show_plot=plot_figs, inner_colour=inner_colour))
    for t in range(1, n_rounds + 1):
        if dummy_preds:
            predictor.write_dummy_predictions(pop)
        else:
            predictor.write_nn_predictions(pop)
        pop.play_all_rational(crd)
        figs_list.append(
            pop.draw_network(timestep=t, update_inner_colour=False, path=figs_path, show_plot=plot_figs, inner_colour=inner_colour))
        figs_list.append(
            pop.draw_network(timestep=t + .5, update_inner_colour=True, path=figs_path, show_plot=plot_figs, inner_colour=inner_colour))
    return figs_list


def imgs2gif(figs_list, figs_path, keep_static_figs=False):
    images = [Image.open(fig) for fig in figs_list]
    images[0].save(os.path.join(figs_path, "animation.gif"), save_all=True, append_images=images[1:], duration=500, loop=0)

    if not keep_static_figs:
        for fig in figs_list:
            os.remove(fig)


class FullStats:
    def __init__(self, pop, n_rounds, optimizing):
        assert optimizing in ["individual", "group", "accuracy", "multitask", "accuracy_perf"]
        self.n_groups = pop.count_agents()
        self.n_rounds = n_rounds
        self.optimizing = optimizing
        self.track_groups = []  # list of lists with count of successful groups per round
        self.track_acc = []  # list of lists with accuracy per round
        self.track_agents = []  # list of lists with count of cooperators per round
        self.track_loss = []  # list of loss (summed across rounds) per epoch

        self.best_group = 0
        self.best_acc = 0
        self.best_indiv = 0

    def get_track_groups(self, formatted=True):
        if not formatted:
            return self.track_groups
        else:
            return [sum(l)/(self.n_groups*self.n_rounds) for l in self.track_groups]

    def get_track_agents(self, formatted=True):
        if not formatted:
            return self.track_agents
        else:
            return [sum(l)/(self.n_groups*self.n_rounds) for l in self.track_agents]

    def get_track_acc(self, formatted=True):
        if not formatted:
            return self.track_acc
        else:
            return [sum(l)/self.n_rounds for l in self.track_acc]

    def get_track_loss(self):
        return self.track_loss

    def update(self, coop_groups, accs, coop_agents, loss):
        assert len(coop_groups) == self.n_rounds
        assert len(accs) == self.n_rounds
        assert len(coop_agents) == self.n_rounds

        self.track_groups.append(coop_groups)
        self.track_acc.append(accs)
        self.track_agents.append(coop_agents)
        self.track_loss.append(loss)

        # define criteria
        group_improved = sum(self.track_groups[-1]) > sum(self.track_groups[self.best_group])
        group_same = sum(self.track_groups[-1]) == sum(self.track_groups[self.best_group])
        acc_improved = sum(self.track_acc[-1]) > sum(self.track_acc[self.best_acc])
        acc_same = sum(self.track_acc[-1]) == sum(self.track_acc[self.best_acc])
        individuals_improved = sum(self.track_agents[-1]) > sum(self.track_agents[self.best_indiv])
        individuals_same = sum(self.track_agents[-1]) == sum(self.track_agents[self.best_indiv])

        if group_improved or (group_same and acc_improved):
            self.best_group = len(self.track_groups) - 1
        if acc_improved or (acc_same and group_improved):
            self.best_acc = len(self.track_acc) - 1
        if individuals_improved or (individuals_same and acc_improved):
            self.best_indiv = len(self.track_agents) - 1

    def get_idx_stats(self, idx, discard_individuals=True):
        """ returns (% of successful groups across all rounds,
                     avg accuracy per round,
                     % of cooperators across all rounds)"""
        stats = sum(self.track_groups[idx]) / (self.n_groups*self.n_rounds),\
                sum(self.track_acc[idx]) / self.n_rounds,\
                sum(self.track_agents[idx]) / (self.n_groups*self.n_rounds)
        if discard_individuals:
            stats = stats[:2]
        return stats

    def get_best_group(self):
        return self.get_idx_stats(self.best_group)

    def get_best_acc(self):
        return self.get_idx_stats(self.best_acc)

    def get_best_n_coops(self):
        return self.get_idx_stats(self.best_indiv)

    def get_best(self):
        if self.optimizing == "group":
            return self.get_best_group()
        elif self.optimizing == "accuracy":
            return self.get_best_acc()
        elif self.optimizing == "individual":
            return self.get_best_n_coops()
        else:
            return NotImplementedError

    def get_pareto(self):
        """ get Pareto front between group success and accuracy (ignores individual cooperators)"""
        pareto_candidates = set()
        assert len(self.track_groups) == len(self.track_acc) == len(self.track_agents)
        for i in range(len(self.track_groups)):
            stats = self.get_idx_stats(i)
            pareto_candidates.update([stats])
        pareto_candidates = np.array(list(pareto_candidates))
        pareto_candidate_costs = 1 - pareto_candidates  # minimize instead of maximize
        is_pareto = is_pareto_efficient(pareto_candidate_costs)
        pareto_points = pareto_candidates[is_pareto]
        return pareto_points


class BestStats:
    def __init__(self, optimizing="group"):
        assert optimizing in ["individual", "group", "accuracy", "multitask"]
        self.optimizing = optimizing
        self.coop_groups = 0
        self.acc = 0
        self.epoch = None

    def update(self, coop_groups, acc, epoch):
        """ update best stats, if criterion is fulfilled"""
        if self.optimizing in ["individual", "group"]:
            criterion = (coop_groups > self.coop_groups) or (coop_groups == self.coop_groups and acc > self.acc)
        elif self.optimizing == "accuracy":
            criterion = (acc > self.acc) or (acc == self.acc and coop_groups > self.coop_groups)
        elif self.optimizing == "multitask":
            raise NotImplementedError  # todo define behaviour here — maybe search for all points which aren't pareto-dominated
        else:
            raise NotImplementedError

        if criterion:
            self.coop_groups = coop_groups
            self.acc = acc
            self.epoch = epoch


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def TwoDimensionsPareto(data):
    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=True)
    pareto_idx = list()
    pareto_idx.append(0)
    cutt_off = sorted_data[0][1]
    for i in range(1, len(sorted_data)):
        if sorted_data[i][1] > cutt_off:
            pareto_idx.append(i)
            cutt_off = sorted_data[i][1]
    return pareto_idx


def read_precomputed(thres, loss, seed, args=None, architecture=None, lr=None, epochs=None, path=None):
    if architecture is None:
        architecture = args.architecture
    if lr is None:
        lr = args.lr
    if epochs is None:
        epochs = args.epochs
    if path is None:
        if args is None:
            path = 'crd_stats'
        else:
            path = args.stats_path
    filename = f"full_stats_thres{thres}_{loss}_seed{seed}_{epochs}epochs_{architecture}_lr{lr}.pkl"
    fpath = Path(path) / filename
    if fpath.is_file():
        f = open(fpath, "rb")
        stats = pickle.load(f)
        return stats
    else:
        return None


def store_precomputed(stats, thres, loss, seed, args, architecture=None):
    if architecture is None:
        architecture = args.architecture
    filename = f"full_stats_thres{thres}_{loss}_seed{seed}_{args.epochs}epochs_{architecture}_lr{args.lr}.pkl"
    fpath = Path(args.stats_path) / filename
    assert not fpath.is_file()
    f = open(fpath, "wb")
    pickle.dump(stats, f)
    f.flush()


if __name__ == "__main__":
    dummy_points = np.array([[0.2, 0.7],  # true
                             [0.2, 0.6],  # false
                             [0.15, 0.6],  # false
                             [0.3, 0.1],  # true
                             [0.25, 0.15]])  # true
    is_pareto_efficient_dumb(dummy_points*(-1))
    is_pareto_efficient_simple(dummy_points*(-1))
    is_pareto_efficient(dummy_points*(-1))
    TwoDimensionsPareto(dummy_points*(-1))
    print("bye")
