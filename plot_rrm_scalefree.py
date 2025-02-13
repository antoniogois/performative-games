import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from performative_game.game import CRD
from performative_game.predicted_population.population import LatticePopulation, ScaleFreePopulation, Population
from utils import crd_parser
import os
import pickle

class RRMPredictor:
    def __init__(self, pop_size):
        self.prev_actions = None
        self.pop_size = pop_size

    def write_rrm_predictions(self,
                              population: Population):
        if self.prev_actions is None:
            # random initialize (with binary preds) in {0,1}^n
            binom_param = torch.rand(1)  # sample parameter from uniform
            binomial_dist = torch.distributions.Binomial(total_count=self.pop_size, probs=binom_param)
            sample = int(binomial_dist.sample().item())  # count of successes
            samples = [1] * sample + [0] * (self.pop_size - sample)
            samples = torch.tensor(samples)
            self.prev_actions = samples[torch.randperm(samples.size(0))]

        for i, node in enumerate(population.G.nodes()):
            node.last_prediction = self.prev_actions[i]


def plot_histogram(fractions, t, n, weights=None, normalize=True, save_path=None, save_plot=True):
    # Define the number of bins and width
    num_bins = 10
    bin_width = 1.0 / num_bins

    # Generate bin edges
    bin_edges = np.arange(0, 1 + bin_width, bin_width)

    # Ensure that one bin edge is exactly at t
    # Adjust bin edges to ensure one starts at t
    if t not in bin_edges:
        # bin_edges = np.concatenate(([0], bin_edges[1:]))
        bin_edges[np.searchsorted(bin_edges, t)] = t

    if normalize:
        weights = 100.0 * (weights / np.sum(weights))

    # Compute the histogram data below the red line (x=t)
    hist, _ = np.histogram(fractions, bins=bin_edges, weights=weights)

    # Find the bin index corresponding to t and sum the counts up to the bin index
    bin_index = np.searchsorted(bin_edges, t) - 1

    # Sum the counts up to the bin index
    cumulative_counts = np.cumsum(hist)
    proportion_below_t = cumulative_counts[bin_index]
    
    # Normalize the sum if required
    proportion_below_t = proportion_below_t / np.sum(hist)
        
    # import ipdb; ipdb.set_trace()
    
    # Plot histogram if save_plot is True
    if save_plot:
        plt.figure(figsize=(10,8))
        plt.hist(fractions, weights=weights, bins=bin_edges, edgecolor='black', alpha=0.7)
        # Add a vertical dashed line at t
        vline = plt.axvline(x=t, color='red', linestyle='--', linewidth=2, label='Threshold')

        # changing the font size of the ticks
        plt.tick_params(axis='both', which='major', labelsize=24, length=10, width=1)
        
        plt.ylim(0, 100)
        plt.xlabel('Fraction of cooperators per group', fontsize=24)
        plt.ylabel('% of groups', fontsize=24)
        plt.legend(handles=[vline], fontsize=24)
        plt.title(f'RRM with a {n}-node scale-free graph', fontsize=24)
        # plt.show()
        
        if save_path is not None:
            save_path = os.path.join(save_path, f'cooperation_histogram_{n}_nodes_{t}_threshold.pdf')
            plt.savefig(save_path)
        plt.show()
    
    return proportion_below_t

def rrm_iteration(predictor, pop, crd):
    predictor.write_rrm_predictions(pop)
    pop_actions = pop.play_all_rational(crd, update_trust=False)
    predictor.prev_actions = torch.tensor(pop_actions, dtype=torch.int)
    return pop_actions

def rrm_single_trial(args, pop, crd):
    pop_actions_history = []
    pop_size = pop.count_agents()
    # import ipdb; ipdb.set_trace()
    predictor = RRMPredictor(pop_size=pop_size)
    prev_pop_actions = None
    pop_actions = None
    converged = False
    cycled = False
    cycle_len = 0
    fractions = []
    weights = []
    
    # could take exponentially long to converge/cycle, but empirically does not happen
    while True:
        if pop_actions is not None:
            prev_pop_actions = pop_actions
        pop_actions = rrm_iteration(predictor, pop, crd)
        if pop_actions == prev_pop_actions:
            converged = True
            break
        if pop_actions in pop_actions_history:
            cycled = True
            break
        pop_actions_history.append(pop_actions)

    if converged:
        # print('Converged after {} rounds'.format(i))
        # print(pop_actions)
        # print(f"{sum(pop.get_group_successes(crd)).item()} out of {args.pop_size} groups overcame threshold")
        fractions.extend(pop.get_group_fractions())
        weights.extend([1] * pop_size)
    elif cycled:
        # print('Cycled after {} rounds'.format(i))
        start_pop_actions = pop_actions
        cycle_fractions = pop.get_group_fractions()
        cycle_len = 1
        pop_actions = rrm_iteration(predictor, pop, crd)
        while pop_actions != start_pop_actions:
            cycle_fractions.extend(pop.get_group_fractions())
            cycle_len += 1
            pop_actions = rrm_iteration(predictor, pop, crd)
        fractions.extend(cycle_fractions)
        weights.extend([1 / cycle_len] * len(cycle_fractions))
    
    return converged, cycled, cycle_len, fractions, weights

def rrm_single_pop(args):
    torch.manual_seed(args.seed)
    crd = CRD(e=args.endowment,
              r=args.risk,
              c=args.coop_cost,
              t=args.threshold)
    pop = ScaleFreePopulation(seed=args.seed, n=args.pop_size, prior_trust=1.0, m=2)
    fractions, weights = [], []

    n_converged, n_cycled = 0, 0
    for _ in range(args.n_trials):
        converged, cycled, _, trial_fractions, trial_weights = rrm_single_trial(args, pop, crd)
        fractions.extend(trial_fractions)
        weights.extend(trial_weights)
        if converged:
            n_converged += 1
        if cycled:
            n_cycled += 1

    print(f"{n_cycled} out of {args.n_trials} trials cycled")
    return n_converged, n_cycled, fractions, weights

def plot_rrm_single_pop(args):
    _, _, fractions, weights = rrm_single_pop(args)
    plot_histogram(fractions, args.threshold, args.pop_size, weights=weights, save_path=args.save_path)
    
    
def rrm_multi_pop(args):
    n_cyled_list, n_converged_list, cycle_len_list, proportion_below_thr_list, std_n_cycled_list, std_prop_below_thr_list = [], [], [], [], [], []
    torch.manual_seed(args.seed)
    for pop_size in range(args.min_pop_size, args.max_pop_size + 1, args.step_size):
        print(f"Running RRM for population size {pop_size}")
        crd = CRD(e=args.endowment,
                r=args.risk,
                c=args.coop_cost,
                t=args.threshold)
        n_cycled_pop_list = []
        n_converged_pop_list = []
        cycle_len_pop_list = []
        prop_below_thr_pop_list = []

        for n in range(args.n_rounds_per_size):
            pop = ScaleFreePopulation(seed=args.seed+n, n=pop_size, prior_trust=1.0, m=2)
            n_converged, n_cycled, cycle_len_sum = 0, 0, 0
            fractions, weights = [], []
            for _ in range(args.n_trials):
                # import ipdb; ipdb.set_trace()
                converged, cycled, cycle_len, trial_fractions, trial_weights = rrm_single_trial(args, pop, crd)
                if converged:
                    n_converged += 1
                if cycled: 
                    n_cycled += 1
                    cycle_len_sum += cycle_len
                    print(f'Cycle length: {cycle_len}, n_cycled: {n_cycled}')
                fractions.extend(trial_fractions)
                weights.extend(trial_weights)
            
            n_cycled_pop_list.append(n_cycled/args.n_trials)
            n_converged_pop_list.append(n_converged/args.n_trials)
            cycle_len_pop_list.append(cycle_len_sum/n_cycled if n_cycled > 0 else 0)
            
            
            # calculate the proportion of groups below the threshold
            proportion_below_thr = plot_histogram(fractions, args.threshold, pop_size, weights=weights, save_path=args.save_path, save_plot=False)
            prop_below_thr_pop_list.append(proportion_below_thr)
            
            
        avg_n_cycled = sum(n_cycled_pop_list) / args.n_rounds_per_size
        avg_n_converged = sum(n_converged_pop_list) / args.n_rounds_per_size    
        avg_cycle_len = sum(cycle_len_pop_list) / args.n_rounds_per_size 
        avg_prop_below_thr = sum(prop_below_thr_pop_list) / args.n_rounds_per_size
        std_n_cycled = np.std(n_cycled_pop_list)
        std_prop_below_thr = np.std(prop_below_thr_pop_list)
        
        n_cyled_list.append(avg_n_cycled)
        n_converged_list.append(avg_n_converged)
        cycle_len_list.append(avg_cycle_len)
        proportion_below_thr_list.append(avg_prop_below_thr)
        std_n_cycled_list.append(std_n_cycled)
        std_prop_below_thr_list.append(std_prop_below_thr)
        print(f"prop below threshold {avg_prop_below_thr}")
        print(f"For population size {pop_size}, {avg_n_cycled} trials cycled on average")
            
        
    # save n_cycled_list, n_converged_list, and cycle_len_list to a file in data_save_path
    if args.save_path is not None:
        data_save_path = os.path.join(args.save_path, 'multi_pop_exp')
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        with open(os.path.join(data_save_path, 'n_cycled_list.pkl'), 'wb') as f:
            pickle.dump(n_cyled_list, f)
        with open(os.path.join(data_save_path, 'n_converged_list.pkl'), 'wb') as f:
            pickle.dump(n_converged_list, f)
        with open(os.path.join(data_save_path, 'cycle_len_list.pkl'), 'wb') as f:
            pickle.dump(cycle_len_list, f)
        with open(os.path.join(data_save_path, 'proportion_below_thr_list.pkl'), 'wb') as f:
            pickle.dump(proportion_below_thr_list, f)
        with open(os.path.join(data_save_path, 'proportion_below_thr_list_std.pkl'), 'wb') as f:
            pickle.dump(std_prop_below_thr_list, f)
        with open(os.path.join(data_save_path, 'n_cycled_list_std.pkl'), 'wb') as f:
            pickle.dump(std_n_cycled_list, f)

       
def plot_rrm_multi_pop(args):
    if args.save_path is not None:
        data_save_path = os.path.join(args.save_path, 'multi_pop_exp')
        
    # load the data from the file
        with open(os.path.join(data_save_path, 'n_cycled_list.pkl'), 'rb') as f:
            n_cycled_list = pickle.load(f)
        with open(os.path.join(data_save_path, 'proportion_below_thr_list.pkl'), 'rb') as f:
            proportion_below_thr_list = pickle.load(f)
        with open(os.path.join(data_save_path, 'proportion_below_thr_list_std.pkl'), 'rb') as f:
            std_prop_below_thr_list = pickle.load(f)
        with open(os.path.join(data_save_path, 'n_cycled_list_std.pkl'), 'rb') as f:
            std_n_cycled_list = pickle.load(f)

    plt.figure(figsize=(12,10))
    plt.tick_params(axis='both', which='major', labelsize=24, length=10, width=1)
    _, ax = plt.subplots()
    pop_size = range(args.min_pop_size, args.max_pop_size + 1, args.step_size)  # x-axis

    color1 = 'gray'
    color2 = 'royalblue'
    plt.fill_between(pop_size,
                     np.array(n_cycled_list) - np.array(std_n_cycled_list),
                     np.array(n_cycled_list) + np.array(std_n_cycled_list),
                     color=color1, alpha=0.3)
    plt.fill_between(pop_size,
                     np.array(proportion_below_thr_list) - np.array(std_prop_below_thr_list),
                     np.array(proportion_below_thr_list) + np.array(std_prop_below_thr_list),
                     color=color2, alpha=0.3)
    ax.plot(pop_size, proportion_below_thr_list, color=color2, label='Time below threshold\nafter convergence', linewidth=2)
    ax.plot(pop_size, n_cycled_list, color=color1, label='Cycled trials', linewidth=2)
    ax.set_xlabel('Population size', fontsize=22)
    ax.set_ylabel('Fraction', fontsize=22)
    ax.tick_params(axis='y')

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, fontsize=13, loc='lower right')

    if args.save_path is not None:
        # Save the plot
        plt.savefig(os.path.join(data_save_path, 'curves.pdf'), bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    crd_parser(parser)  # add args for game parameters

    parser.add_argument('-n', '--pop_size', type=int, default=20, help='Population size, for a single histogram')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument( '--n_trials', type=int, default=10, help='number of random initializations for predictor, per graph')
    parser.add_argument('--min_pop_size', type=int, default=5, help='Minimum population size to test for rrm')
    parser.add_argument('--max_pop_size', type=int, default=20, help='Maximum population size to test for rrm')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for population size in plotting cycle/convergence fractions')
    parser.add_argument('--n_rounds_per_size', type=int, default=5, help='Number of graphs to average over for each population size')
    parser.add_argument('--plot_single_pop', action='store_true', help='Plot histogram for a single population size')
    
    parser.add_argument('--save_path', help='path to save plots', default="rrm")


    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.plot_single_pop:
        # compute and plot histogram for a single graph G, averaged over "n_trials" initializations of predictions
        plot_rrm_single_pop(args)
    else:
        # compute (n_cycled_list, n_converged_list), then plot
        rrm_multi_pop(args)
        plot_rrm_multi_pop(args)
    
    
    