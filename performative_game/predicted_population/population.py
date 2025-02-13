import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import torch
from performative_game.predicted_population.agent import Agent


class Population:
    def __init__(self, g, seed=None, prior_trust=0.5, log_wandb=False, alpha=None, epsilon=None,
                 block_trust_grad=False, block_prev_trust=False, use_custom_grad=False):
        """

        :param g: networkx graph where nodes are not yet agents, and edges determine neigbhours in game
        """
        # convert nodes to agents, for any given graph
        if alpha is None:
            mapping = {node: Agent(group_size=g.degree(node) + 1,
                                   _id=node, prior_trust=prior_trust,
                                   log_wandb=log_wandb,
                                   epsilon=epsilon,
                                   block_trust_grad=block_trust_grad,
                                   block_prev_trust=block_prev_trust,
                                   use_custom_grad=use_custom_grad) for node in g.nodes()}
        else:
            mapping = {
                node: Agent(group_size=g.degree(node) + 1,
                            _id=node,
                            prior_trust=prior_trust,
                            log_wandb=log_wandb,
                            epsilon=epsilon,
                            block_trust_grad=block_trust_grad,
                            block_prev_trust=block_prev_trust,
                            use_custom_grad=use_custom_grad,
                            alpha=alpha) for node in g.nodes()}
        self.G = nx.relabel_nodes(g, mapping)
        self.pos = self.set_nodes_pos(self.G, seed=seed)
        self.label_pos = {node: (x, y + 0.05) for node, (x, y) in self.pos.items()}
        self.cmap = plt.get_cmap('Blues')

    @staticmethod
    def set_nodes_pos(G, seed=None):
        pos = nx.spring_layout(G, seed=seed)
        # pos = nx.shell_layout(G)
        # pos = nx.spectral_layout(G)
        return pos

    def count_agents(self):
        return self.G.number_of_nodes()

    def reset_population(self):
        for node in self.G.nodes:
            node.reset_agent()
        self.update_coop_colours()
        self.update_trust_colours()
        self.update_prediction_colours()

    def get_prev_acts(self):
        prev_acts = []
        for node in self.G:
            prev_acts.append(node.last_action)

        # replace Nones by a special symbol, to indicate "no action played yet"
        prev_acts = [int(act) if act is not None else 2 for act in prev_acts]
        return prev_acts

    def get_group_successes(self, crd):
        successes = []
        for node in self.G:
            threshold = crd.get_threshold(node.group_size)
            _, nei_actions = self.get_neighbours_data(node)
            nei_actions.append(node.last_action)  # add own action
            successes.append(sum(nei_actions) >= threshold)
        return successes

    def get_group_fractions(self):
        fractions = []
        for node in self.G:
            _, nei_actions = self.get_neighbours_data(node)
            nei_actions.append(node.last_action)  # add own action
            fractions.append((sum(nei_actions)/node.group_size).item())
        return fractions

    def draw_network(self, timestep=None, update_inner_colour=True, path=None, show_plot=True, inner_colour='trust'):
        self.update_coop_colours()
        if update_inner_colour:
            self.update_trust_colours()
            self.update_prediction_colours()

        node_outline = [self.G.nodes[node]['coop_color'] for node in self.G.nodes]
        colorbar = plt.colorbar(cm.ScalarMappable(cmap=self.cmap))
        if inner_colour == 'trust':
            node_fill = [self.G.nodes[node]['trust_color'] for node in self.G.nodes]
            colorbar.set_label('Trust in predictor', rotation=90)
        elif inner_colour == 'prediction':
            node_fill = [self.G.nodes[node]['prediction_color'] for node in self.G.nodes]
            colorbar.set_label('previous prediction', rotation=90)
        else:
            raise NotImplementedError
        nx.draw_networkx(self.G, self.pos, with_labels=False, node_size=100, node_color=node_fill,
                         edgecolors=node_outline, linewidths=3.0,
                         font_size=10, font_color='black', font_weight='bold')
        # nx.draw_networkx_labels(self.G, self.label_pos, font_size=10, font_color='black', font_weight='bold')

        if timestep is not None:
            plt.title(f'time step {timestep}')

        if path is not None:
            fig_name = "t_" + f'{timestep:.1f}' if timestep is not None else "0"
            full_path = os.path.join(path, fig_name + ".png")
            plt.savefig(full_path, dpi=200, bbox_inches='tight')
        else:
            full_path = None

        if show_plot:
            plt.show()
        else:
            plt.close()
        return full_path

    def update_coop_colours(self):
        for node in self.G:
            if node.last_action is None:  # no action yet
                self.G.nodes[node]['coop_color'] = 'gray'  # 'skyblue'
            elif node.last_action:  # cooperated
                self.G.nodes[node]['coop_color'] = 'green'
            else:  # defected
                self.G.nodes[node]['coop_color'] = 'tab:red'

    def update_trust_colours(self):
        for node in self.G:
            self.G.nodes[node]['trust_color'] = self.cmap(node.trust.detach().numpy())

    def update_prediction_colours(self):
        for node in self.G:
            last_p = node.last_prediction
            if last_p is None:
                colour = 'black'
            else:
                colour = self.cmap(last_p.detach().numpy())
            self.G.nodes[node]['prediction_color'] = colour

    def get_neighbours_data(self, node):
        nei_predictions = []
        nei_actions = []
        for j in self.G.neighbors(node):
            nei_predictions.append(j.last_prediction)
            nei_actions.append(j.last_action)
        return nei_predictions, nei_actions

    def get_accuracy(self):
        accuracies = []
        for i in self.G:
            pred = i.last_prediction  # float [0, 1]
            coop = i.last_action  # int {0, 1}
            acc = pred if coop == 1 else 1 - pred
            accuracies.append(acc)
        # return their mean instead of prod
        return sum(accuracies) / len(accuracies)

    def play_all_random(self):
        for node in self.G:
            node.play_random()

    def play_all_rational(self, crd, update_trust=True):
        pop_actions = []
        for node in self.G:
            nei_preds, _ = self.get_neighbours_data(node)
            act = node.play_rational_strategy(crd, nei_preds)
            pop_actions.append(act)
        if update_trust:
            # re-iterate, now with newly computed nei_actions
            for node in self.G:
                nei_preds, nei_actions = self.get_neighbours_data(node)
                node.update_trust_posterior(nei_preds, nei_actions)
        return pop_actions

    def get_last_action_probs(self):
        coop_probs = []
        for node in self.G:
            prob = torch.sigmoid(node.last_expected_gain)
            coop_probs.append(prob)
        coop_probs = torch.cat(coop_probs)
        return coop_probs

    def get_last_expected_gains(self):
        gains = []
        for node in self.G:
            gains.append(node.last_expected_gain)
        gains = torch.cat(gains)
        return gains


class LatticePopulation(Population):
    def __init__(self, side, prior_trust=0.5, alpha=None, epsilon=None):
        g = nx.grid_graph(dim=(side, side), periodic=True)
        self.side = side
        super().__init__(g, prior_trust=prior_trust, alpha=alpha, epsilon=epsilon)

    def set_nodes_pos(self, G=None, seed=None, visualize_periodicity=True):
        # Create a dictionary of node positions in a grid layout
        # pos = {(x, y): (y, -x) for x, y in self.G.nodes()}
        pos = {agent: (agent._id[1], -agent._id[0]) for agent in self.G.nodes()}

        if visualize_periodicity:
            # shift slightly topmost and leftmost nodes
            epsilon = self.side / 20
            for agent in self.G.nodes():
                if agent._id[0] == 0:
                    pos[agent] = (pos[agent][0] - epsilon, pos[agent][1])
                if agent._id[1] == 0:
                    pos[agent] = (pos[agent][0], pos[agent][1] + epsilon)

        return pos


class ScaleFreePopulation(Population):
    def __init__(self, n=20, m=2, seed=0, prior_trust=0.5, log_wandb=False, alpha=None, epsilon=None,
                 block_trust_grad=False, block_prev_trust=False, use_custom_grad=False):
        """

        :param n: Number of nodes
        :param m: Number of edges to attach from a new node to existing nodes
        """
        g = nx.barabasi_albert_graph(n, m, seed=seed)
        super().__init__(g, seed=seed, prior_trust=prior_trust, log_wandb=log_wandb, alpha=alpha, epsilon=epsilon,
                         block_trust_grad=block_trust_grad, block_prev_trust=block_prev_trust,
                         use_custom_grad=use_custom_grad)

class FullyConnectedPopulation(Population):
    def __init__(self, n=20, prior_trust=0.5, alpha=None, epsilon=None):
        g = nx.complete_graph(n)
        super().__init__(g, seed=None, prior_trust=prior_trust, alpha=alpha, epsilon=epsilon)

    def set_nodes_pos(self, G, seed=None):
        return nx.circular_layout(G)


if __name__ == '__main__':
    # lpop = LatticePopulation(4)
    # lpop.draw_network()

    sfpop = ScaleFreePopulation()
    sfpop.draw_network()
