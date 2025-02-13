import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
from performative_game.predicted_population.population import Population
from performative_game.predicted_population.torch_poisson_binomial import PoissonBinomial
from performative_game.game import CRD


class FeedForward(nn.Module):
    def __init__(self, pop_size, embedding_dim=128):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Embedding(num_embeddings=3,  # cooperate, defect, or no "action played yet"
                         embedding_dim=embedding_dim),
            nn.Flatten(start_dim=0),  # receive a sequence of embeddings with length = pop_size; flatten all dims including 0 since we're not using batches for now
            nn.Linear(embedding_dim*pop_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, pop_size),
            nn.Sigmoid()  # each logit is a separate Bernoulli, independent from the others
        )

    def forward(self, x):
        '''

        :param x: shape (pop_size) one integer representing each player's past action \in {0, 1, 2}
        :return: shape (pop_size) one float representing each player's predicted probability of cooperating next
        '''
        return self.linear_relu_stack(x)


class GCN(torch.nn.Module):
    def __init__(self, embedding_dim=512, steps_memorized=1, add_layer="linear", pop_size=None):
        super().__init__()
        self.embs = nn.Embedding(num_embeddings=3,  # cooperate, defect, or no "action played yet"
                                 embedding_dim=embedding_dim)
        self.conv1 = GCNConv(embedding_dim*steps_memorized, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)

        self.add_layer = add_layer
        if add_layer is None:
            self.conv4 = GCNConv(512, 1)  # return logit for "cooperate"
        elif add_layer == "linear":
            assert isinstance(pop_size, int)
            self.layer = nn.Linear(512*pop_size, pop_size)
        elif add_layer == "mlp":
            assert isinstance(pop_size, int)
            self.layer = nn.Sequential(
                nn.Linear(512*pop_size, 512),
                nn.ReLU(),
                nn.Linear(512, pop_size)
            )
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()  # each logit is a separate Bernoulli, independent from the others

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embs(x)
        # convert from (num_nodes, num_steps, embedding_dim) into (num_nodes, num_steps*embedding_dim)
        x = x.view(x.shape[0], -1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        if self.add_layer is None:
            x = self.conv4(x, edge_index).squeeze(-1)
        elif self.add_layer in ["linear", "mlp"]:
            x = self.layer(x.view(-1))
        else:
            raise NotImplementedError
        return self.sigmoid(x)


class Predictor(nn.Module):
    def __init__(self, pop=None, architecture="mlp", random_gnn_graph=False, gnn_add_layer="linear", device="cpu"):
        super().__init__()
        self.edge_index = None
        if pop is None:
            self.predictor = None
            self.architecture = architecture
            return
        pop_size = pop.count_agents()
        if architecture == "mlp":
            self.predictor = FeedForward(pop_size)
        elif architecture == "gnn":
            if not random_gnn_graph:
                self.edge_index = pyg_utils.from_networkx(pop.G).edge_index.to(device)
            else:
                # use a random graph for debug purposes
                from performative_game.predicted_population.population import ScaleFreePopulation
                # self.edge_index = pyg_utils.from_networkx(ScaleFreePopulation(seed=47).G).edge_index  # this accidentally keeps several correct edges (e.g. 0 is still a hub)
                data = pyg_utils.from_networkx(ScaleFreePopulation(seed=47).G)
                num_nodes = data.num_nodes  # Shuffle node indices
                shuffled_indices = torch.randperm(num_nodes)
                data.edge_index = shuffled_indices[data.edge_index]  # Apply the shuffled indices to edge index
                self.edge_index = data.edge_index.to(device)

            self.predictor = GCN(add_layer=gnn_add_layer, pop_size=pop_size)
        else:
            raise NotImplementedError
        self.architecture = architecture
        self.device = device

    @staticmethod
    def write_dummy_predictions(population: Population):
        for node in population.G.nodes():
            node.last_prediction = torch.tensor(0.5)

    def write_nn_predictions(self,
                             population: Population,
                             h0: torch.tensor = None):
        if self.predictor is None:
            raise RuntimeError("predictor's neural network hasn't been instantiated")
        prev_acts = population.get_prev_acts()

        if self.architecture == "mlp":
            preds = self.predictor(torch.tensor(prev_acts).to(self.device))
        elif self.architecture == "gnn":
            if type(prev_acts[0]) == int:
                prev_acts = torch.tensor(prev_acts).unsqueeze(1).to(self.device)  # add dim for features per node
            prev_acts = Data(x=prev_acts, edge_index=self.edge_index)
            preds = self.predictor(prev_acts)
        else:
            raise NotImplementedError

        assert len(population.G.nodes()) == len(preds)
        for node, pred in zip(population.G.nodes(), preds):
            node.last_prediction = pred
        return preds, h0


class ProxyCoopLoss(nn.Module):
    """
    simplest proxy loss, aiming at maximizing all agents' propensity to cooperate,
     regardless of their position in the population topology
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                population: Population):
        acum_loss = 0
        # this only considers current time step
        for agent in population.G:
            acum_loss += nn.Sigmoid()(agent.last_expected_gain)
        return -1 * acum_loss


class ProxyGroupLoss(nn.Module):
    """
    proxy loss which considers group structure, maximizing successful groups while minimizing unnecessary coop cost
    """
    def __init__(self,
                 crd: CRD):
        super().__init__()
        self.crd = crd

    def forward(self,
                population: Population):
        cost = self.crd.c
        acum_loss = 0
        sigmoid = nn.Sigmoid()
        # this only considers current time step
        for agent in population.G:
            threshold = self.crd.get_threshold(agent.group_size)
            self_coop_prob = sigmoid(agent.last_expected_gain)
            group_coop_prob = [self_coop_prob, ]
            for nei in population.G.neighbors(agent):
                nei_coop_prob = sigmoid(nei.last_expected_gain)
                group_coop_prob.append(nei_coop_prob)
            prob_above_thres = PoissonBinomial(group_coop_prob).x_or_more(threshold)

            # we're not just maximizing prob_group_success, but balancing it with the individual cost of coop
            # but to be more principled (to max pop's utility) we should multiply prob_above_thres by its utility gain
            # which is self.crd.e*self.crd.r
            acum_loss += prob_above_thres - cost * self_coop_prob
        return -1 * acum_loss


class BinaryCrossEntropyBackpropLabels(Function):
    @staticmethod
    def forward(ctx, input, preTarget):
        # Save tensors for backward computation
        ctx.save_for_backward(input, preTarget)

        # Compute target as the step function (binary values)
        target = (preTarget > 0).float()

        # Compute the binary cross-entropy loss
        loss = F.binary_cross_entropy(input, target)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, preTarget = ctx.saved_tensors

        target = (preTarget > 0).float()
        # Gradient of the binary cross-entropy loss with respect to input
        grad_input = grad_output * (input - target) / torch.clamp(input * (1 - input), min=1e-15)  # avoid division by zero (when input rounds up to 1 by numerical error)

        # (1) Gradient of binary cross-entropy with respect to target (note typically we compute grad of CE wrt input, not wrt target!)
        #  = - (ln(input) - ln(1 - input))  # we'll also clamp ln() to -100, like in PyTorch's implementation
        # (2) Grad target wrt to preTarget is simply sigmoid_preTarget * (1 - sigmoid_preTarget)
        # Gradient of binCE with respect to preTarget [assuming here that target=sigmoid(preTarget)]
        sigmoid_preTarget = torch.sigmoid(preTarget)
        grad_preTarget = (grad_output
                          * (torch.clamp(torch.log(1 - input), min=-100) - torch.clamp(torch.log(input), min=-100) )  # (1)
                          * sigmoid_preTarget * (1 - sigmoid_preTarget))  # (2)
        # grad_preTarget = torch.zeros_like(grad_preTarget)  # DEBUG ONLY; this should be equivalent to vanilla binary CE

        return grad_input, grad_preTarget
