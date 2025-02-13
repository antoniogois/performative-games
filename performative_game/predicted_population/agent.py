import torch
from torch import log, exp, logsumexp
from torch.autograd import Function
from scipy.stats import bernoulli
import wandb
from performative_game.predicted_population.torch_poisson_binomial import PoissonBinomial
from performative_game.game import CRD


class Agent:
    def __init__(self, group_size=5, prior_trust=0.5, alpha=.8, _id=None, log_wandb=False, epsilon=None,
                 block_trust_grad=False, block_prev_trust=False, use_custom_grad=False):
        """
        :param group_size: group size, including neighbours and self
        :param trust: probability of predictor being more trustworthy than innate expectation of neighbours
        :param alpha: innate expectation that a neigbhour will cooperate; currently doesn't support a unique alpha per neighbour
        :param _id: to optionally preserve the label assigned bu networkx to the node
        """
        self.prior_trust = torch.Tensor([prior_trust])

        self.group_size = group_size
        self.trust = self.prior_trust
        self.alphas = torch.Tensor([alpha] * (group_size - 1))
        self._id = _id
        self.last_action = None
        self.last_prediction = None  # store external prediction of whether Agent itself will cooperate
        self.last_expected_gain = None

        self.use_custom_grad = use_custom_grad
        self.init_trust_estimate = 0.7  # store predictor's estimate of Agent's trust; may match ground-truth, empirically is not seem sensitive to initialization
        self.trust_estimate = self.init_trust_estimate
        self.use_trust_estimate = False  # for backward, use an estimate of trust instead of ground-truth
        self.prev_log_trust = None  # initialization is irrelevant here
        self.epsilon = epsilon if epsilon is None else \
            log(torch.tensor([epsilon]))  # avoid NaNs if needed, computing trust posterior (None, or e.g. log(torch.tensor([1e-12])) )
        self.block_trust_grad = block_trust_grad
        self.block_prev_trust = block_prev_trust

        # auxiliary log-products to compute posterior
        self.log_predicted_prod = log(self.prior_trust)
        self.log_innate_prod = log(1 - self.prior_trust)

        # pre-compute probability of being at threshold given innate model, for each possible threshold.
        # only possible since alphas are fixed, later we'll need to recompute each time
        self.gs_alpha = PoissonBinomial(self.alphas).pmf

        self.log_wandb = log_wandb
        self.plot_step = -1 if log_wandb else None

    def __str__(self):
        # return str(self._id)
        return str(f'{self.trust:.2f}')

    def reset_agent(self):
        self.trust = self.prior_trust
        self.last_action = None
        self.last_prediction = None  # store external prediction of whether Agent itself will cooperate
        self.last_expected_gain = None

        # auxiliary log-products to compute posterior
        self.log_predicted_prod = log(self.prior_trust)
        self.log_innate_prod = log(1 - self.prior_trust)

        self.trust_estimate = self.init_trust_estimate

    def update_trust_posterior(self,
                               predictions: list,
                               actions: list,
                               block_grad=False):
        if not self.use_custom_grad:
            self.update_trust_posterior_autograd_original(predictions, actions, self.block_trust_grad)
            # self.update_trust_posterior_autograd(predictions, actions, self.block_trust_grad)
        else:
            self.update_trust_posterior_customgrad(predictions, actions, self.block_trust_grad)

    def _get_neighbours_likelihood(self, predictions: list, actions: list):
        assert self.group_size - 1 == len(predictions)
        new_log_predicted_prod = torch.zeros(1)
        new_log_innate_prod = torch.zeros(1)
        for j in range(self.group_size - 1):
            coop = actions[j]  # whether neighbour j cooperated
            pred_prob_j = predictions[j] if coop else 1 - predictions[j]
            inn_prob_j = self.alphas[j] if coop else 1 - self.alphas[j]

            new_log_predicted_prod += log(pred_prob_j)
            new_log_innate_prod += log(inn_prob_j)
        return new_log_predicted_prod, new_log_innate_prod

    def update_trust_posterior_customgrad(self,
                                          predictions: list,
                                          actions: list,
                                          block_grad=False):
        assert self.group_size - 1 == len(predictions) == len(actions)
        if self.epsilon is not None:
            denominator = logsumexp(torch.cat((self.log_predicted_prod, self.log_innate_prod, self.epsilon), dim=0), dim=0)
        else:
            denominator = logsumexp(torch.cat((self.log_predicted_prod, self.log_innate_prod), dim=0), dim=0)

        prev_log_trust = self.log_predicted_prod - denominator

        self.prev_log_trust = prev_log_trust  # for grad of estimated trust
        new_log_predicted_prod, new_log_innate_prod = self._get_neighbours_likelihood(predictions, actions)

        log_trust_estimate = log(torch.tensor(self.trust_estimate)) if self.use_trust_estimate else None
        self.trust = get_updated_trust(new_log_predicted_prod, prev_log_trust, new_log_innate_prod,
                                       log_trust_estimate, self.epsilon, self.block_prev_trust)
        self.log_predicted_prod = self.log_predicted_prod + new_log_predicted_prod
        self.log_innate_prod = self.log_innate_prod + new_log_innate_prod

        if block_grad:
            self.trust = self.trust.detach()

    def update_trust_posterior_autograd(self,
                                        predictions: list,
                                        actions: list,
                                        block_grad=False):
        assert self.group_size - 1 == len(predictions) == len(actions)

        new_log_predicted_prod = torch.zeros(1)
        new_log_innate_prod = torch.zeros(1)
        for j in range(self.group_size - 1):
            coop = actions[j]  # whether neighbour j cooperated
            pred_prob_j = predictions[j] if coop else 1 - predictions[j]
            inn_prob_j = self.alphas[j] if coop else 1 - self.alphas[j]

            new_log_predicted_prod += log(pred_prob_j)
            new_log_innate_prod += log(inn_prob_j)

        self.trust = exp(new_log_predicted_prod + self.log_predicted_prod
                         - logsumexp(torch.cat((new_log_predicted_prod + self.log_predicted_prod,
                                                        new_log_innate_prod + self.log_innate_prod), dim=0), dim=0))
        self.log_predicted_prod += new_log_predicted_prod
        self.log_innate_prod += new_log_innate_prod

        if block_grad:
            self.trust = self.trust.detach()

    def update_trust_posterior_autograd_original(self,
                                        predictions: list,
                                        actions: list,
                                        block_grad=False):
        assert self.group_size - 1 == len(predictions) == len(actions)

        for j in range(self.group_size - 1):
            coop = actions[j]  # whether neighbour j cooperated
            pred_prob_j = predictions[j] if coop else 1 - predictions[j]
            inn_prob_j = self.alphas[j] if coop else 1 - self.alphas[j]

            self.log_predicted_prod += log(pred_prob_j)
            self.log_innate_prod += log(inn_prob_j)
        self.trust = exp(self.log_predicted_prod - logsumexp(torch.cat((self.log_predicted_prod,
                                                                        self.log_innate_prod), dim=0), dim=0))

        if block_grad:
            self.trust = self.trust.detach()

    def update_predictors_trust_estimate(self,
                                         action,
                                         g_predictor,
                                         crd,
                                         smoothing_factor=.6):
        """

        :param action: action taken by agent in current time-step
        :param g_predictor: predicted probability of being at threshold in current time-step
        :param crd:

        Update predictor's estimate of Agent's trust:
         1. Binary estimate of whether Agent trusted current prediction or not, by checking if observed action matches
        action taken if trust were = 1; this will be biased towards 1, since we don't know if the match was caused by
        g_predictor, or by chance due to a similar g_alpha. But if there is a mismatch we are sure of the causality, so
        there may be some signal.
         2. Use exponential moving average to update predictor's estimate
        """
        action_from_preds = bool(crd.r * crd.e * g_predictor - crd.c>0)  # hypothetical action if trust were = 1
        current_trust_estimate = action == action_from_preds
        self.trust_estimate = smoothing_factor * current_trust_estimate + (1 - smoothing_factor) * self.trust_estimate

    def play_rational_strategy(self,
                               crd: CRD,
                               predictions: list):
        """

        :param crd: instance of game being played
        :param predictions: list with predicted probability of each neighbour cooperating
        :return boolean: True if cooperate, False if defect
        """
        assert len(predictions) == self.group_size - 1
        if self.log_wandb:
            self.plot_step += 1
        threshold = crd.get_threshold(self.group_size)
        g_alpha = self.gs_alpha[threshold - 1]  # prob of being last agent required to cooperate
        g_predictor = PoissonBinomial(predictions).pmf[threshold - 1]

        # expected_gain = crd.r * crd.e * (self.trust * g_predictor + (1 - self.trust) * g_alpha) - crd.c  # autograd
        if self.use_trust_estimate:
            self.update_predictors_trust_estimate(action=bool(crd.r * crd.e * (self.trust * g_predictor + (1 - self.trust) * g_alpha) - crd.c > 0),
                                                  g_predictor=g_predictor.detach(), crd=crd)
        else:
            self.trust_estimate = None
        expected_gain = get_expected_gain(self.trust, g_predictor, g_alpha, crd.r, crd.e, crd.c, self.trust_estimate, self._id, self.plot_step)

        action = expected_gain > 0
        # comp_action = Compute_action.apply
        # action = comp_action(expected_gain)
        self.last_action = action
        self.last_expected_gain = expected_gain  # to compute predictor's loss
        return action

    def play_random(self):
        action = bernoulli.rvs(.5)
        self.last_action = bool(action)
        return action

class GetExpectedGain(Function):
    @staticmethod
    def forward(ctx, trust, g_predictor, g_alpha, crd_r, crd_e, crd_c, trust_estimate=None, agent_id=0, plot_step=None):
        ctx.save_for_backward(trust, g_predictor, g_alpha)
        ctx.crd_r = crd_r
        ctx.crd_e = crd_e
        ctx.trust_estimate = trust_estimate
        ctx.agent_id = agent_id
        ctx.plot_step = plot_step

        # Forward pass uses the actual value of trust
        output = crd_r * crd_e * (trust * g_predictor + (1 - trust) * g_alpha) - crd_c
        return output

    @staticmethod
    def backward(ctx, grad_output):
        trust, g_predictor, g_alpha = ctx.saved_tensors
        crd_r = ctx.crd_r
        crd_e = ctx.crd_e
        trust_estimate = ctx.trust_estimate
        agent_id = ctx.agent_id
        plot_step = ctx.plot_step

        # trust = .8  # we can wrongly assume trust has a constant value
        if trust_estimate is not None:
            trust = trust_estimate

        grad_g_predictor = grad_output * crd_r * crd_e * trust
        grad_trust = grad_output * crd_r * crd_e * (g_predictor - g_alpha)
        grad_g_alpha = None  # grad_output * crd_r * crd_e * (1- trust)  # unnecessary since grad_g_alpha doesn't depend on model params
        # if plot_step is not None:
        #     game_round = plot_step%20
        #     # if game_round == 19:
        #     #     wandb.log({
        #     #                   f'per_agent/grad_trust/agent{agent_id:02d}_galpha{g_alpha:.3f}/round{game_round:02d}': crd_r * crd_e * (
        #     #                               g_predictor - g_alpha), }, step=int((plot_step-19)/20))
        #
        #     # wandb.log({f'per_agent/grad_trust/agent{agent_id:02d}_galpha{g_alpha:.3f}/round{game_round:02d}':crd_r * crd_e * (g_predictor - g_alpha),})
        #     wandb.log({f'per_agent/grad_g_predictor/agent{agent_id:02d}_galpha{g_alpha:.3f}/round{game_round:02d}':crd_r * crd_e * trust,})
        #     # wandb.log({f'per_agent/agent{agent_id:02d}_galpha{g_alpha:.3f}':
        #     #                {'grad_trust': grad_trust,
        #     #                 'grad_g_predictor': grad_g_predictor},
        #     #            },
        #     #           step=plot_step)

        del ctx.crd_r
        del ctx.crd_e
        del ctx.trust_estimate
        del ctx.agent_id
        del ctx.plot_step
        # Return gradients for all inputs (matching the number of forward inputs)
        return grad_trust, grad_g_predictor, grad_g_alpha, None, None, None, None, None, None

# def get_expected_gain(trust, g_predictor, g_alpha, crd_r, crd_e, crd_c, trust_estimate=None, agent_id=0):
#     return GetExpectedGain.apply(trust, g_predictor, g_alpha, crd_r, crd_e, crd_c, trust_estimate, agent_id)
get_expected_gain = GetExpectedGain.apply

class GetUpdatedTrust(Function):
    @staticmethod
    def forward(ctx, new_log_predicted_prod, prev_log_trust, new_log_innate_prod, log_trust_estimate=None,
                epsilon=None, block_prev_trust=False):
        if log_trust_estimate is None:
            backw_prev_log_trust = prev_log_trust
        else:
            backw_prev_log_trust = log_trust_estimate
        ctx.save_for_backward(new_log_predicted_prod, backw_prev_log_trust, new_log_innate_prod)
        ctx.epsilon = epsilon
        ctx.block_prev_trust = block_prev_trust

        prev_log_not_trust = torch.log1p(-exp(prev_log_trust))  # log(1-trust) innate_prod

        # epsilon = log(torch.tensor([1e-12]))
        if epsilon is not None:
            denominator = logsumexp(torch.cat((new_log_predicted_prod + prev_log_trust,
                                                new_log_innate_prod + prev_log_not_trust,
                                               epsilon), dim=0), dim=0)
        else:
            denominator = logsumexp(torch.cat((new_log_predicted_prod + prev_log_trust,
                                               new_log_innate_prod + prev_log_not_trust), dim=0), dim=0)
        trust = exp(new_log_predicted_prod + prev_log_trust - denominator)
        return  trust

    @staticmethod
    def backward(ctx, grad_output):
        new_log_predicted_prod, prev_log_trust, new_log_innate_prod = ctx.saved_tensors
        epsilon = ctx.epsilon
        block_prev_trust = ctx.block_prev_trust

        prev_log_not_trust = torch.log1p(-exp(prev_log_trust))  # log(1-trust)

        # epsilon = log(torch.tensor([1e-12]))
        if epsilon is not None:
            denominator = logsumexp(torch.cat((new_log_predicted_prod + prev_log_trust,
                                          new_log_innate_prod + prev_log_not_trust,
                                               epsilon), dim=0), dim=0)
        else:
            denominator = logsumexp(torch.cat((new_log_predicted_prod + prev_log_trust,
                                               new_log_innate_prod + prev_log_not_trust), dim=0), dim=0)

        # for probabilities provided in log-space, we add new_log_predicted_prod
        grad_new_log_predicted_prod = grad_output * exp( prev_log_trust + prev_log_not_trust + new_log_innate_prod + new_log_predicted_prod
            - 2 * denominator)

        # for log input we add "prev_log_trust"
        grad_prev_log_trust = grad_output * exp( new_log_predicted_prod + new_log_innate_prod + prev_log_trust
            - 2 * denominator)

        if block_prev_trust:
            grad_prev_log_trust = None

        # # for log input we add "prev_log_trust"
        # grad_prev_log_predicted_prod = grad_output  * exp( new_log_predicted_prod + new_log_innate_prod + prev_log_not_trust + prev_log_trust
        #     - 2 *logsumexp(torch.cat((new_log_predicted_prod + prev_log_trust,
        #                               new_log_innate_prod + prev_log_not_trust), dim=0), dim=0))

        # grad_new_log_predicted_prod = None
        # grad_prev_trust = None

        grad_new_log_innate_prod = None
        grad_prev_log_innate_prod = None
        grad_log_trust_estimate = None
        del ctx.epsilon
        del ctx.block_prev_trust
        # return grad_new_log_predicted_prod, grad_prev_log_predicted_prod, grad_new_log_innate_prod
        return (grad_new_log_predicted_prod, grad_prev_log_trust, grad_new_log_innate_prod, grad_log_trust_estimate,
                None, None)

# def get_updated_trust(new_log_predicted_prod, prev_log_trust, new_log_innate_prod, log_trust_estimate=None):
#     return GetUpdatedTrust.apply(new_log_predicted_prod, prev_log_trust, new_log_innate_prod, log_trust_estimate)
get_updated_trust = GetUpdatedTrust.apply

class Compute_action(Function):
    """
    This class inherits from torch.autograd.Function so we can implement the backward
    function manually. This way, autograd will call this backward function while computing
    the gradient.
    source: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(input)
        return input > 0

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * torch.sigmoid(input) * (1 - torch.sigmoid(input))
