import argparse
import os
import torch
from torch import nn
import wandb
import numpy as np
from utils import crd_parser, get_torch_device, run_simulation, imgs2gif, FullStats, read_precomputed, store_precomputed
from performative_game.game import CRD
from performative_game.coop_predictor import Predictor, ProxyCoopLoss, ProxyGroupLoss, BinaryCrossEntropyBackpropLabels
from performative_game.predicted_population.population import LatticePopulation, ScaleFreePopulation
binary_cross_entropy_backprop_labels = BinaryCrossEntropyBackpropLabels.apply


def compute_single_loss(predictor, pop, crd, loss_fn, args, loss_name=None):
    if loss_name is None:
        assert args.loss != "multitask", "in multitask loss, need to specify 'group' or 'accuracy'"
        loss_name = args.loss
    epoch_actions = []
    epoch_successes = []  # count of groups which overcame threshold
    epoch_accuracies = []
    loss = None
    h0 = None
    pop.reset_population()
    for t in range(args.n_rounds):
        pred_probs, h0 = predictor.write_nn_predictions(pop)
        pop_actions = pop.play_all_rational(crd)
        if loss_name == "accuracy":
            torch_actions = torch.tensor(pop_actions, dtype=torch.float)  # torch.cat(pop_actions).float() may preserve requires_grad==True, but currently the list pop_actions already comes with requires_grad==False
            # torch_actions = pop.get_last_action_probs()  # this provides probs instead of 0-1 to CE, performs poorly when measuring accuracy
            loss = loss_fn(pred_probs, torch_actions) if loss is None else loss + loss_fn(pred_probs, torch_actions)
        elif loss_name == "accuracy_perf":  # backpropagates through actions, not only through predictions
            xp_gains = pop.get_last_expected_gains()
            loss = binary_cross_entropy_backprop_labels(pred_probs, xp_gains) \
                if loss is None else loss + binary_cross_entropy_backprop_labels(pred_probs, xp_gains)
        else:
            loss = loss_fn(pop) if loss is None else loss + loss_fn(pop)
        epoch_actions.append(sum(pop_actions).item())
        epoch_successes.append(sum(pop.get_group_successes(crd)).item())
        epoch_accuracies.append(pop.get_accuracy().item())
    return loss, epoch_actions, epoch_successes, epoch_accuracies


def compute_multi_loss(predictor, pop, crd, loss_fn, optimizer, args):
    grads = []
    for l_name in ["group", "accuracy"]:
        optimizer.zero_grad()
        l, _, _, _ = compute_single_loss(predictor, pop, crd, loss_fn[l_name], args, l_name)
        l.backward()
        # store vector with gradient for current task
        grad = [param.grad.flatten() for param in predictor.parameters() if param.grad is not None]
        flattened_grad = torch.cat(grad)
        flattened_grad = flattened_grad.detach()  # only used for "scalar", detach from computational graph
        grads.append(flattened_grad)

    # compute scalar weight for task 0 (group-loss task)
    inner_p_00 = torch.dot(grads[0], grads[0])
    inner_p_01 = torch.dot(grads[0], grads[1])
    inner_p_11 = torch.dot(grads[1], grads[1])
    if inner_p_01 >= inner_p_00:
        scalar = 1
    elif inner_p_01 >= inner_p_11:
        scalar = 0
    else:
        scalar = torch.dot(grads[1] - grads[0], grads[1]) / torch.linalg.vector_norm(grads[0] - grads[1])**2

    optimizer.zero_grad()
    l_gr, epoch_actions, epoch_successes, epoch_accuracies = compute_single_loss(predictor, pop, crd, loss_fn["group"],
                                                                                 args, loss_name="group")
    loss = l_gr * scalar
    l_acc, _, _, _ = compute_single_loss(predictor, pop, crd, loss_fn["accuracy"],
                                         args, loss_name="accuracy")
    loss = loss + l_acc * (1 - scalar)
    return loss, epoch_actions, epoch_successes, epoch_accuracies


def compute_loss(args, predictor, pop, crd, loss_fn, optimizer):
    if args.loss == "multitask":
        loss, epoch_actions, epoch_successes, epoch_accuracies = compute_multi_loss(predictor, pop, crd, loss_fn,
                                                                                    optimizer, args)
    else:
        loss, epoch_actions, epoch_successes, epoch_accuracies = compute_single_loss(predictor, pop, crd, loss_fn,
                                                                                     args)
    return loss, epoch_actions, epoch_successes, epoch_accuracies


def train(args, pop, predictor, crd, loss_fn, optimizer):
    pop_size = pop.count_agents()
    full_stats = FullStats(pop, args.n_rounds, args.loss)

    for i in range(args.epochs):
        print(f"epoch {i}")
        loss, epoch_actions, epoch_successes, epoch_accuracies = compute_loss(args, predictor, pop, crd, loss_fn,
                                                                              optimizer)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"{sum(epoch_actions)} out of {args.n_rounds * pop_size} cooperated. Details of each round:"
              f" {epoch_actions} (each out of {pop_size})")
        print(f"{sum(epoch_successes)} groups out of {args.n_rounds * pop_size} overcame. Details of each round:"
              f" {epoch_successes} (each out of {pop_size})")
        print(f"{sum(epoch_accuracies) / len(epoch_accuracies):.3f} was the mean likelihood. Details of each round: ["
              + " ".join(f"{acc:.3f}" for acc in epoch_accuracies) + "]")
        if args.log_wandb:
            wandb.log({"cooperators": sum(epoch_actions),
                       "successful groups": sum(epoch_successes),
                       "avg round accuracy": sum(epoch_accuracies) / len(epoch_accuracies)},
                      step=i)

        full_stats.update(epoch_successes, epoch_accuracies, epoch_actions, loss.__float__())
    return predictor, full_stats


def main(args):
    torch.manual_seed(args.seed)
    device = get_torch_device(ignore_mps=True)
    torch.use_deterministic_algorithms(True)  # may compromise speed

    if args.log_wandb:
        wandb.init(
            project="perf_pred_coop",
            name=f"{args.architecture}_seed{args.seed}_gseed{args.graph_seed}_lr_{args.lr}",
            config={
                "architecture": args.architecture,
                "epochs": args.epochs,
                "seed": args.seed,
                "learning_rate": args.lr,
                "loss": args.loss,
                "CRD": {
                    "threshold": args.threshold,
                    "coop_cost": args.coop_cost,
                    "risk": args.risk,
                    "endowment": args.endowment,
                    "game_rounds": args.n_rounds
                },
                "topology": args.topology,
                "pop_size": args.popsize
            },
        )

    crd = CRD(e=args.endowment,
              r=args.risk,
              c=args.coop_cost,
              t=args.threshold)

    if args.topology == "lattice":
        pop = LatticePopulation(side=args.lattice_side)
    elif args.topology == "scale-free":
        pop = ScaleFreePopulation(seed=args.graph_seed,
                                  n=args.sf_popsize,
                                  log_wandb=args.log_wandb,
                                  epsilon=args.epsilon,
                                  block_trust_grad=args.block_trust_grad,
                                  block_prev_trust=args.block_prev_trust,
                                  use_custom_grad=args.use_custom_grad)
    else:
        raise NotImplementedError

    architectures = {
        "mlp": (pop, "mlp"),
        "gnn": (pop, "gnn", False, None),  # pop, architecture, random_gnn_graph, gnn_add_layer
        "gnn+linear": (pop, "gnn", False, "linear"),
        # "gnn+linear+rand_graph": (pop, "gnn", True, "linear")
        "gnn+mlp": (pop, "gnn", False, "mlp")
    }

    predictor = Predictor(*architectures[args.architecture], device=device).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    if args.loss == "individual":
        loss_fn = ProxyCoopLoss()
    elif args.loss == "group":
        loss_fn = ProxyGroupLoss(crd)
    elif args.loss in {"accuracy", "accuracy_perf"}:
        loss_fn = nn.BCELoss()
    elif args.loss == "multitask":
        loss_fn = {
            "group": ProxyGroupLoss(crd),
            "accuracy": nn.BCELoss()
        }
    else:
        raise NotImplementedError

    if args.save_stats:
        full_stats = read_precomputed(args.threshold, args.loss, args.seed, args, architecture=args.architecture)
        assert full_stats is None, f"a stats file already exists for this config {args.threshold, args.loss, args.seed, args.architecture, args.lr, args.epochs}"
    predictor, full_stats = train(args,
                                 pop,
                                 predictor,
                                 crd,
                                 loss_fn,
                                 optimizer)

    if args.save_stats:
        store_precomputed(full_stats, args.threshold, args.loss, args.seed, args, architecture=args.architecture)

    # print(f"mean accuracy: {best_stats.acc:.3f}, mean group success: {best_stats.coop_groups:.3f},"
    #       f" in epoch {best_stats.epoch}")
    if args.save_figs:
        pop.reset_population()
        figs_list = run_simulation(crd, pop, predictor, args.figs_path, plot_figs=False, n_rounds=args.test_rounds, inner_colour="trust")
        imgs2gif(figs_list, args.figs_path)

    if args.plot_frames:
        pop.reset_population()
        run_simulation(crd, pop, predictor, None, plot_figs=True, n_rounds=args.test_rounds, inner_colour="trust")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train a performative predictor for collective risk dilemma")

    crd_parser(parser)

    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument("--architecture", type=str, choices={"mlp", "gnn", "gnn+linear", "gnn+mlp"}, default="mlp")
    misc.add_argument("--topology", type=str, choices={"lattice", "scale-free"}, default="scale-free")
    misc.add_argument("--sf_popsize", type=int, default=20, help="population size of scale-free network")
    misc.add_argument("--lattice_side", type=int, default=4, help="length of lattice side")
    misc.add_argument("-w","--log_wandb", action='store_true')
    misc.add_argument("--graph_seed", type=int, default=None, help="seed to generate scale-free network; use --seed if None")

    opt = parser.add_argument_group("optimization")
    opt.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    opt.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    opt.add_argument("--n_rounds", type=int, default=20, help="rounds per epoch")
    opt.add_argument("--seed", type=int, default=0, help="seed for Pytorch, and optionally for graph generation")
    opt.add_argument("--loss", type=str, choices={"individual", "group", "accuracy", "multitask", "accuracy_perf"}, default="group",
                     help="when training predictor, group loss maximizes successful groups "
                                 "whereas individual loss maximizes each agent's prob of cooperating ignoring "
                                 "topology; accuracy_perf computes a performative gradient of cross-entropy through "
                                 "labels, not only the gradient through predictions")
    opt.add_argument("--save_stats", action='store_true', help="whether to save stats")
    opt.add_argument("--stats_path", type=str, default="crd_stats", help="name of folder to store visualizations")
    opt.add_argument("--epsilon", type=float, default=None,
                     help="add epsilon (e.g. 1e-12) to denominator, in posterior trust computation, to avoid NaN when"
                          "trust is 1 and likelihood is 0")
    opt.add_argument("--block_trust_grad", action="store_true",
                     help="don't backprop through trust, for ablation")
    opt.add_argument("--block_prev_trust", action="store_true",
                     help="block only part of trust backprop; backprop through current prediction, block through previous trust")
    opt.add_argument("--use_custom_grad", action="store_true", help="Use manually implemented backward pass. "
                                                                    "Required to block components of the gradient.")

    sim = parser.add_argument_group("simulation", description="run a simulation after training and store as gif")
    sim.add_argument("--test_rounds", type=float, default=20, help="number of rounds for test time")
    sim.add_argument("--figs_path", type=str, default="figs", help="name of folder to store visualizations")
    sim.add_argument("--save_figs", action='store_true')
    sim.add_argument("--plot_frames", action='store_true')

    args = parser.parse_args()

    if not args.save_figs:
        args.figs_path = None
    elif not os.path.exists(args.figs_path):
        os.makedirs(args.figs_path)

    if args.topology == "scale-free":
        args.popsize = args.sf_popsize
    elif args.topology == "lattice":
        args.popsize = args.lattice_side * args.lattice_side
    else:
        raise NotImplementedError
    if args.graph_seed is None:
        args.graph_seed = args.seed

    main(args)
