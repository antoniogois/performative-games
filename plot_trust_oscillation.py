import argparse
import numpy as np
import matplotlib.pyplot as plt

def step_3nodes(trusts, preds, alpha=0.8, cr=0.2, current_iter=0):
    """chain3, threshold 2/3"""
    act0 = trusts[0]*preds[1] + (1-trusts[0])*alpha > cr
    act1 = trusts[1]*(preds[0]* (1-preds[2]) + preds[2] *(1-preds[0]))\
      + (1- trusts[1])*2*alpha* (1-alpha) > cr
    act2 = trusts[2]*preds[1] + (1-trusts[2])*alpha > cr
    actions = [act0, act1, act2]

    def update_trust(trust, preds, acts):
        likelihood1 = np.prod([p if acts[i] else 1-p for i,p in enumerate(preds)])
        likelihood2 = np.prod([p if acts[i] else 1-p for i,p in enumerate([alpha]*len(acts))])
        return trust*likelihood1/(trust*likelihood1 + (1-trust)*likelihood2)

    new_trusts = [0,0,0]
    new_trusts[0] = update_trust(trusts[0], [preds[1]], [actions[1]])
    new_trusts[1] = update_trust(trusts[1], [preds[0],preds[2]], [actions[0],actions[2]])
    new_trusts[2] = update_trust(trusts[2], [preds[1]], [actions[1]])

    # print(actions)
    # print(new_trusts)
    print(f"iter: {current_iter:3d}", actions, "\t trust:", new_trusts, "\t last pred:", preds)
    return new_trusts, actions


def get_optimal_theta2(trust1, cr, alpha):
    """Compute largest theta2 which still makes node1 cooperate.
    This together with theta0=1 makes node1 cooperate while minimally reducing node1's trust"""
    epsilon = 0.01
    return 1 - (cr - (1 - trust1) * (2 * alpha * (1 - alpha))) / trust1 - epsilon

def get_optimal_theta1(trust02, cr, alpha):
    """Compute lowest theta1 which still makes node0 and node2 cooperate.
    This minimally reduces trust of 0 and 2"""
    epsilon = 0.01
    theta1 = epsilon + (cr - (1 - trust02) * alpha) / trust02
    assert theta1 > 0
    return theta1

def compute_avg_welfare(actions, c=0, r=1):
    # with c=0, r=1, we get proportion of groups that overcame threshold, without considering cost c nor risk r
    welfare = 3
    if not (actions[0] and actions[1]):
        welfare -= r  # node0 didn't overcome
    if not sum(actions)>=2:
        welfare -= r # node1 didn't overcome
    if not (actions[1] and actions[2]):
        welfare -= r  # node2 didn't overcome
    # discount coop cost
    welfare -= c*sum(actions)
    return welfare / 3


def make_plot(trust1, pred2, welfare, discount_rate):
    # Create the x-axis values for each list
    x_trust1 = list(range(len(trust1)))
    x_pred2 = list(range(1, len(pred2) + 1))
    x_welfare = list(range(1, len(welfare) + 1))

    plt.figure(figsize=(6.4, 3.8))  # default figsize=(6.4, 4.8)
    # Plot the lines
    plt.plot(x_welfare, welfare, label='Proportion of group successes', color='#A9A9A9', linestyle='--', linewidth=1.5)
    plt.plot(x_pred2, pred2, label='Accuracy for right-hand node', color='lightblue', linestyle='-.', linewidth=1.5)
    plt.plot(x_trust1, trust1, label='Trust for center node', color='blue', linewidth=3)

    # Add labels and title
    plt.xlabel('Time Step')
    # plt.ylabel('Value')
    plt.title('Trust, Accuracy, and Welfare Over Time')  # \nfor 3-node Chain (0-1-2), Threshold=2/3')

    # Add legend
    plt.legend(loc="lower right")

    # Display the plot
    plt.savefig(f'trust_{discount_rate}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main(args):
    # trusts =[.5, .5, .5]
    # trusts = step_3nodes(trusts, [.9, .9, .1])

    alpha = .8
    trusts = [.5, .5, .5]
    cr = .6  # c/r
    r = .5
    c = cr * r  # c = c/r * r
    print(f"cr: {cr}")
    # trusts = [0.9886021927250579, 0.05882352941176468, 0.9886021927250579]
    print(f"initial trust: {trusts}")

    t1_record = [trusts[1]]  # store trust record of node1 (trust of 0 and 2 always increase)
    pred2_record = []  # store accuracy of node2 prediction (accuracy of 0 and 1 always 100% ) — except if we boost trust with 0-1-0 instead of 0-0-0; but impact is negligible
    welfare_record = []  # store binary welfare; in this setting either all groups or none succeeded

    trust_boosts = 1  # how many boosts to give, after minimum trust is reached
    remaining_boosts = 0
    discount_rate = args.discount_rate
    for i in range(40):
        if discount_rate == "high":
            theta0, theta1, theta2 = 0, 0, 0
        else:
            theta0, theta1 = 1, 1
            theta2 = get_optimal_theta2(trusts[1], cr, alpha)
            if theta2 <= 0 or remaining_boosts > 0:
                if remaining_boosts == 0:
                    remaining_boosts = trust_boosts
                remaining_boosts -= 1
                if discount_rate == "low":
                    assert trusts[0] - trusts[2] < 1e-3
                    theta1 = get_optimal_theta1(trusts[0], cr, alpha)  # makes 0,2 cooperate while minimally harming their trust
                    theta0 = theta2 = 1
                elif discount_rate == "med":
                    theta0, theta1, theta2 = 0, 0, 0
                else:
                    raise NotImplementedError
        trusts, actions = step_3nodes(trusts, [theta0, theta1, theta2], alpha=alpha, cr=cr, current_iter=i)
        t1_record.append(trusts[1])
        pred2_record.append(
            theta2 if actions[2] else 1 - theta2)  # we only predict 0 when action2=False, with accuracy 1-0
        welfare_record.append(compute_avg_welfare(actions))

    print(t1_record)  # [:-1])
    print(pred2_record)
    print(welfare_record)

    make_plot(t1_record, pred2_record, welfare_record, discount_rate)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--discount_rate', type=str, default="med", choices=["low", "med"])
    args = argparser.parse_args()
    main(args)
