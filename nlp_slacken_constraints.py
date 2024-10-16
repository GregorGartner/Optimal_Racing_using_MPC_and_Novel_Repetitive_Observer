import numpy as np
import casadi as ca

def get_slackened_problem(nx, nh, x, f_exp, h_exp, x_min, x_max, h_min, h_max, slack_idx):

    ns = len(slack_idx)
    slack_upper = ca.MX.sym('slack_upper', ns)
    slack_lower = ca.MX.sym('slack_lower', ns)

    h_exp_new = []
    h_min_new = []
    h_max_new = []
    j = 0
    for i in range(nh):
        h = h_exp[i]

        if i in slack_idx:
            su = slack_upper[j]
            sl = slack_lower[j]
            j += 1

            h_exp_new = ca.vertcat(h_exp_new, h - su)
            h_exp_new = ca.vertcat(h_exp_new, h + sl)
            h_min_new.append(-1e10)
            h_min_new.append(h_min[i])
            h_max_new.append(h_max[i])
            h_max_new.append(1e10)
        else:
            h_exp_new = ca.vertcat(h_exp_new, h)
            h_min_new.append(h_min[i])
            h_max_new.append(h_max[i])

    h_min_new = np.array(h_min_new)
    h_max_new = np.array(h_max_new)

    x = ca.veccat(x, slack_upper, slack_lower)
    x_min = np.concatenate((x_min, np.zeros(2 * ns)))
    x_max = np.concatenate((x_max, 1e10 * np.ones(2 * ns)))

    nx = len(x_min)
    nh = len(h_min_new)

    for i in range(ns):
        f_exp += 1e3 * slack_upper[i]
        f_exp += 1e3 * slack_lower[i]

    return nx, nh, x, f_exp, h_exp_new, x_min, x_max, h_min_new, h_max_new
