import numpy as np
from math import pi, atan2
import matplotlib
import matplotlib.pyplot as plt
from fsqp_pacejka_model_spline import get_pacejka_problem, calc_step_rk4
#from optimal_lap_generator import get_pacejka_problem_optimal_lap
from feasible_sqp import *
from track_object import Track
from nlp_slacken_constraints import get_slackened_problem
from time import perf_counter
import csv
import pickle
matplotlib.use("TkAgg")

PLOT_LEN = 200

# target speeds
# 135 = 3.2
# 126 = 5.0
# 120 = 9.0

# cost params
p_Q1 = 1.5  # contouring cost
p_Q2 = 15.0  # lag cost
p_R1 = 1.0  # dtorque cost
p_R2 = 1.0  # dsteer cost
p_R3 = 1.0  # darclength cost
p_q = 3.6  # target speed
# size params
p_lr = 0.038
p_lf = 0.052
p_m = 0.181
p_I = 0.000505
# lateral force params
p_Df = 0.65
p_Cf = 1.5
p_Bf = 5.2
p_Dr = 1.0
p_Cr = 1.45
p_Br = 8.5
# longitudinal force params
p_Cm1 = 0.98028992
p_Cm2 = 0.01814131
p_Cd = 0.02750696
p_Croll = 0.08518052

model = np.array([p_Q1, p_Q2, p_R1, p_R2, p_R3, p_q,
                  p_lr, p_lf, p_m, p_I,
                  p_Df, p_Cf, p_Bf, p_Dr, p_Cr, p_Br,
                  p_Cm1, p_Cm2, p_Cd, p_Croll])


def get_ip_solver(x_sym, p_sym, f_exp, h_exp, pl=2):
    print("Compiling IPOPT solver...")
    t0 = perf_counter()
    IP_nlp = {'x': x_sym, 'p': p_sym, 'f': f_exp, 'g': h_exp}
    IP_solver = ca.nlpsol('S', 'ipopt', IP_nlp,
                          {'ipopt': {'print_level': pl, 'linear_solver': 'mumps', 'max_iter': 10}, 'jit': True,
                           'jit_options': {'flags': '-O3'}}) #'linear_solver': 'ma57'
    t1 = perf_counter()
    print("Finished compiling IPOPT solver in " + str(t1 - t0) + " seconds!")
    input("Press Enter to continue...")

    return IP_solver


def ipopt_ref(ip_solver, y, p, x_min, x_max, h_min, h_max):
    IP_sol = ip_solver(x0=y, p=p, lbx=x_min, ubx=x_max, lbg=h_min, ubg=h_max)
    IP_x = IP_sol['x'].full().T[0]
    IP_lam = IP_sol['lam_g'].full().T[0]
    
    return ip_solver.stats(), IP_x, IP_lam


track_length = 0.3 * (26 + 4*pi)


def track_pos(t):
    trackobj = Track()
    a = t

    t = 0.3
    trackobj.add_line(3.5 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, -pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, -pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, pi / 2)
    trackobj.add_line(4.5 * t)

    x, y = trackobj.point_at_arclength(a)
    x += 0.15
    y -= 1.05
    return x, y


# def track_pos(t):
#     trackobj = Track()
#     trackobj.add_line(l)
#     trackobj.add_turn(r, pi / 2)
#     trackobj.add_line(l)
#     trackobj.add_turn(r, pi / 2)
#     trackobj.add_line(l)
#     trackobj.add_turn(r, pi / 2)
#     trackobj.add_line(l)
#     trackobj.add_turn(r, pi / 2)
#
#     return trackobj.point_at_arclength(t)


# def track_pos(t):
#     T = [l, l + r * pi / 2,
#          2 * l + r * pi / 2, 2 * l + 2 * r * pi / 2,
#          3 * l + 2 * r * pi / 2, 3 * l + 3 * r * pi / 2,
#          4 * l + 3 * r * pi / 2, 4 * l + 4 * r * pi / 2]
#     t = fmod(t, track_length)
#
#     if t < T[0]:
#         return t, 0
#     elif t < T[1]:
#         return l + r * sin((t - T[0]) / r), r * (1 - cos((t - T[0]) / r))
#     elif t < T[2]:
#         return l + r, r + (t - T[1])
#     elif t < T[3]:
#         return l + r - r * (1 - cos((t - T[2]) / r)), l + r + r * sin((t - T[2]) / r)
#     elif t < T[4]:
#         return l - (t - T[3]), l + 2 * r
#     elif t < T[5]:
#         return - r * sin((t - T[4]) / r), l + 2 * r - r * (1 - cos((t - T[4]) / r))
#     elif t < T[6]:
#         return - r, l + r - (t - T[5])
#     else:
#         return - r + r * (1 - cos((t - T[6]) / r)), r - r * sin((t - T[6]) / r)


def track_params(t):
    trackarr = []

    for ti in t:
        xtrack, ytrack = track_pos(ti)

        eps = 0.001
        x_p, y_p = track_pos(ti + eps)
        x_m, y_m = track_pos(ti - eps)
        xrate = (x_p - x_m) / (2 * eps)
        yrate = (y_p - y_m) / (2 * eps)

        arcLength = ti
        tangentAngle = atan2(yrate, xrate)

        tracki = np.array([xtrack, ytrack, xrate, yrate, tangentAngle, arcLength])
        trackarr = np.concatenate((trackarr, tracki))

    return trackarr


def plot_track():
    track_N = 600
    track_t = np.linspace(0, track_length, track_N)

    xtrack = np.zeros(track_N)
    ytrack = np.zeros(track_N)
    xrate = np.zeros(track_N)
    yrate = np.zeros(track_N)

    for ii in range(track_N):
        t = track_t[ii]
        xtrack[ii], ytrack[ii] = track_pos(t)

        eps = 0.001
        x_p, y_p = track_pos(t + eps)
        x_m, y_m = track_pos(t - eps)
        xrate[ii] = (x_p - x_m) / (2 * eps)
        yrate[ii] = (y_p - y_m) / (2 * eps)

    half_track_width = 0.23

    # plot the track and its boundaries
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xtrack, ytrack, 'k--')
    ax.plot(xtrack + half_track_width * yrate, ytrack - half_track_width * xrate, 'k')
    ax.plot(xtrack - half_track_width * yrate, ytrack + half_track_width * xrate, 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax


def get_constraint_violation(y, p, h_func, h_min, h_max):
    # print(        np.max(np.concatenate((np.zeros((len(h_min), 1)), h_func(y, p) - h_max), 1), 1) +
    #     np.max(np.concatenate((np.zeros((len(h_min), 1)), h_min - h_func(y, p)), 1), 1))
    return np.sum(
        np.max(np.concatenate((np.zeros((len(h_min), 1)), h_func(y, p) - h_max), 1), 1) +
        np.max(np.concatenate((np.zeros((len(h_min), 1)), h_min - h_func(y, p)), 1), 1)
    )


def run_simulations(num_laps, solver_type, noises, filenames, show_plot):
    ax = plot_track()

    # problem setup...
    N = 30
    dt = 1.0 / N
    nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, r_exp, x_min, x_max, h_min, h_max = get_pacejka_problem(N, dt)
    #nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, r_exp, x_min, x_max, h_min, h_max = get_pacejka_problem_optimal_lap(N, dt)

    db_N = 6
    db_dt = dt
    db_nx, db_nh, db_npar, db_x_sym, db_p_sym, db_f_exp, db_f0_exp, db_h_exp, db_r_exp, db_x_min, db_x_max, db_h_min, db_h_max = get_pacejka_problem(db_N, db_dt)

    state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
    state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
    input_min = [-2.0, -15.0, 0.0]
    input_max = [2.0, 15.0, 5.0]

    h_func_hard = ca.Function('h', [x_sym, p_sym], [h_exp])
    h_min_hard = h_min
    h_max_hard = h_max

    slack_contouring = True
    slack_everything = False
    slack_idx = []
    if slack_contouring:
        slack_idx = [i for i in range(N * 9, N * 9 + N)]
        db_slack_idx = [i for i in range(db_N * 9, db_N * 9 + db_N)]
    if slack_everything:
        slack_idx = [i for i in range(nh)]
    slacken = slack_contouring or slack_everything
    if slacken:
        nx, nh, x_sym, f_exp, h_exp, x_min, x_max, h_min, h_max = \
            get_slackened_problem(nx, nh, x_sym, f_exp, h_exp, x_min, x_max, h_min, h_max, slack_idx)
        db_nx, db_nh, db_x_sym, db_f_exp, db_h_exp, db_x_min, db_x_max, db_h_min, db_h_max = \
            get_slackened_problem(db_nx, db_nh, db_x_sym, db_f_exp, db_h_exp, db_x_min, db_x_max, db_h_min, db_h_max, db_slack_idx)

    ns = 9
    nu = 3

    #f_func = ca.Function('f', [x_sym, p_sym], [f_exp])
    #h_func = ca.Function('h', [x_sym, p_sym], [h_exp])

    db_f_func = ca.Function('f', [db_x_sym, db_p_sym], [db_f_exp])
    db_h_func = ca.Function('h', [db_x_sym, db_p_sym], [db_h_exp])
    # db_IP_solver = get_ip_solver(db_x_sym, db_p_sym, db_f_exp, db_h_exp, pl=2)

    #lap_x = np.load("opt_lap_tightened_40.npy")
    lap_states = np.load("lap2_states.npy")
    lap_inputs = np.load("lap2_inputs.npy")
    load_state_idx = 30

    #N_lap = int((len(lap_x) - ns) / (ns + nu))
    N_lap = int((len(lap_states) - ns) / ns)
    # lap_states = [lap_x[i * ns: (i + 1) * ns] for i in range(N_lap)]
    # lap_inputs = [lap_x[(N_lap + 1) * ns + i * nu:(N_lap + 1) * ns + (i + 1) * nu] for i in range(N_lap)]

    # csvnames = ['lap_states', 'lap_inputs']
    # for csvname in csvnames:
    #     with open(csvname + '.csv', 'w', encoding='UTF8') as f:
    #         writer = csv.writer(f)
    #         for row in eval(csvname):
    #             writer.writerow(row)



    xN_flag = [0]

    # fsqp_solver = feasible_sqp(nx, np=npar)
    # fsqp_solver.init()
    # fsqp_solver.set_max_outer_it(1)
    # fsqp_solver.set_max_inner_it(100)
    # fsqp_solver.set_max_cum_nwsr(150)
    # if solver_type == 2:
    #     fsqp_solver.set_max_outer_it(1)
    #     fsqp_solver.set_max_inner_it(1)
    #     fsqp_solver.set_max_cum_nwsr(1000)
    if solver_type == "ip":
        IP_solver = get_ip_solver(x_sym, p_sym, f_exp, h_exp, pl=1)

    for qqq in range(len(noises)):
        noise = noises[qqq]
        filename = filenames[qqq]

        laps = 0
        first_iter = True

        current_speed = 0
        x_sim = []
        y_sim = []
        sim_traj, = ax.plot(x_sim, y_sim, 'b-o', alpha=0.5, markersize=3.0, linewidth=2.0)
        if show_plot:
            sim_traj, = ax.plot(x_sim, y_sim, linewidth=3.0, alpha=0.5)
            pred_traj, = ax.plot([], [], 'r-')
            terminal_traj, = ax.plot([], [], 'b-o', alpha=0.5, markersize=3.0, linewidth=2.0)

        #plt.show()
        full_results = []
        np.random.seed(int(noise * 100))
        constraint_violations = 0
        for iii in range(120 * num_laps):
            load_states = lap_states
            load_inputs = lap_inputs
            N_load = N_lap

            if first_iter:
                # x0 = np.array([0.15, -1.05, 0, 1.0, 0, 0, 0, 0, 0])
                # y = np.concatenate((lap_x[0:(N + 1) * ns], lap_x[(N_lap + 1) * ns:(N_lap + 1) * ns + N * nu]))
                # for i in range(31):
                #     y[i * 9:(i+1) * 9] = x0
                # for i in range(30):
                #     y[31 * 9 + i * 3: 31 * 9 + (i + 1) * 3] = np.array([0.0, 0.0, 0.0])

                y = np.concatenate((lap_x[0:(N + 1) * ns], lap_x[(N_lap + 1) * ns:(N_lap + 1) * ns + N * nu]))
                for i in range(31):
                    y[i * 9 + 0] += 0.15
                    y[i * 9 + 1] -= 1.05
                x0 = y[0:9]
            else:
                y_prev = solution
                # print(get_constraint_violation(y_prev, p, h_func, h_min, h_max))
                # print(fsqp_solver.get_constraint_violation_L1(y), fsqp_solver.get_constraint_violation_L1(y_prev))
                # if fsqp_solver.get_constraint_violation_L1(y) < fsqp_solver.get_constraint_violation_L1(
                #         y_prev) and fsqp_solver.get_constraint_violation_L1(y_prev) > 1e-8 and False:
                #     print("Reverting to previous solution due to infeasibility.")
                #     y_prev = y.reshape(y_prev.shape)
                # else:
                #     print("Using new solution.")
                #     pass
                prev_state = y_prev[0:ns].transpose()[0]
                prev_input = y_prev[(N + 1) * ns: (N + 1) * ns + nu].transpose()[0]

                for pp in range(len(prev_input)):
                    prev_input[pp] = max(prev_input[pp], input_min[pp])
                    prev_input[pp] = min(prev_input[pp], input_max[pp])

                # dt_noise = (0.8 * (2 * np.random.rand() - 1) + 1)
                # dt_noise = (0.5 * np.random.rand() + 1)
                # if np.random.rand() < 0.01:
                #     print("BIG NOISE")
                #     dt_noise = 3
                # dt_noise = 1
                #
                # noise_dt = dt_noise * dt + 0 * max(solver_runtime - dt, 0.0)
                # input_noise = prev_input
                # prev_input /= dt_noise
                x0 = prev_state + calc_step_rk4(prev_state, prev_input, model, dt, numerical=True)

                x0[0] += noise * (np.random.rand() - 0.5)
                x0[1] += noise * (np.random.rand() - 0.5)
                # x0[2] += noise * (np.random.rand() - 0.5)

                # x0[0] += noise * np.random.rand() * (-np.cos(x0[2]))
                # x0[1] += noise * np.random.rand() * (-np.sin(x0[2]))

                for pp in range(len(x0)):
                    x0[pp] = max(x0[pp], state_min[pp])
                    x0[pp] = min(x0[pp]
                                 , state_max[pp])

                u = y_prev[(N + 1) * ns + nu: (N + 1) * ns + N * nu]
                u = np.concatenate((u.squeeze(), u[-3:].squeeze()))

                y = x0
                xi = x0
                for i in range(N):
                    ui = u[nu * i: nu * (i + 1)]
                    xi = xi + calc_step_rk4(xi, ui, model, dt, True)

                    # if any((state_max - xi < 0.01) | (xi - state_min < 0.01)) or any((input_max - ui < 0.01) | (ui - input_min < 0.01)):
                    #     print(xi)
                    #     print(ui)

                    for pp in range(len(xi)):
                        xi[pp] = max(xi[pp], state_min[pp])
                        xi[pp] = min(xi[pp], state_max[pp])

                    y = np.concatenate((y, xi))
                y = np.concatenate((y.squeeze(), u))

                deadbeat = False
                if deadbeat:
                    db_model = model  # model
                    db_x0 = y[(N-db_N) * 9:(N-db_N+1) * 9]  # x0
                    db_xN = np.copy(load_states[load_state_idx])  # xN
                    db_xN[8] += track_length * laps
                    db_xN[2] += 2 * pi * laps
                    db_xN[0] += 0.15
                    db_xN[1] -= 1.05

                    db_states = y[(N-db_N) * 9:(N + 1) * 9]
                    db_inputs = y[(N + 1) * 9 + (N-db_N) * nu:]
                    db_y = np.concatenate((db_states, db_inputs, np.zeros(2 * db_N)))

                    db_t = np.array([db_states[i * ns + 8] for i in range(db_N + 1)])
                    db_track = track_params(db_t)
                    db_p = np.concatenate((db_model, db_x0, db_xN, db_track, xN_flag))

                    db_f_exp_p = db_f_func(db_x_sym, db_p)
                    db_h_exp_p = db_h_func(db_x_sym, db_p)

                    save_stdout = sys.stdout
                    # sys.stdout = open('trash', 'w')

                    t0 = perf_counter()
                    db_ip_stats, db_ip_sol, db_ip_lam = ipopt_ref(db_IP_solver, db_y, db_p, db_x_min, db_x_max, db_h_min, db_h_max)
                    t1 = perf_counter()

                    results = dict()
                    results['param'] = db_p
                    results['primal'] = db_y
                    results['ip_time'] = t1 - t0
                    results['ip_stats'] = db_ip_stats
                    results['ip_sol'] = db_ip_sol
                    results['ip_lam'] = db_ip_lam

                    sys.stdout = save_stdout
                    # print(ip_x_db[0:9])
                    # print(y[(N - N_db) * 9:(N - N_db + 1) * 9])
                    print("Deadbeat controller feasiblility ", results['ip_stats']['success'], results['ip_time'])

                    y[(N-db_N) * 9:(N + 1) * 9] = results['ip_sol'][0:(db_N + 1) * 9]
                    y[(N + 1) * 9 + (N-db_N) * nu:] = results['ip_sol'][(db_N + 1) * 9:(db_N + 1) * 9 + (db_N) * nu]
                    # input("Press Enter to continue...")

            xN = np.copy(load_states[load_state_idx])
            xN[8] += track_length * laps
            xN[2] += 2 * pi * laps
            xN[0] += 0.15
            xN[1] -= 1.05
            # y[N*9:(N+1)*9] = xN
            # print("xN", xN)
            # print("xi", y[N*ns:(N+1)*ns])
            # load_state_idx += 1
            if x0[8] > track_length * laps:
                print("Lap " + str(laps) + " completed at time step " + str(iii))
                load_state_idx = 0
                laps += 1

            first_iter = False

            x_pred = [y[i * ns + 0] for i in range(N + 1)]
            y_pred = [y[i * ns + 1] for i in range(N + 1)]

            t = np.array([y[i * ns + 8] for i in range(N + 1)])

            track = track_params(t)

            p = np.concatenate((model, x0, xN, track, xN_flag))

            if slacken:
                # slack = np.max(np.concatenate((np.zeros((len(h_min_hard), 1)), h_min_hard - h_func_hard(y, p), h_func_hard(y, p) - h_max_hard), 1), 1)
                # y = np.concatenate((y, slack))

                slack_upper = 0 * np.max(
                    np.concatenate((np.zeros((len(h_min_hard), 1)), h_func_hard(y, p) - h_max_hard), 1), 1)
                slack_lower = 0 * np.max(
                    np.concatenate((np.zeros((len(h_min_hard), 1)), h_min_hard - h_func_hard(y, p)), 1), 1)
                y = np.concatenate((y, slack_upper[slack_idx], slack_lower[slack_idx]))
                # y = np.concatenate((y, np.zeros(len(slack_idx) * 2)))

            # if solver_type == 0 or solver_type == 2:
            #     fsqp_solver.set_primal_guess(y.reshape((y.shape[0], 1)))
            #     fsqp_solver.set_param(p.reshape((p.shape[0], 1)))

            #     t0 = perf_counter()
            #     fsqp_solver.solve()
            #     t1 = perf_counter()

            #     results = dict()
            #     results['param'] = p
            #     results['primal'] = y
            #     results['fsqp_time'] = t1 - t0
            #     results['fsqp_stats'] = fsqp_solver.get_stats()
            #     results['fsqp_sol'] = fsqp_solver.get_primal_sol().transpose()[0]
            #     results['fsqp_dual'] = fsqp_solver.get_dual_sol(348).transpose()[0]

            #     iters = np.count_nonzero(results['fsqp_stats']['cpu_time'])
            #     cv = get_constraint_violation(results['fsqp_sol'], results['param'], h_func, h_min, h_max)
            #     converged = (cv < 1e-3 and iters < 30)

            #     solution = fsqp_solver.get_primal_sol()
            #     if solver_type == 0 and not converged:
            #         solution = y.reshape((y.shape[0], 1))
            if solver_type == 1:
                # f_exp_p = f_func(x_sym, p)
                # h_exp_p = h_func(x_sym, p)

                # save_stdout = sys.stdout
                # sys.stdout = open('trash', 'w')

                t0 = perf_counter()
                ip_stats, ip_sol, ip_lam = ipopt_ref(IP_solver, y, p, x_min, x_max, h_min, h_max)
                t1 = perf_counter()

                # sys.stdout = save_stdout

                results = dict()
                results['param'] = p
                results['primal'] = y
                results['ip_time'] = t1 - t0
                results['ip_stats'] = ip_stats
                results['ip_sol'] = ip_sol
                results['ip_lam'] = ip_lam

                solution = ip_sol.reshape((ip_sol.shape[0], 1))
            else:
                print("Invalid solver type!!!")

            x_pred = [solution[i * ns + 0] for i in range(N + 1)]
            y_pred = [solution[i * ns + 1] for i in range(N + 1)]

            t = np.array([solution[i * ns + 8] for i in range(N + 1)])

            print(iii)
            full_results.append(results)

            if len(x_sim) > PLOT_LEN:
                x_sim.append(x0[0])
                y_sim.append(x0[1])
                x_sim.pop(0)
                y_sim.pop(0)
            else:
                x_sim.append(x0[0])
                y_sim.append(x0[1])

            current_speed = np.sqrt(x0[3] ** 2 + x0[4] ** 2)

            if show_plot:
                sim_traj.set_data(x_sim, y_sim)
                pred_traj.set_data(x_pred, y_pred)
                terminal_traj.set_data([xN[0]], [xN[1]])

                plt.draw()
                plt.pause(0.001)

        pickle.dump(full_results, open("first_full_results_file" + filename + ".p", "wb"))


if __name__ == "__main__":
    # install_dependencies(hsl_lib_path="/usr/local/lib/libcoinhsl.so")
    solver_type_strings = ["ip"] #, "rti_O3"] # ["fsqp_O3", "ip_O3", "rti_O3"]
    for solver_type in solver_type_strings:
        # noises = [0.01, 0.02, 0.04, 0.08]
        noises = [0.00]
        filenames = ["filename_" + str(int(noise * 100)) for noise in noises]
        run_simulations(10, solver_type, noises, filenames, True)


