import numpy as np
from math import pi, atan2
import matplotlib
import matplotlib.pyplot as plt
from fsqp_pacejka_model import get_pacejka_problem, calc_step_rk4
# from fsqp_pacejka_model_deadbeat_simple import get_pacejka_problem_deadbeat_simple
from feasible_sqp import *
from track_object import Track
from nlp_slacken_constraints import get_slackened_problem
from time import perf_counter
import csv
matplotlib.use("TkAgg")

PLOT_LEN = 200

# cost params
p_Q1 = 1.5  # contouring cost
p_Q2 = 15.0  # lag cost
p_R1 = 0.5  # dtorque cost
p_R2 = 0.5  # dsteer cost
p_R3 = 0.2  # darclength cost
p_q = 2.0  # target speed
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


def ipopt_ref(x, f_exp, h_exp, x_guess, x_min, x_max, h_min, h_max, pl=2):
    IP_nlp = {'x': x, 'f': f_exp, 'g': h_exp}
    IP_solver = ca.nlpsol('S', 'ipopt', IP_nlp, {'ipopt': {'print_level': pl, 'max_iter': 3000}})
    IP_sol = IP_solver(x0=x_guess, lbx=x_min, ubx=x_max, lbg=h_min, ubg=h_max)
    IP_x = IP_sol['x'].full().T[0]
    IP_lam = IP_sol['lam_g'].full().T[0]
    return IP_solver.stats()['success'], IP_solver.stats()['iterations']['inf_pr'][0], IP_solver.stats()['iterations']['inf_pr'][-1], IP_x, IP_lam


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


if __name__ == "__main__":
    ax = plot_track()

    # problem setup...
    N = 30
    dt = 1.0 / N
    nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, r_exp, x_min, x_max, h_min, h_max = get_pacejka_problem(N, dt)

    h_func_hard = ca.Function('h', [x_sym, p_sym], [h_exp])
    h_min_hard = h_min
    h_max_hard = h_max

    slack_contouring = True
    slack_everything = False
    slack_idx = []
    if slack_contouring:
        slack_idx = [i for i in range(N * 9, N * 9 + N)]
    if slack_everything:
        slack_idx = [i for i in range(nh)]
    slacken = slack_contouring or slack_everything
    if slacken:
        nx, nh, x_sym, f_exp, h_exp, x_min, x_max, h_min, h_max = \
            get_slackened_problem(nx, nh, x_sym, f_exp, h_exp, x_min, x_max, h_min, h_max, slack_idx)

    ns = 9
    nu = 3

    fsqp_solver = feasible_sqp(nx, np=npar)
    fsqp_solver.init()
    fsqp_solver.set_max_outer_it(1)
    fsqp_solver.set_max_inner_it(100)
    # fsqp_solver.set_max_nwsr(10000)

    # solver.set_kappa_tilde(float(10000))
    # solver.set_kappa_bar(float(10000))
    # solver.set_kappa_max(float(10000))

    f_func = ca.Function('f', [x_sym, p_sym], [f_exp])
    h_func = ca.Function('h', [x_sym, p_sym], [h_exp])

    # deadbeat = False
    # if deadbeat:
    #     N_db = 5
    #     nx_db, nh_db, npar_db, x_sym_db, p_sym_db, f_exp_db, f0_exp_db, h_exp_db, x_min_db, x_max_db, h_min_db, h_max_db = get_pacejka_problem_deadbeat_simple(
    #         N, dt, N_db)
    #     f_func_db = ca.Function('f_db', [x_sym_db, p_sym_db], [f_exp_db])
    #     h_func_db = ca.Function('h_db', [x_sym_db, p_sym_db], [h_exp_db])

    count_success = 0
    count_fail = 0
    count_infeasible = 0
    count_deadbeat_infeasible = 0

    load_terminal_set = True
    if not load_terminal_set:  # initialize fsqp
        x0 = np.array([0, 0, 0, 1.0, 0, 0, 0, 0, 0])
        xi = x0
        y = x0
        u = []
        for i in range(N):
            ui = np.array([0.0, 0.0, 0.0])
            theta = xi[8]
            xi = xi + calc_step_rk4(xi, ui, model, dt, True)
            xi[8] = xi[0]
            dtheta = (xi[8] - theta) / dt
            ui[2] = dtheta
            y = np.concatenate((y, xi))
            u = np.concatenate((u, ui))
        y = np.concatenate((y, u))
        t = np.array([y[i * ns + 8] for i in range(N+1)])

        xN = x0
        xN_flag = [0]

        track = track_params(t)

        p = np.concatenate((model, x0, xN, track, xN_flag))
        # p = np.concatenate((model, x0, track))

        fsqp_solver.set_primal_guess(y)
        fsqp_solver.set_param(p)
        # print(solver.get_constraint_violation_L1(y))

        fsqp_solver.solve()

    current_speed= 0
    x_sim = []
    y_sim = []
    # sim_traj, = ax.plot(x_sim, y_sim, 'b-o', alpha=0.5, markersize=3.0, linewidth=2.0)
    sim_traj, = ax.plot(x_sim, y_sim, linewidth=3.0, alpha=0.5)
    pred_traj, = ax.plot([], [], 'r-')

    if not load_terminal_set:

        lap1_states = []
        lap1_inputs = []
        lap1_saved = False

        lap2_states = []
        lap2_inputs = []
        lap2_saved = False

        laps = 0

        while True:  # run fsqp
            y_prev = fsqp_solver.get_primal_sol()
            prev_state = y_prev[0:ns]
            prev_input = y_prev[(N+1)*ns: (N+1)*ns + nu]

            x0 = prev_state + calc_step_rk4(prev_state, prev_input, model, dt, numerical=True)
            xN = x0
            xN_flag = [0]

            u = y_prev[(N+1)*ns + nu:]
            uN = u[-nu:]
            u = np.concatenate((u, uN))

            prev_state = y_prev[0:ns]
            prev_input = y_prev[(N+1)*ns: (N+1)*ns + nu]
            if prev_state[8] < track_length:
                lap1_states.append(prev_state)
                lap1_inputs.append(prev_input)
            if prev_state[8] > track_length and not lap1_saved:
                lap1_saved = True
                np.save("ts_data/lap1_states.npy", lap1_states)
                np.save("ts_data/lap1_inputs.npy", lap1_inputs)

            if track_length < prev_state[8] < 2 * track_length:
                lap2_states.append(prev_state)
                lap2_inputs.append(prev_input)
            if prev_state[8] > 2 * track_length and not lap2_saved:
                lap2_saved = True
                np.save("ts_data/lap2_states.npy", lap2_states)
                np.save("ts_data/lap2_inputs.npy", lap2_inputs)

            x_pred = [x0[0]]
            y_pred = [x0[1]]

            y = x0
            xi = x0
            for i in range(N):
                ui = u[nu * i: nu * (i+1)]
                xi = xi + calc_step_rk4(xi, ui, model, dt, True)
                y = np.concatenate((y, xi))

                x_pred.append(xi[0])
                y_pred.append(xi[1])

            y = np.concatenate((y, u))
            t = np.array([y[i * ns + 8] for i in range(N + 1)])

            track = track_params(t)

            p = np.concatenate((model, x0, xN, track, xN_flag))
            # p = np.concatenate((model, x0, track))

            fsqp_solver.set_primal_guess(y)
            fsqp_solver.set_param(p)

            fsqp_solver.solve()

            f_exp_p = f_func(x_sym, p)
            h_exp_p = h_func(x_sym, p)
            #
            ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, y, x_min, x_max, h_min, h_max)

            print('initial guess constraint violation: ', fsqp_solver.get_constraint_violation_L1(y))
            print('fsqp constraint violation: ', fsqp_solver.get_constraint_violation_L1(fsqp_solver.get_primal_sol()))
            # print('ipopt constraint violation: ', solver.get_constraint_violation_L1(ip_x))

            x_sim.append(x0[0])
            y_sim.append(x0[1])
            sim_traj.set_data(x_sim, y_sim)
            pred_traj.set_data(x_pred, y_pred)

            plt.draw()
            plt.pause(0.01)

    else:
        tightened = True
        if tightened:
            lap_x = np.load("terminal_set_fsqp/ts_data/opt_lap_tightened.npy")
            init_x = np.load("terminal_set_fsqp/ts_data/opt_init_tightened.npy")
        else:
            lap_x = np.load("terminal_set_fsqp/ts_data/opt_lap.npy")
            init_x = np.load("terminal_set_fsqp/ts_data/opt_init.npy")
        load_state_idx = 30
        first_iter = True

        N_lap = int((len(lap_x)-ns) / (ns + nu))
        lap_states = [lap_x[i*ns: (i+1)*ns] for i in range(N_lap)]
        lap_inputs = [lap_x[(N_lap+1)*ns + i*nu:(N_lap+1)*ns + (i+1)*nu] for i in range(N_lap)]

        N_init = int((len(init_x)-ns) / (ns + nu))
        init_states = [init_x[i*ns: (i+1)*ns] for i in range(N_init)]
        init_inputs = [init_x[(N_init+1)*ns + i*nu:(N_init+1)*ns + (i+1)*nu] for i in range(N_init)]

        csvnames = ['init_states', 'init_inputs', 'lap_states', 'lap_inputs']
        for csvname in csvnames:
            with open('terminal_set_fsqp/ts_data/' + csvname + '.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                for row in eval(csvname):
                    writer.writerow(row)

        laps = 0
        xN_flag = [1]

        solver_runtime = 0
        total_runtime = 0
        number_of_solves = 0
        iii = 0
        while True:  # run fsqp
            iii += 1
            if laps == 0:
                load_states = init_states
                load_inputs = init_inputs
                N_load = N_init
            else:
                load_states = lap_states
                load_inputs = lap_inputs
                N_load = N_lap

            if first_iter:
                y = np.concatenate((init_x[0:(N+1)*ns], init_x[(N_init+1)*ns:(N_init+1)*ns+N*nu]))
                x0 = np.array([0, 0, 0, 1.0, 0, 0, 0, 0, 0])
            else:
                y_prev = fsqp_solver.get_primal_sol()
                print(fsqp_solver.get_constraint_violation_L1(y), fsqp_solver.get_constraint_violation_L1(y_prev))
                if fsqp_solver.get_constraint_violation_L1(y) < fsqp_solver.get_constraint_violation_L1(y_prev) and fsqp_solver.get_constraint_violation_L1(y_prev) > 1e-8:
                    # print("Reverting to previous solution due to infeasibility.")
                    y_prev = y
                else:
                    # print("Using new solution.")
                    pass
                prev_state = y_prev[0:ns]
                prev_input = y_prev[(N + 1) * ns: (N + 1) * ns + nu]

                dt_noise = (0.4 * (2 * np.random.rand() - 1) + 1)
                # dt_noise = (0.5 * np.random.rand() + 1)
                # if np.random.rand() < 0.01:
                #     print("BIG NOISE")
                #     dt_noise = 3
                # dt_noise = 1

                noise_dt = dt_noise * dt + 0*max(solver_runtime - dt, 0.0)
                input_noise = prev_input
                prev_input /= dt_noise
                x0 = prev_state + calc_step_rk4(prev_state, prev_input, model, noise_dt, numerical=True)

                # noise = 0.03
                # x0[0] += noise * (np.random.rand() - 0.5)
                # x0[1] += noise * (np.random.rand() - 0.5)

                u = y_prev[(N + 1) * ns + nu: (N + 1) * ns + N * nu]
                u = np.concatenate((u.squeeze(), load_inputs[(load_state_idx - 1 + N_load) % N_load]))

                y = x0
                xi = x0
                for i in range(N):
                    ui = u[nu * i: nu * (i + 1)]
                    xi = xi + calc_step_rk4(xi, ui, model, dt, True)
                    y = np.concatenate((y, xi))
                y = np.concatenate((y.squeeze(), u))

            xN = np.copy(load_states[load_state_idx])
            xN[8] += track_length * laps
            xN[2] += 2 * pi * laps
            # y[N*9:(N+1)*9] = xN
            # print("xN", xN)
            # print("xi", y[N*ns:(N+1)*ns])
            load_state_idx += 1
            if load_state_idx >= N_load:
                load_state_idx = 0
                laps += 1

            first_iter = False

            x_pred = [y[i * ns + 0] for i in range(N+1)]
            y_pred = [y[i * ns + 1] for i in range(N+1)]

            t = np.array([y[i * ns + 8] for i in range(N + 1)])

            track = track_params(t)

            p = np.concatenate((model, x0, xN, track, xN_flag))
            fsqp_solver.set_param(p)

            if slacken:
                # slack = np.max(np.concatenate((np.zeros((len(h_min_hard), 1)), h_min_hard - h_func_hard(y, p), h_func_hard(y, p) - h_max_hard), 1), 1)
                # y = np.concatenate((y, slack))

                slack_upper = 0*np.max(
                    np.concatenate((np.zeros((len(h_min_hard), 1)), h_func_hard(y, p) - h_max_hard), 1), 1)
                slack_lower = 0*np.max(
                    np.concatenate((np.zeros((len(h_min_hard), 1)), h_min_hard - h_func_hard(y, p)), 1), 1)
                y = np.concatenate((y, slack_upper[slack_idx], slack_lower[slack_idx]))
                # y = np.concatenate((y, np.zeros(len(slack_idx) * 2)))

            print(fsqp_solver.get_constraint_violation_L1(y))

            feasible_db = True
            # if deadbeat:
            #     y_db = np.zeros(nx_db)
            #     p_db = np.zeros(npar_db)
            #
            #     p_db[0:20] = p[0:20]  # model
            #     p_db[20:20 + 9] = p[20:20 + 9]  # x0
            #     p_db[20 + 9:20 + 9 + 9] = p[20 + 9:20 + 9 + 9]  # xN
            #     p_db[20 + 9 + 9:] = y[(N + 1) * 9:]  # input guess
            #
            #     y_db = y[(N + 1) * 9:(N + 1) * 9 + N_db * 3]  # inputs
            #
            #     f_exp_p_db = f_func_db(x_sym_db, p_db)
            #     h_exp_p_db = h_func_db(x_sym_db, p_db)
            #
            #     save_stdout = sys.stdout
            #     sys.stdout = open('trash', 'w')
            #     feasible_db, cv_init_db, cv_db, ip_x_db, ip_lam_db = ipopt_ref(x_sym_db, f_exp_p_db, h_exp_p_db,
            #                                                                    y_db, x_min_db, x_max_db, h_min_db,
            #                                                                    h_max_db, pl=3)
            #     sys.stdout = save_stdout
            #     # # print(ip_x_db[0:9])
            #     # # print(y[(N - N_db) * 9:(N - N_db + 1) * 9])
            #     # print("Deadbeat controller feasiblility ", feasible_db, cv_init_db, cv_db)
            #     # if not feasible_db:
            #     #     count_deadbeat_infeasible += 1
            #
            #     # ip_x_db = y_db
            #     y[(N + 1) * 9:(N + 1) * 9 + N_db * 3] = ip_x_db
            #     for j in range(N):
            #         y[(j + 1) * 9:(j + 2) * 9] = y[(j + 0) * 9:(j + 1) * 9] + calc_step_rk4(
            #             y[(j + 0) * 9:(j + 1) * 9], y[(N + 1) * 9 + j * 3: (N + 1) * 9 + (j + 1) * 3], p[0:20],
            #             1.0 / 30.0, True)

            print(fsqp_solver.get_constraint_violation_L1(y))
            fsqp_solver.set_primal_guess(y)

            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')
            time_before = perf_counter()

            fsqp_solver.solve()

            # f_exp_p = f_func(x_sym, p)
            # h_exp_p = h_func(x_sym, p)

            # ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, y, x_min, x_max, h_min, h_max)

            solver_runtime = perf_counter() - time_before
            number_of_solves += 1
            total_runtime += solver_runtime
            sys.stdout = save_stdout

            print('runtime ', solver_runtime)
            print('average runtime ', total_runtime / number_of_solves)
            # print(solver.get_constraint_violation_L1(y), solver.get_constraint_violation_L1(solver.get_primal_sol()))

            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')
            f_exp_p = f_func(x_sym, p)
            h_exp_p = h_func(x_sym, p)
            feasible = True
            # feasible, cv_init, cv, ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, y, x_min, x_max, h_min, h_max, pl=5)
            sys.stdout = save_stdout

            # print('initial guess constraint violation: ', solver.get_constraint_violation_L1(y))
            # print('fsqp constraint violation: ', solver.get_constraint_violation_L1(solver.get_primal_sol()))
            # print('ipopt constraint violation: ', solver.get_constraint_violation_L1(ip_x))

            if not feasible:
                count_infeasible += 1
            elif fsqp_solver.get_constraint_violation_L1(fsqp_solver.get_primal_sol()) < fsqp_solver.get_constraint_violation_L1(
                    y) or fsqp_solver.get_constraint_violation_L1(fsqp_solver.get_primal_sol()) < 1e-6:
                count_success += 1
            else:
                count_fail += 1

            if not feasible_db:
                count_deadbeat_infeasible += 1

            print(iii, "   ", count_success, count_fail, count_infeasible,
                  count_deadbeat_infeasible)

            if len(x_sim) > PLOT_LEN:
                x_sim.append(x0[0])
                y_sim.append(x0[1])
                x_sim.pop(0)
                y_sim.pop(0)
            else:
                x_sim.append(x0[0])
                y_sim.append(x0[1])

            current_speed = np.sqrt(x0[3]**2 + x0[4]**2)
            sim_traj.set_data(x_sim, y_sim)
            pred_traj.set_data(x_pred, y_pred)

            print(fsqp_solver.get_primal_sol()[369:])

            plt.draw()
            plt.pause(0.001)






