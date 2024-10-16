import numpy as np
import casadi as ca
from math import pi
import matplotlib
import matplotlib.pyplot as plt
from fsqp_pacejka_model import calc_step_rk4
#from terminal_set_rti import track_params, plot_track, model, ipopt_ref, track_length #, get_ip_solver
from terminal_set_rti_new2_no_terminal_set import track_params, plot_track, model, ipopt_ref, track_length, get_ip_solver
import yaml
matplotlib.use("TkAgg")

# track spline
def get_demo_track_spline():
    track = yaml.load(open("DEMO_TRACK.yaml", "r"))["track"]
    trackLength = track["trackLength"]
    indices = [i for i in range(1500, 5430, 10)]
    x = np.array(track["xCoords"])[indices]
    y = np.array(track["yCoords"])[indices]
    dx = np.array(track["xRate"])[indices]
    dy = np.array(track["yRate"])[indices]
    t = np.array(track["arcLength"])[indices] - 0.5 * trackLength

    t_sym = ca.SX.sym('t')

    x_spline = ca.interpolant('LUT', 'bspline', [t], x)
    x_exp = x_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    x_fun = ca.Function('f', [t_sym], [x_exp])

    y_spline = ca.interpolant('LUT', 'bspline', [t], y)
    y_exp = y_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    y_fun = ca.Function('f', [t_sym], [y_exp])

    dx_spline = ca.interpolant('LUT', 'bspline', [t], dx)
    dx_exp = dx_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    dx_fun = ca.Function('f', [t_sym], [dx_exp])

    dy_spline = ca.interpolant('LUT', 'bspline', [t], dy)
    dy_exp = dy_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    dy_fun = ca.Function('f', [t_sym], [dy_exp])

    return x_fun, y_fun, dx_fun, dy_fun


def get_pacejka_problem_optimal_lap(N, dt, use_spline):

    # state and input dimensions
    ns = 9
    nu = 3

    # state and input constraints
    state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
    state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
    input_min = [-2.0, -15.0, 0.0]
    input_max = [2.0, 15.0, 5.0]

    # symbolic variables
    states = ca.SX.sym('s', ns, N + 1)
    xp = states[0, :]
    yp = states[1, :]
    yaw = states[2, :]
    vx = states[3, :]
    vy = states[4, :]
    omega = states[5, :]
    T = states[6, :]
    delta = states[7, :]
    theta = states[8, :]

    inputs = ca.SX.sym('u', nu, N)
    dT = inputs[0, :]
    ddelta = inputs[1, :]
    dtheta = inputs[2, :]

    # track params
    track = ca.SX.sym('t', 6, N+1)
    xd = track[0, :]
    yd = track[1, :]
    xdgrad = track[2, :]
    ydgrad = track[3, :]
    phid = track[4, :]
    ttheta = track[5, :]

    # dynamics params...
    model = ca.SX.sym('p', 20)
    # cost params
    p_Q1 = model[0]
    p_Q2 = model[1]
    p_R1 = model[2]
    p_R2 = model[3]
    p_R3 = model[4]
    p_dtheta_target = model[5]

    # cost...
    r_exp = []
    r0_exp = []

    # x_spline, y_spline, dx_spline, dy_spline = get_demo_track_spline()

    # input costs weighed by R and stacked to vector (will later 
    # take dot product with itseld to get least squares cost)
    for i in range(N):
        r_input_i = ca.vertcat(
            p_R1 * dT[i],
            p_R2 * ddelta[i],
            p_R3 * (dtheta[i] - p_dtheta_target)
        )
        r_exp = ca.vertcat(r_exp, r_input_i)
        if i == 0:
            r0_exp = ca.vertcat(r0_exp, r_input_i)

    # state costs 
    for i in range(0, N + 1):
        eC = ca.sin(phid[i]) * (xp[i] - xd[i] - xdgrad[i] * (theta[i] - ttheta[i])) - \
             ca.cos(phid[i]) * (yp[i] - yd[i] - ydgrad[i] * (theta[i] - ttheta[i]))
        eL = - ca.cos(phid[i]) * (xp[i] - xd[i] - xdgrad[i] * (theta[i] - ttheta[i])) - \
             ca.sin(phid[i]) * (yp[i] - yd[i] - ydgrad[i] * (theta[i] - ttheta[i]))

        r_state_i = ca.vertcat(
            p_Q1 * eC,
            p_Q2 * eL
        )

        if i == 0:
            r0_exp = ca.vertcat(r0_exp, r_state_i)

        r_exp = ca.vertcat(r_exp, r_state_i)

    # least squares objective
    f_exp = 1.0 / 2.0 * ca.dot(r_exp, r_exp)
    f0_exp = 1.0 / 2.0 * ca.dot(r0_exp, r0_exp)

    # constraints...
    h_exp = []

    # state evolution dynamics constraints
    for i in range(N):
        h_dyn_i = states[:, i + 1] - states[:, i] - calc_step_rk4(states[:, i], inputs[:, i], model, dt)

        h_exp = ca.vertcat(h_exp, h_dyn_i)

    # track contouring constraints
    for i in range(0, N):
        if use_spline:
            h_track_i = ca.constpow(xp[i] - x_spline(theta[i]) + 0.14833, 2) + ca.constpow(yp[i] - y_spline(theta[i]) - 1.05, 2)
            print(xd[0])
            print(x_spline(0))
            print(yd[0])
            print(y_spline(0))
        else:
            h_track_i = (xp[i] - xd[i]) * (xp[i] - xd[i]) + (yp[i] - yd[i]) * (yp[i] - yd[i])

        h_exp = ca.vertcat(h_exp, h_track_i)

    # closed loop constraints
    for i in range(ns):
        h_loop_i = states[N * ns + i] - states[i]
        if i == 2:
            h_loop_i -= 2 * pi
        if i == 8:
            h_loop_i -= track_length
        h_exp = ca.vertcat(h_exp, h_loop_i)

    # start/endpoint arclength constraints
    h_theta_0 = states[8]
    h_exp = ca.vertcat(h_exp, h_theta_0)

    # constraint limit values
    h_min = np.concatenate((np.zeros(ns * N), np.zeros(N), np.zeros(ns + 1)))
    h_max = np.concatenate((np.zeros(ns * N), 0.23 * 0.23 * np.ones(N), np.zeros(ns + 1)))  # 0.10 for most

    x = ca.veccat(states, inputs)
    x_min = np.concatenate((np.tile(state_min, N + 1), np.tile(input_min, N)))
    x_max = np.concatenate((np.tile(state_max, N + 1), np.tile(input_max, N)))

    p = ca.veccat(model, track)
    npar = p.size()[0]

    nx = len(x_min)
    nh = len(h_min)

    return nx, nh, npar, x, p, f_exp, f0_exp, h_exp, x_min, x_max, h_min, h_max


lap_states = np.load("lap1_states.npy")
lap_inputs = np.load("lap1_inputs.npy")


N = 30 #len(lap_states)
dt = 1.0 / 30.0
nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, x_min, x_max, h_min, h_max = get_pacejka_problem_optimal_lap(N, dt, False)
ns = 9
nu = 3


def update_plot(traj, x):
    x_traj = [x[i * ns + 0] for i in range(N+1)]
    y_traj = [x[i * ns + 1] for i in range(N+1)]
    traj.set_data(x_traj, y_traj)
    plt.draw()
    plt.pause(0.1)


x_guess = 0 * x_min
for i in range(N):
    x_guess[i * ns: (i + 1) * ns] = lap_states[i] #- 1
    x_guess[(N + 1) * ns + i * nu: (N + 1) * ns + (i + 1) * nu] = lap_inputs[i]
x_guess[N * ns: (N + 1) * ns] = lap_states[0] #- 1
x_guess[(N + 1) * ns - 1] += track_length
x_guess[(N + 1) * ns - 1 - 6] += 2 * pi

for i in range(N+1):
    x_guess[i * ns + 8] -= track_length
    x_guess[i * ns + 2] -= 2 * pi

t = [x_guess[i * ns + 8] for i in range(N+1)]
track = track_params(t)
p = np.concatenate((model, track))

f_func = ca.Function('f', [x_sym, p_sym], [f_exp])
h_func = ca.Function('h', [x_sym, p_sym], [h_exp])
f_exp_p = f_func(x_sym, p)
h_exp_p = h_func(x_sym, p)

ax = plot_track()
opt_traj, = ax.plot([], [], 'b-*')

update_plot(opt_traj, x_guess)

ip_solver = get_ip_solver(x_sym, p_sym, f_exp, h_exp, pl=2)

for j in range(5):
    #_, _, _, ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, x_guess, x_min, x_max, h_min, h_max, pl=5)
    _, _, _, ip_x, ip_lam = ipopt_ref(ip_solver, x_guess, p_sym, x_min, x_max, h_min, h_max) #, pl=5)
    update_plot(opt_traj, ip_x)
    t = [ip_x[i * ns + 8] for i in range(N+1)]
    track = track_params(t)
    p = np.concatenate((model, track))
    f_exp_p = f_func(x_sym, p)
    h_exp_p = h_func(x_sym, p)

np.save("opt_lap", ip_x)

for k in range(63):
    cut = 1
    cutloc = np.random.randint(1, N)
    # x_guess = np.concatenate((ip_x[0:(N+1-cut)*ns], ip_x[(N+1)*ns:-(cut*nu)]))
    if k % 2 < 2:
        x_guess = np.concatenate((ip_x[0:cutloc * ns], ip_x[(cutloc + 1) * ns:(N+1) * ns], ip_x[(N + 1) * ns:(N + 1) * ns + cutloc * nu], ip_x[(N + 1) * ns + (cutloc + 1) * nu:]))
        N = N - cut
    else:
        x_guess = ip_x
    print("Trying N = ", N)
    nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, x_min, x_max, h_min, h_max = get_pacejka_problem_optimal_lap(N, dt, False)
    # track = track[0:-(cut*6)]
    if k % 2 < 2:
        track = np.concatenate((track[0:cutloc * 6], track[(cutloc+1) * 6:]))
    p = np.concatenate((model, track))
    f_func = ca.Function('f', [x_sym, p_sym], [f_exp])
    h_func = ca.Function('h', [x_sym, p_sym], [h_exp])
    f_exp_p = f_func(x_sym, p)
    h_exp_p = h_func(x_sym, p)
    update_plot(opt_traj, x_guess)

    num_solves = 1
    for j in range(num_solves):
        _, _, _, ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, x_guess, x_min, x_max, h_min, h_max, pl=5)
        update_plot(opt_traj, ip_x)
        t = [ip_x[i * ns + 8] for i in range(N+1)]
        track = track_params(t)
        p = np.concatenate((model, track))
        f_exp_p = f_func(x_sym, p)
        h_exp_p = h_func(x_sym, p)

for j in range(20):
    _, _, _, ip_x, ip_lam = ipopt_ref(x_sym, f_exp_p, h_exp_p, x_guess, x_min, x_max, h_min, h_max, pl=5)
    update_plot(opt_traj, ip_x)
    t = [ip_x[i * ns + 8] for i in range(N+1)]
    track = track_params(t)
    p = np.concatenate((model, track))
    f_exp_p = f_func(x_sym, p)
    h_exp_p = h_func(x_sym, p)

np.save("opt_lap_tightened_126_17", ip_x)