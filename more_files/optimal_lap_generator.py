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

# track spline (currently not used)
def get_demo_track_spline():
    track = yaml.load(open("DEMO_TRACK.yaml", "r"))["track"]
    trackLength = track["trackLength"]
    indices = [i for i in range(1500, 5430, 10)]
    x = np.array(track["xCoords"])[indices]
    y = np.array(track["yCoords"])[indices]
    dx = np.array(track["xRate"])[indices]
    dy = np.array(track["yRate"])[indices]
    t = np.array(track["arcLength"])[indices] - 0.5 * trackLength

    t_sym = ca.MX.sym('t')

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

# implementation of dynamic bicycle model with Pacejka tire model
def calc_dx(x, u, model, numerical=False):
    # size params
    p_lr = model[6]
    p_lf = model[7]
    p_m = model[8]
    p_I = model[9]
    # lateral force params
    p_Df = model[10]
    p_Cf = model[11]
    p_Bf = model[12]
    p_Dr = model[13]
    p_Cr = model[14]
    p_Br = model[15]
    # longitudinal force params
    p_Cm1 = model[16]
    p_Cm2 = model[17]
    p_Cd = model[18]
    p_Croll = model[19]

    yaw = x[2]
    vx = x[3]
    vy = x[4]
    omega = x[5]
    T = x[6]
    delta = x[7]

    dT = u[0]
    ddelta = u[1]
    dtheta = u[2]

    Fx = (p_Cm1 - p_Cm2 * vx) * T - p_Cd * vx * vx - p_Croll
    beta = ca.arctan(vy / vx)
    ar = -beta + p_lr * omega / vx
    af = delta - beta - p_lf * omega / vx
    Fr = p_Dr * ca.sin(p_Cr * ca.arctan(p_Br * ar))
    Ff = p_Df * ca.sin(p_Cf * ca.arctan(p_Bf * af))

    if numerical:
        return np.array([
            vx * np.cos(yaw) - vy * np.sin(yaw),
            vx * np.sin(yaw) + vy * np.cos(yaw),
            omega,
            1 / p_m * (Fx - Ff * np.sin(delta) + p_m * vy * omega),
            1 / p_m * (Fr + Ff * np.cos(delta) - p_m * vx * omega),
            1 / p_I * (Ff * p_lf * np.cos(delta) - Fr * p_lr),
            0,
            0,
            0
        ])
    else:
        return ca.vertcat(
            vx * ca.cos(yaw) - vy * ca.sin(yaw),
            vx * ca.sin(yaw) + vy * ca.cos(yaw),
            omega,
            1 / p_m * (Fx - Ff * ca.sin(delta) + p_m * vy * omega),
            1 / p_m * (Fr + Ff * ca.cos(delta) - p_m * vx * omega),
            1 / p_I * (Ff * p_lf * ca.cos(delta) - Fr * p_lr),
            0,
            0,
            0
        )

# integrator calculating a weighted average of different approximation 
# in a Runge-Kutta 4th order method
def calc_step_rk4(x, u, model, dt, numerical=False):
    k1 = calc_dx(x, u, model, numerical)
    k2 = calc_dx(x + dt / 2.0 * k1, u, model, numerical)
    k3 = calc_dx(x + dt / 2.0 * k2, u, model, numerical)
    k4 = calc_dx(x + dt * k3, u, model, numerical)

    out = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    out[6] = dt * u[0]
    out[7] = dt * u[1]
    out[8] = dt * u[2]
    return out


def get_pacejka_problem_optimal_lap(N, dt, use_spline):

    # state and input dimensions
    ns = 9
    nu = 3

    # defining state and input constraints
    state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
    state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
    input_min = [-2.0, -15.0, 0.0]
    input_max = [2.0, 15.0, 5.0]

    # defining the state as symbolic variable and its components for N+1 steps
    states = ca.MX.sym('s', ns, N + 1)
    xp = states[0, :]
    yp = states[1, :]
    yaw = states[2, :]
    vx = states[3, :]
    vy = states[4, :]
    omega = states[5, :] # yaw rate
    T = states[6, :]
    delta = states[7, :] # steering angle (?)
    theta = states[8, :] # progress along track

    # defining the input as symbolic variable and its components for N steps
    inputs = ca.MX.sym('u', nu, N)
    dT = inputs[0, :]
    ddelta = inputs[1, :]
    dtheta = inputs[2, :] 

    # defining track params
    track = ca.MX.sym('t', 6, N+1)
    xd = track[0, :]
    yd = track[1, :]
    xdgrad = track[2, :]
    ydgrad = track[3, :]
    phid = track[4, :] # track angle
    ttheta = track[5, :]

    # dynamics parameters of the model are defined in calc_dx
    model = ca.MX.sym('p', 20)
    # cost params
    p_Q1 = model[0]
    p_Q2 = model[1]
    p_R1 = model[2]
    p_R2 = model[3]
    p_R3 = model[4]
    p_dtheta_target = model[5]

    # initializing cost terms
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

    # state costs (contouring and lag error)
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



    # initializing constraint term
    h_exp = []

    # state evolution dynamics constraints
    for i in range(N):
        h_dyn_i = states[:, i + 1] - states[:, i] - calc_step_rk4(states[:, i], inputs[:, i], model, dt)

        h_exp = ca.vertcat(h_exp, h_dyn_i)

    # track contouring constraints
    for i in range(0, N):
        # if use_spline:
        #     h_track_i = ca.constpow(xp[i] - x_spline(theta[i]) + 0.14833, 2) + ca.constpow(yp[i] - y_spline(theta[i]) - 1.05, 2)
        #     print(xd[0])
        #     print(x_spline(0))
        #     print(yd[0])
        #     print(y_spline(0))
        # else:

        # adding track constraints to constraint vector
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

    # start/endpoint arclength constraints (theta at t=0 is 0 ???)
    h_theta_0 = states[8]
    h_exp = ca.vertcat(h_exp, h_theta_0)

    # constraint limit values
    h_min = np.concatenate((np.zeros(ns * N), np.zeros(N), np.zeros(ns + 1))) # need to subtract 9 values if closed loop constraint is omitted
    h_max = np.concatenate((np.zeros(ns * N), 0.23 * 0.23 * np.ones(N), np.zeros(ns + 1)))  # 0.10 for most

    x = ca.veccat(states, inputs)
    x_min = np.concatenate((np.tile(state_min, N + 1), np.tile(input_min, N)))
    x_max = np.concatenate((np.tile(state_max, N + 1), np.tile(input_max, N)))

    p = ca.veccat(model, track)
    npar = p.size()[0]

    nx = len(x_min)
    nh = len(h_min)

    return nx, nh, npar, x, p, f_exp, f0_exp, h_exp, x_min, x_max, h_min, h_max


def update_plot(traj, x):
    x_traj = [x[i * ns + 0] for i in range(N+1)]
    y_traj = [x[i * ns + 1] for i in range(N+1)]
    traj.set_data(x_traj, y_traj)
    plt.draw()
    plt.pause(0.1)







if __name__ == "__main__":
    # load lap states and inputs 
    lap_states = np.load("lap2_states.npy")
    lap_inputs = np.load("lap2_inputs.npy")

    # MPC horizon length
    #N = len(lap_states)
    N = 40 
    dt = 1.0 / 30.0
    nx, nh, npar, x_sym, p_sym, f_exp, f0_exp, h_exp, x_min, x_max, h_min, h_max = get_pacejka_problem_optimal_lap(N, dt, False)
    ns = 9
    nu = 3

    x_guess = 0 * x_min
    # arranging the importet lap data in the initial guess
    for i in range(N):
        x_guess[i * ns: (i + 1) * ns] = lap_states[i]
        x_guess[(N + 1) * ns + i * nu: (N + 1) * ns + (i + 1) * nu] = lap_inputs[i]
    x_guess[N * ns: (N + 1) * ns] = lap_states[0]
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