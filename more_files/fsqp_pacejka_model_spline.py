import numpy as np
import casadi as ca
from math import sin, cos, pi, atan2
import yaml

# dynamics model
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


def calc_step_ee(x, u, model, dt, numerical=False):
    return dt * calc_dx(x, u, model, numerical)


# integrator
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


# track spline
def get_demo_track_spline():
    track = yaml.load(open("DEMO_TRACK.yaml", "r"), Loader=yaml.SafeLoader)["track"]
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


def get_pacejka_problem(N, dt):
    get_demo_track_spline()

    # state and input dimensions
    ns = 9
    nu = 3

    # state and input constraints
    state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
    state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
    input_min = [-2.0, -15.0, 0.0]
    input_max = [2.0, 15.0, 3.5]

    # symbolic variables
    states = ca.MX.sym('s', ns, N + 1)
    xp = states[0, :]
    yp = states[1, :]
    yaw = states[2, :]
    vx = states[3, :]
    vy = states[4, :]
    omega = states[5, :]
    T = states[6, :]
    delta = states[7, :]
    theta = states[8, :]

    inputs = ca.MX.sym('u', nu, N)
    dT = inputs[0, :]
    ddelta = inputs[1, :]
    dtheta = inputs[2, :]

    # track params
    track = ca.MX.sym('t', 6, N+1)
    xd = track[0, :]
    yd = track[1, :]
    xdgrad = track[2, :]
    ydgrad = track[3, :]
    phid = track[4, :]
    ttheta = track[5, :]

    # initial state in params
    x0 = ca.MX.sym('x0', ns)
    # terminal state in params
    xN = ca.MX.sym('xN', ns)
    # terminal state flag
    xN_flag = ca.MX.sym('xN_flag', 1)

    # dynamics params...
    model = ca.MX.sym('p', 20)
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

    x_spline, y_spline, dx_spline, dy_spline = get_demo_track_spline()

    # input costs
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

        # eC = dy_spline(theta[i]) * (xp[i] - x_spline(theta[i])) - \
        #      dx_spline(theta[i]) * (yp[i] - y_spline(theta[i]))
        # eL = -dx_spline(theta[i]) * (xp[i] - x_spline(theta[i])) - \
        #     dy_spline(theta[i]) * (yp[i] - y_spline(theta[i]))

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
    for i in range(1, N + 1):
        h_track_i = ca.constpow(xp[i] - x_spline(theta[i]), 2) + ca.constpow(yp[i] - y_spline(theta[i]), 2)
        # h_track_i = (xp[i] - xd[i]) * (xp[i] - xd[i]) + (yp[i] - yd[i]) * (yp[i] - yd[i])

        h_exp = ca.vertcat(h_exp, h_track_i)

    # initial state constraints
    for i in range(ns):
        h_x0_i = states[i] - x0[i]
        h_exp = ca.vertcat(h_exp, h_x0_i)

    # terminal state constraints
    for i in range(ns):
        h_xN_i = xN_flag * (states[i, N] - xN[i])
        h_exp = ca.vertcat(h_exp, h_xN_i)

    # constraint limit values
    h_min = np.concatenate((np.zeros(ns * N), np.zeros(N), np.zeros(2 * ns)))
    h_max = np.concatenate((np.zeros(ns * N), 0.23 * 0.23 * np.ones(N), np.zeros(2 * ns)))

    x = ca.veccat(states, inputs)
    x_min = np.concatenate((np.tile(state_min, N + 1), np.tile(input_min, N)))
    x_max = np.concatenate((np.tile(state_max, N + 1), np.tile(input_max, N)))

    p = ca.veccat(model, x0, xN, track, xN_flag)
    # p = ca.veccat(model, x0, track)
    npar = p.size()[0]

    nx = len(x_min)
    nh = len(h_min)

    return nx, nh, npar, x, p, f_exp, f0_exp, h_exp, r_exp, x_min, x_max, h_min, h_max
