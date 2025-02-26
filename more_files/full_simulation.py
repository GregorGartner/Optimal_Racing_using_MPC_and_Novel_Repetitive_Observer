import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
from time import perf_counter
from track_object import Track

# plot trajectory
def plot_trajectory(traj, final_traj):
    ax = plot_track()
    
    x = traj[:, 0]
    y = traj[:, 1]

    # PLotting the trajectory
    plt.scatter(x, y)
    if final_traj == 0:
        plt.pause(5)
    else:
        plt.show()

# animate trajectory as video
def animate_trajectory(traj, tail_length=2, dt=0.1):
    """
    Animate a car trajectory with a dark blue current point and a light blue tail.

    Parameters:
    - x: Array of x coordinates.
    - y: Array of y coordinates.
    - tail_length: Number of points in the tail.
    - dt: Time between frames (in seconds).
    """

    x = traj[:, 0]
    y = traj[:, 1]
    
    # Calculate interval in milliseconds
    interval = 1000 * dt

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
    
    # Call plot_track with the axis object to plot the track on the same axis
    plot_track(ax)

    # Line that will represent the trajectory (initialize empty)
    line, = ax.plot([], [], lw=5, color="red")

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot at each frame
    def update(frame):
        current_idx = frame
        start_idx = max(0, current_idx - tail_length)

        # Plot the current point as dark blue and the previous points as light blue
        xdata = x[start_idx:current_idx+1]
        ydata = y[start_idx:current_idx+1]

        # Tail color gradient: last points as lighter
        line.set_data(xdata, ydata)
        # line.set_color("blue")
        line.set_color("red")
        line.set_linewidth(5)

        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(
        x), init_func=init, interval=interval, blit=True)

    # Show the animation
    plt.show()

# track spline
def get_demo_track_spline():
    track = yaml.load(open("DEMO_TRACK.yaml", "r"),
                      Loader=yaml.SafeLoader)["track"]
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

# get initial guess to "warm start" the solver
def get_initial_guess(N, ns, nu):
    lap_states = np.load("lap1_states.npy")
    lap_inputs = np.load("lap1_inputs.npy")
    track_length = 0.3 * (26 + 4*np.pi)

    x_guess = np.zeros((N + 1) * ns + N * nu)
    # arranging the importet lap data in the initial guess
    for i in range(N):
        x_guess[i * ns: (i + 1) * ns] = lap_states[i]
        x_guess[(N + 1) * ns + i * nu: (N + 1) *
                ns + (i + 1) * nu] = lap_inputs[i]
    x_guess[N * ns: (N + 1) * ns] = lap_states[0]
    x_guess[(N + 1) * ns - 1] += track_length
    x_guess[(N + 1) * ns - 1 - 6] += 2 * np.pi

    for i in range(N+1):
        x_guess[i * ns + 8] -= track_length
        x_guess[i * ns + 2] -= 2 * np.pi
        x_guess[i * ns + 1] -= 1
    return x_guess

# creates plot of track spline w\o plt.show()
def plot_track_spline(ax = None):
    x_spline, y_spline, dy_spline, dx_spline = get_demo_track_spline()

    # Generate parameter values for plotting
    theta = np.linspace(0, 12, 100)  # 100 points from 0 to 4

    # Evaluate the spline functions
    x_vals = x_spline(theta)
    y_vals = y_spline(theta)

    if ax is None:
        # Step 3: Plot the splines
        plt.figure()
        plt.plot(x_vals, y_vals, label='Spline Curve', color='blue')
        plt.title('Spline Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.grid()
        plt.legend()
    
    else:
        # Plot the track spline on the provided axes
        ax.plot(x_vals, y_vals, label='Track', color='blue')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid()
        ax.legend()


def track_pos(t):
    trackobj = Track()
    a = t

    t = 0.3
    trackobj.add_line(3.5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, -np.pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, -np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(4.5 * t)

    x, y = trackobj.point_at_arclength(a)
    x += 0.15
    y -= 1.05
    return x, y

# plotting track with center line and boundaries
def plot_track(ax = None):
    track_length = 0.3 * (26 + 4*np.pi)
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
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(xtrack, ytrack, 'k--')
    ax.plot(xtrack + half_track_width * yrate, ytrack - half_track_width * xrate, 'k')
    ax.plot(xtrack - half_track_width * yrate, ytrack + half_track_width * xrate, 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax




class CarModel():
    def __init__(self, lr, lf, m, I, Df, Cf, Bf, Dr, Cr, Br, Cm1, Cm2, Cd, Croll):
        # initializing car parameters
        self.lr = lr
        self.lf = lf
        self.m = m
        self.I = I
        self.Df = Df
        self.Cf = Cf
        self.Bf = Bf
        self.Dr = Dr
        self.Cr = Cr
        self.Br = Br
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cd = Cd
        self.Croll = Croll

    # dynamics function of the car
    def dynamics(self, x, u, numerical=False):

        # extract state information
        yaw = x[2]
        v_x = x[3]
        v_y = x[4]
        omega = x[5]
        acc = x[6]
        delta = x[7]

        Fx = (self.Cm1 - self.Cm2 * v_x) * acc - \
            self.Cd * v_x * v_x - self.Croll
        
        # Logans Implementation
        # beta = ca.arctan(v_y / v_x)
        # alpha_f = delta - beta - self.lf * omega / v_x
        # alpha_r = -beta + self.lr * omega / v_x
        # Ff = self.Df * ca.sin(self.Cf * ca.arctan(self.Bf * alpha_f))
        # Fr = self.Dr * ca.sin(self.Cr * ca.arctan(self.Br * alpha_r))

        # # Following the paper
        alpha_f = - ca.arctan((omega * self.lf + v_y) / v_x) + delta
        alpha_r = ca.arctan((omega * self.lr - v_y) / v_x)
        Fr = self.Dr * ca.sin(self.Cr * ca.arctan(self.Br * alpha_r))
        Ff = self.Df * ca.sin(self.Cf * ca.arctan(self.Bf * alpha_f))

        # returning the differential equations governing the car dynamics
        if numerical:
            return np.array([
                v_x * np.cos(yaw) - v_y * np.sin(yaw),
                v_x * np.sin(yaw) + v_y * np.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * np.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * np.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * np.cos(delta) - Fr * self.lr),
                u[0],
                u[1],
                u[2]
            ])
        else:
            return ca.vertcat(
                v_x * ca.cos(yaw) - v_y * ca.sin(yaw),
                v_x * ca.sin(yaw) + v_y * ca.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * ca.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * ca.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * ca.cos(delta) - Fr * self.lr),
                u[0],
                u[1],
                u[2]
            )

    # integrator
    def state_update_rk4(self, x, u, dt, numerical=False):
        # Runge-Kutta 4th order integration
        k1 = self.dynamics(x, u, numerical)
        k2 = self.dynamics(x + dt / 2.0 * k1, u, numerical)
        k3 = self.dynamics(x + dt / 2.0 * k2, u, numerical)
        k4 = self.dynamics(x + dt * k3, u, numerical)

        # calculates weighted average of 4 different approximations
        out = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # x[6] = dt * u[0]
        # x[7] = dt * u[1]
        # x[8] = dt * u[2]
        return out


class MPC():
    def __init__(self, Q1, Q2, R1, R2, R3, target_speed, N, car, dt):
        self.Q1 = Q1
        self.Q2 = Q2
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.target_speed = target_speed
        self.N = N
        self.car = car
        self.dt = dt

    def get_ip_solver(self, N, dt, ns, nu):

        # state and input constraint values
        state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
        state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
        input_min = [-2.0, -15.0, 0.0]
        input_max = [2.0, 15.0, 3.5]


        # symbolic state variables
        states = ca.MX.sym('s', 9, N + 1)
        x_p = states[0, :]
        y_p = states[1, :]
        yaw = states[2, :]
        v_x = states[3, :]
        v_y = states[4, :]
        omega = states[5, :]
        acc = states[6, :]
        delta = states[7, :]
        theta = states[8, :]

        # symbolic input variables
        inputs = ca.MX.sym('u', 3, N)
        dacc = inputs[0, :]
        ddelta = inputs[1, :]
        dtheta = inputs[2, :]

        # track params
        # track = ca.MX.sym('t', 6, N+1)
        # x_ref = track[0, :]
        # y_ref = track[1, :]
        # x_ref_grad = track[2, :]
        # y_ref_grad = track[3, :]
        # phi_ref = track[4, :]
        # theta_ref = track[5, :]

        # initial state in params
        x0 = ca.MX.sym('x0', ns)

        # get track spline for track constraints
        x_spline, y_spline, dy_spline, dx_spline = get_demo_track_spline()

        # defining objective
        objective = 0
        for i in range(N):
            # adding quadratic input cost to objective
            objective += (self.R1 * dacc[i])**2 \
                + (self.R2 * ddelta[i])**2 \
                + (self.R3 * (dtheta[i] - self.target_speed))**2

        for i in range(N+1):
            # defining lag and contouring error
            eC = dy_spline(theta[i]) * (x_p[i] - x_spline(theta[i])) - \
                dx_spline(theta[i]) * (y_p[i] - y_spline(theta[i]))
            eL = -dx_spline(theta[i]) * (x_p[i] - x_spline(theta[i])) - \
                dy_spline(theta[i]) * (y_p[i] - y_spline(theta[i]))
            # eC = ca.sin(phi_ref[i]) * (x_p[i] - x_ref[i] - x_ref_grad[i] * (theta[i] - theta_ref[i])) - \
            #      ca.cos(phi_ref[i]) * (y_p[i] - y_ref[i] - y_ref_grad[i] * (theta[i] - theta_ref[i]))
            # eL = - ca.cos(phi_ref[i]) * (x_p[i] - x_ref[i] - x_ref_grad[i] * (theta[i] - theta_ref[i])) - \
            #      ca.sin(phi_ref[i]) * (y_p[i] - y_ref[i] - y_ref_grad[i] * (theta[i] - theta_ref[i]))

            # adding quadratic stage cost to objective
            objective += (self.Q1 * eC)**2 \
                + (self.Q2 * eL)**2 \


            objective = 0.5 * objective

        # defining constraints
        constraints = []

        # initial state constraint
        constraints = ca.vertcat(constraints, states[:, 0] - x0)

        # constraint the states to follow car dynamics
        for i in range(N):
            constraints = ca.vertcat(
                constraints, states[:, i + 1] - states[:, i] - self.car.state_update_rk4(states[:, i], inputs[:, i], dt))

        # constraint the states to be within the track limits
        for i in range(1, N+1):
            constraints = ca.vertcat(constraints, ca.constpow(x_p[i] - x_spline(theta[i]), 2) + ca.constpow(y_p[i] - y_spline(theta[i]), 2))
            # constraints = ca.vertcat(constraints, (x_p[i] - x_ref[i]) * (x_p[i] - x_ref[i]) + (y_p[i] - y_ref[i]) * (y_p[i] - y_ref[i]))
            pass

        # initial state ns, dynamics N*ns, track limits N
        h_min = np.concatenate((np.zeros(ns), np.zeros(N*ns), np.zeros(N)))
        h_max = np.concatenate((np.zeros(ns), np.zeros(N*ns), 0.23 * 0.23 * np.ones(N))) # 0.23 * 0.23

        x_min = np.concatenate(
            (np.tile(state_min, N + 1), np.tile(input_min, N)))
        x_max = np.concatenate(
            (np.tile(state_max, N + 1), np.tile(input_max, N)))

        x = ca.veccat(states, inputs)
        parameters = ca.veccat(x0)

        print("Compiling IPOPT solver...")
        t0 = perf_counter()
        IP_nlp = {'x': x, 'f': objective, 'p': parameters, 'g': constraints}
        IP_solver = ca.nlpsol('S', 'ipopt', IP_nlp, {'ipopt': {'linear_solver': 'mumps' , 'max_iter': 100}}) #'linear_solver': 'ma57'
                              
        t1 = perf_counter()
        print("Finished compiling IPOPT solver in " + str(t1 - t0) + " seconds!")

        return IP_solver, x_min, x_max, h_min, h_max


    # solve MPC optimization problem
    def mpc_controller(self, IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu):
        sol = IP_solver(x0=initial_guess, p=parameters,
                        lbx=x_min, ubx=x_max, lbg=h_min, ubg=h_max)
        print("Optimization has ended!")
        # input("Press Enter to continue...")

        # Extract the solution and separate states and inputs
        sol_x = sol['x'].full().flatten()
        opt_states = sol_x[0:ns*(N+1)].reshape((N+1, ns))
        opt_inputs = sol_x[ns*(N+1):].reshape((N, nu))

        if IP_solver.stats()['return_status'] != 'Solve_Succeeded':
            print("---------------------------------" + IP_solver.stats()['return_status'] + "---------------------------------")
            #plot_trajectory(opt_states, 1)

        return opt_states, opt_inputs


if __name__ == "__main__":

    ### --- Initializing the car model --- ###
    # size params
    lr = 0.038  # distance from center of mass to rear axle
    lf = 0.052  # distance from center of mass to front axle
    m = 0.181  # mass of the car
    I = 0.000505  # moment of inertia
    # lateral force params
    Df = 0.65
    Cf = 1.5
    Bf = 5.2
    Dr = 1.0
    Cr = 1.45
    Br = 8.5
    # longitudinal force params
    Cm1 = 0.98028992
    Cm2 = 0.01814131
    Cd = 0.02750696
    Croll = 0.08518052

    car = CarModel(lr, lf, m, I, Df, Cf, Bf, Dr, Cr, Br, Cm1, Cm2, Cd, Croll)

    ### --- Initializing the MPC --- ###
    # weights for the cost function
    Q1 = 1.5 # contouring cost
    Q2 = 12.0  # lag cost
    R1 = 1.0  # cost for change of acceleration
    R2 = 1.0  # cost for change of steering angle
    R3 = 1.3  # cost for deviation of progress speed dtheta to target progress speed
    target_speed = 3.6  # target speed of progress theta

    # problem parameters
    dt = 1/30  # step size
    N = 80  # horizon length of MPC
    ns = 9  # number of states
    nu = 3  # number of inputs
    controller = MPC(Q1, Q2, R1, R2, R3, target_speed, N, car, dt)


    ### --- Initializing the Simulation --- ###
    steps = 250
    prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0])
    traj_guess = np.zeros((N+1, 9))
    traj_spline = np.zeros((N+1, 2))
    traj = np.zeros((steps, 9))
    traj_guess[0] = prev_state
    traj_spline[0] = [0.48106088, -1.05]
    inputs_guess = np.zeros((N, 3))
    initial_guess = np.zeros(((N + 1) * ns + N * nu)) # takes traj and input guess up to MPC horizon
    # initial_guess = get_initial_guess(N, ns, nu)
    # prev_state = initial_guess[0:ns]

    x_spline, y_spline, dy_spline, dx_spline = get_demo_track_spline()

    # manual (hard coded) control
    for i in range(N):
        if i < 15-N:
            u = np.array([0.8, 0, 0])
        elif i < 37-N:
            u = np.array([0.0, 0.0, 0])
        elif i < 43-N:
            u = np.array([0.0, 1.0, 0])
        elif i < 51-N:
            u = np.array([-0.05, 0.0, 0])
        elif i < 56-N: 
            u = np.array([0.0, -1.0, 0])
        elif i < 64-N:
            u = np.array([0.0, 0, 0])
        elif i < 80-N:
            u = np.array([0.0, 1.1, 0])
        else:
            u = np.array([0.0, 0.0, 0.0])

        new_state = prev_state + car.state_update_rk4(prev_state, u, dt, True)


        # Define the optimization variable theta
        theta = ca.MX.sym('theta')
        # Define the point on the spline for a given theta
        spline_point = [x_spline(theta), y_spline(theta)]
        # Define the distance function
        distance = ca.sqrt((spline_point[0] - new_state[0])**2 + (spline_point[1] - new_state[1])**2)
        # Create an NLP solver to minimize the distance
        nlp = {'x': theta, 'f': distance}
        solver = ca.nlpsol('solver', 'ipopt', nlp)
        # Solve the NLP to find the optimal theta
        sol = solver(x0=0.5, lbx=0, ubx=8)  # Initial guess and bounds for theta
        # Get the optimal theta
        theta_opt = sol['x'].full().flatten()[0]
        # update progress parameter theta according to current position
        new_state[8] = theta_opt

        traj_spline[i+1] = np.array([x_spline(theta_opt), y_spline(theta_opt)]).flatten()

        traj_guess[i+1] = new_state
        inputs_guess[i] = u

        prev_state = new_state


        
    #print("--- Plotting initial guess for solver ---")
    #plot_trajectory(traj_guess, 1)
    #plot_trajectory(traj_spline, 1)
    #animate_trajectory(traj_guess, tail_length=7, dt=dt)
    #input("Press Enter to continue...")







    #######################################################################################
    ########################## --- MPC controlled simulation --- ##########################
    #######################################################################################
    # t = [initial_guess[i * ns + 8] for i in range(N+1)]
    # track = track_params(t)

    IP_solver, x_min, x_max, h_min, h_max = controller.get_ip_solver(
        N, dt, ns, nu)
    input("Press Enter to continue...")

    prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0.37])
    # prev_state = np.array([0, -1.05, 0, 0.1, 0, 0, 0, 0, 0])
    #initial_guess[0:(N+1)*ns] = traj_guess[:(N+1), :].flatten()
    #initial_guess[(N+1)*ns:] = inputs_guess[:N, :].flatten()
    initial_guess = ca.veccat(traj_guess.flatten(), inputs_guess.flatten())

    
    # # calculating trajectory
    for i in range(steps):
        # parameters = np.concatenate((track, prev_state))
        parameters = prev_state
        #initial_guess[0:(N+1)*ns] = traj_guess[i:(N+1+i), :].flatten()
        #initial_guess[(N+1)*ns:] = inputs_guess[i:(N+i), :].flatten()
        opt_states, opt_inputs = controller.mpc_controller(IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu)

        # calculating next state
        u0 = opt_inputs[0, :]
        new_state = prev_state + car.state_update_rk4(prev_state, u0, dt, True)
        prev_state = new_state
        traj[i] = new_state

        print("Step: ", i)
        #input("Press Enter to continue...")

        # updating initial guess
        uN = opt_inputs[-1, :]
        initial_guess[: N*ns] = opt_states[1:, :].flatten()
        initial_guess[N*ns: (N+1)*ns] = opt_states[-1, :] + car.state_update_rk4(opt_states[-1, :], uN, dt, True)
        initial_guess[(N+1)*ns: (N+1)*ns+(N-1) * nu] = opt_inputs[1:, :].flatten()
        initial_guess[(N + 1) * ns + (N-1)* nu:] = uN
        

        # plot open-loop prediction
        #plot_trajectory(opt_states, 1)
        # input("Press Enter to continue...")
    #######################################################################################
    #######################################################################################
    #######################################################################################


            
    plot_trajectory(traj, 1)
    animate_trajectory(traj, tail_length=5, dt=dt)
