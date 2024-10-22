import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
from time import perf_counter



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


        Fx = (self.Cm1 - self.Cm2 * v_x) * acc - self.Cd * v_x * v_x - self.Croll
        # Logans Implementation
        beta = ca.arctan(v_y / v_x)
        alpha_f = delta - beta - self.lf * omega / v_x
        alpha_r = -beta + self.lr * omega / v_x
        Ff = self.Df * ca.sin(self.Cf * ca.arctan(self.Bf * alpha_f))
        Fr = self.Dr * ca.sin(self.Cr * ca.arctan(self.Br * alpha_r))
        # # Following the paper
        # alpha_f = ca.arctan((v_y + omega * lf) / v_x) + delta
        # alpha_r = ca.arctan((v_y - omega * lr) / v_x)
        # Fr = Dr * ca.sin(Cr * ca.arctan(Br * alpha_r))
        # Ff = Df * ca.sin(Cf * ca.arctan(Bf * alpha_f))

        # returning the differential equations governing the car dynamics
        if numerical:
            return np.array([
                v_x * np.cos(yaw) - v_y * np.sin(yaw),
                v_x * np.sin(yaw) + v_y * np.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * np.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * np.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * np.cos(delta) - Fr * self.lr),
                0,
                0,
                0
            ])
        else:
            return ca.vertcat(
                v_x * ca.cos(yaw) - v_y * ca.sin(yaw),
                v_x * ca.sin(yaw) + v_y * ca.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * ca.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * ca.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * ca.cos(delta) - Fr * self.lr),
                0,
                0,
                0
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
        out[6] = dt * u[0]
        out[7] = dt * u[1]
        out[8] = dt * u[2]
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

    def mpc_controller(self, x_0, N, dt):
        ns = 9
        nu = 3

        state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
        state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
        input_min = [-2.0, -15.0, 0.0]
        input_max = [2.0, 15.0, 5.0]

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
        track = ca.MX.sym('t', 6, N+1)
        x_ref = track[0, :]
        y_ref = track[1, :]
        x_ref_grad = track[2, :]
        y_ref_grad = track[3, :]
        phi_ref = track[4, :]
        theta_ref = track[5, :]

        # initial state in params
        #x0 = ca.MX.sym('x0', ns)

        x_spline, y_spline, _, _ = get_demo_track_spline()

        # defining objective
        objective = 0
        for i in range(N):
            # adding quadratic input cost to objective
            objective +=  (self.R1 * dacc[i])**2 \
                        + (self.R2 * ddelta[i])**2 \
                        + (self.R3 * (dtheta[i] - self.target_speed))**2
            
        for i in range(N+1):
            # defining lag and contouring error
            eC = ca.sin(phi_ref[i]) * (x_p[i] - x_ref[i] - x_ref_grad[i] * (theta[i] - theta_ref[i])) - \
                 ca.cos(phi_ref[i]) * (y_p[i] - y_ref[i] - y_ref_grad[i] * (theta[i] - theta_ref[i]))
            eL = - ca.cos(phi_ref[i]) * (x_p[i] - x_ref[i] - x_ref_grad[i] * (theta[i] - theta_ref[i])) - \
                 ca.sin(phi_ref[i]) * (y_p[i] - y_ref[i] - y_ref_grad[i] * (theta[i] - theta_ref[i]))
            
            # adding quadratic stage cost to objective
            objective += (self.Q1 * eC)**2 \
                        + (self.Q2 * eL)**2 \
                        
                        
            objective = 0.5 * objective - theta[N]
                        
        
        # defining constraints
        constraints = []
        # initial state constraint
        constraints = ca.vertcat(constraints, states[:, 0] - x_0)
        #terminal state constraint
        #constraints.append(states[:, N] - self.x_N)

        for i in range(N):
                # constraint the states to follow car dynamics
                constraints = ca.vertcat(constraints, states[:, i + 1] - states[:, i] - self.car.state_update_rk4(states[:, i], inputs[:, i], dt))

        for i in range(1,N+1):
                # constraint the states to be within the track limits
                constraints = ca.vertcat(constraints, ca.constpow(x_p[i] - x_spline(theta[i]), 2) + ca.constpow(y_p[i] - y_spline(theta[i]), 2))
                # constraints = ca.vertcat(constraints, (x_p[i] - x_ref[i]) * (x_p[i] - x_ref[i]) + (y_p[i] - y_ref[i]) * (y_p[i] - y_ref[i]))
                pass

        # initial state ns, dynamics N*ns, track limits N
        h_min = np.concatenate((np.zeros(ns), np.zeros(N*ns), np.zeros(N)))
        h_max = np.concatenate((np.zeros(ns), np.zeros(N*ns), 1 * 1 * np.ones(N))) # 0.23 * 0.23

        
        x_min = np.concatenate((np.tile(state_min, N + 1), np.tile(input_min, N)))
        x_max = np.concatenate((np.tile(state_max, N + 1), np.tile(input_max, N)))

        x = ca.veccat(states, inputs)
        parameters = ca.veccat(track)


        print("Compiling IPOPT solver...")
        t0 = perf_counter()
        IP_nlp = {'x': x, 'f': objective, 'p': parameters, 'g': constraints}
        IP_solver = ca.nlpsol('S', 'ipopt', IP_nlp, {'ipopt': {'linear_solver': 'mumps'}}) #, 'max_iter': 100}}) #'linear_solver': 'ma57'
        t1 = perf_counter()
        print("Finished compiling IPOPT solver in " + str(t1 - t0) + " seconds!")
        #input("Press Enter to continue...")

        sol = IP_solver(lbx=x_min, ubx=x_max, lbg=h_min, ubg=h_max)

        # Extract the solution and separate states and inputs
        sol_x = sol['x'].full().flatten()
        opt_states = sol_x[0:ns*(N+1)].reshape((ns, N+1))
        opt_inputs = sol_x[ns*(N+1):].reshape((nu, N))

        return opt_states, opt_inputs




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

# animate trajectory as video
def animate_trajectory(x, y, tail_length=3, dt=0.1):
    """
    Animate a car trajectory with a dark blue current point and a light blue tail.

    Parameters:
    - x: Array of x coordinates.
    - y: Array of y coordinates.
    - tail_length: Number of points in the tail.
    - dt: Time between frames (in seconds).
    """
    # Calculate interval in milliseconds
    interval = 1000 * dt

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(y) - 0.1, max(y) + 0.1)

    # Line that will represent the trajectory (initialize empty)
    line, = ax.plot([], [], lw=5, color="blue")

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
        #line.set_color("blue")
        line.set_color("pink")
        line.set_linewidth(5)

        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, interval=interval, blit=True)

    # Show the animation
    plt.show()

# get initial guess to "warm start" the solver
def get_initial_guess(N, ns, nu):
    lap_states = np.load("lap1_states.npy")
    lap_inputs = np.load("lap1_inputs.npy")
    track_length = 0.3 * (26 + 4*np.pi)

    x_guess = np.zeros((N + 1) * ns + N * nu)
    # arranging the importet lap data in the initial guess
    for i in range(N):
        x_guess[i * ns: (i + 1) * ns] = lap_states[i]
        x_guess[(N + 1) * ns + i * nu: (N + 1) * ns + (i + 1) * nu] = lap_inputs[i]
    x_guess[N * ns: (N + 1) * ns] = lap_states[0]
    x_guess[(N + 1) * ns - 1] += track_length
    x_guess[(N + 1) * ns - 1 - 6] += 2 * np.pi

    for i in range(N+1):
        x_guess[i * ns + 8] -= track_length
        x_guess[i * ns + 2] -= 2 * np.pi
        x_guess[i * ns + 1] -= 1
    return x_guess





### --- Initializing the car model --- ###
if True:
    # size params
    lr = 0.038
    lf = 0.052
    m = 0.181
    I = 0.000505
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
if True:
    # weights for the cost function
    Q1 = 1.5  # contouring cost
    Q2 = 15.0  # lag cost
    R1 = 1.0  # cost for change of acceleration
    R2 = 1.0  # cost for change of steering angle 
    R3 = 1.0  # cost devation in progress (theta) ???
    target_speed = 3.6  # target speed
    # problem parameters
    x_0 = np.array([0, -1, 0, 0.001, 0.001, 0, 0, 0, 0])
    dt = 1/30
    horizon = 30
controller = MPC(Q1, Q2, R1, R2, R3, target_speed, horizon, car, dt)




### --- Initializing the Simulation --- ###
steps = 1
prev_state = x_0

# trajectory array
traj = np.zeros((steps, 9))

# calculating trajectory 
for i in range(steps):
    opt_states, opt_inputs = controller.mpc_controller(prev_state, horizon, dt)


    inputs = inputs.reshape(30,3)
    for j in range(inputs.shape[0]):
        new_state = prev_state + car.state_update_rk4(prev_state, inputs[j], dt, True)
        traj[i] = new_state
        prev_state = new_state
        print("--- Time Step: " + str(i) + " ---")



# for i in range(steps):
#     if i < 20:
#         u = np.array([1, 3, 0])
#     else:
#         u = np.array([0, 0, 0])

#     new_state = prev_state + car.state_update_rk4(prev_state, u, dt, True)
#     traj[i] = new_state
#     prev_state = new_state




# extracting x and y coordinates
x = traj[:, 0]
y = traj[:, 1]



# PLotting the trajectory
plt.scatter(x, y)
plt.show()
animate_trajectory(x, y, tail_length=15, dt=dt)


