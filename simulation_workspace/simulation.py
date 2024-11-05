import casadi as ca
import numpy as np
from mpcc import MPC
from car_model import CarModel
from helper_functions import plot_trajectory, animate_trajectory, get_demo_track_spline, \
    get_initial_guess, plot_track_spline, plot_track, track_pos, plot_track1
from kalman_filters import KalmanFilterLinear, KalmanFilterPolynomial, KalmanFilterSpline, KalmanFilterLinearSpline
import matplotlib.pyplot as plt


#plot_track1()
#plt.show()


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
N = 15  # horizon length of MPC
ns = 9  # number of states
nu = 3  # number of inputs
controller = MPC(Q1, Q2, R1, R2, R3, target_speed, N, car, dt)


### --- Initializing the Simulation --- ###
steps = 620 # number of simulatoin steps
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0]) # initial state

traj = np.zeros((steps, 9)) # array to save trajectory
traj_guess = np.zeros((N+1, 9)) # calculated to warm-start the solver
traj_guess[0] = prev_state # first state of initial guess for solver
traj_spline = np.zeros((N+1, 2)) # unused array containing the closest points on center line over the trajectory
traj_spline[0] = [0.48106088, -1.05]
inputs_guess = np.zeros((N, 3)) # inputs to warm-start the solver
initial_guess = np.zeros(((N + 1) * ns + N * nu)) # concatenates traj and input guess up to MPC horizon

# get spline of track
x_spline, y_spline, dx_spline, dy_spline, tracklength = get_demo_track_spline()

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

# creater solver based on MPCC problem
IP_solver, x_min, x_max, h_min, h_max = controller.get_ip_solver(
    N, dt, ns, nu)
# input("Press Enter to continue...")

# initial state car and initial guess to warm-start the solver
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0.37])
initial_guess = ca.veccat(traj_guess.flatten(), inputs_guess.flatten())

# matrix to save progress of the disturance estimates
disturbance_matrix = np.zeros((3, 47))

# factor to get integer multple of periods in one lap
period_factor = 4*np.pi/tracklength

# initialize Kalman filter
estimator = KalmanFilterPolynomial(car, tracklength=tracklength, P=1, R=0.1)

# calculating trajectory
for i in range(steps):
    parameters = prev_state
    opt_states, opt_inputs = controller.mpc_controller(IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu)

    # calculating next state and retrieving next applied input
    u0 = opt_inputs[0, :]
    new_state = prev_state + car.state_update_rk4(prev_state, u0, dt, True)

    # incorporating model mismatch
    new_state[3] = new_state[3] - ca.cos(prev_state[8] * period_factor) * 0.02

    # use Kalman filter for update of disturbance function estimate
    disturbance_vector = estimator.estimate(prev_state, u0, new_state, dt)

    # save current estimates every 199 steps (roughly one lap)
    if i % 199 and i > 0:
        disturbance_matrix[i//199 - 1] = disturbance_vector

    # updating the "new" previous state and the actual trajectory
    prev_state = new_state
    traj[i] = new_state

    print("Step: ", i)

    # updating initial guess
    uN = opt_inputs[-1, :]
    initial_guess[: N*ns] = opt_states[1:, :].flatten()
    initial_guess[N*ns: (N+1)*ns] = opt_states[-1, :] + car.state_update_rk4(opt_states[-1, :], uN, dt, True)
    initial_guess[(N+1)*ns: (N+1)*ns+(N-1) * nu] = opt_inputs[1:, :].flatten()
    initial_guess[(N + 1) * ns + (N-1)* nu:] = uN
    
    # plot open-loop prediction
    # plot_trajectory(opt_states, 1)
    # input("Press Enter to continue...")
#######################################################################################
#######################################################################################
#######################################################################################

# plotting the disturbance estimation over different laps
plt.plot(disturbance_matrix[0], color='blue')
plt.plot(disturbance_matrix[1], color='red')
plt.plot(disturbance_matrix[2], color='green')
plt.show()

# plotting true mismatch vs estimated mismatch
thetas = np.linspace(0, tracklength, 1000)
thetas = thetas[:-1]
disturbance_function = - ca.cos(thetas * period_factor) * 0.02
estimated_function = estimator.function(thetas)
plt.plot(thetas, disturbance_function, color='blue')
plt.plot(thetas, estimated_function, color='red')
plt.show()

# plotting and animating the trajectory   
plot_trajectory(traj, 1)
animate_trajectory(traj, tail_length=5, dt=dt)
