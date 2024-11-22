import casadi as ca
import numpy as np
from mpcc import MPC
from car_model import CarModel
from helper_functions import plot_trajectory, animate_trajectory, get_demo_track_spline, \
    get_initial_guess, plot_track_spline, plot_track, track_pos, plot_track1, plot_all_states
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from gaussian_process import GaussianProcess


# get spline of track
x_spline, y_spline, dx_spline, dy_spline, tracklength = get_demo_track_spline()


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
Cm1 = 0.98028992 # 0.977767
Cm2 = 0.01814131 # 0.017807
Cd = 0.02750696 # 0.0612451
Croll = 0.08518052 # 0.1409

car = CarModel(lr, lf, m, I, Df, Cf, Bf, Dr, Cr, Br, Cm1, Cm2, Cd, Croll)

### --- initialize Kalman filter --- ###
discretization = 0.25 # step size for track discretization
estimator = KalmanFilter(car, tracklength=tracklength, discretization=discretization, KF_type='bspline', P=1, R=0.1)
gaussian_process = GaussianProcess(tracklength, car)

### --- Initializing the MPC --- ###
# weights for the cost function
Q1 = 10.0 # contouring cost 1.5 /// 10.0
Q2 = 1000.0  # lag cost 12.0 /// 1000.0
R1 = 0.1  # cost for change of acceleration 1.0 /// 0.1
R2 = 0.3  # cost for change of steering angle 1.0 /// 0.3
R3 = 30  # cost for deviation of progress speed dtheta to target progress speed 1.3 /// 0.2
target_speed = 3.5  # target speed of progress theta 3.6 /// 3.5
# problem parameters
dt = 1/30  # step size
N = 20  # horizon length of MPC
ns = 9  # number of states
nu = 3  # number of inputs
controller = MPC(Q1, Q2, R1, R2, R3, target_speed, N, car, estimator, dt)



### --- Initializing the Simulation --- ###
steps = 1200 # number of simulatoin steps
n_discretization_points = int(ca.ceil(tracklength / discretization))
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0]) # initial state

traj = np.zeros((steps, ns)) # array to save trajectory

# initial guess for the solver including trajectory and inputs
traj_guess = np.zeros((N+1, ns)) # calculated to warm-start the solver
traj_guess[0] = prev_state # first state of initial guess for solver
inputs_guess = np.zeros((N, nu)) # inputs to warm-start the solver
initial_guess = np.zeros(((N + 1) * ns + N * nu + (N + 1) * ns)) # concatenates traj and input guess up to MPC horizon

# define indices for disturbance
disturbed_state = 3
theta_index = 8

# manual (hard coded) control
for i in range(N):
    if i < 0:
        u = np.array([0.8, 0, 0])
    elif i < 7:
        u = np.array([0.0, 0.0, 0])
    elif i < 13:
        u = np.array([0.0, 1.0, 0])
    elif i < 21:
        u = np.array([-0.05, 0.0, 0])
    elif i < 26: 
        u = np.array([0.0, -1.0, 0])
    elif i < 34:
        u = np.array([0.0, 0, 0])
    elif i < 50:
        u = np.array([0.0, 1.1, 0])
    elif i < 60:
        u = np.array([-0.5, -1.5, 0.0])
    elif i < 70:
        u = np.array([0.0, 0, 0])
    elif i < 100:
        u = np.array([0.1, 1.5, 0.0])



    new_state = prev_state + car.state_update_rk4(prev_state, u, dt, True)


    # find closest point on center line to add progress to initial guess
    theta = ca.MX.sym('theta') # optimization variable
    spline_point = [x_spline(theta), y_spline(theta)]
    distance = ca.sqrt((spline_point[0] - new_state[0])**2 + (spline_point[1] - new_state[1])**2)
    nlp = {'x': theta, 'f': distance}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=0.5, lbx=0, ubx=8)
    theta_opt = sol['x'].full().flatten()[0]

    # Get the optimal theta and update progress parameter theta according to current position
    new_state[theta_index] = theta_opt

    traj_guess[i+1] = new_state
    inputs_guess[i] = u

    prev_state = new_state


    
print("--- Plotting initial guess for solver ---")
plot_trajectory(traj_guess, 1)
#animate_trajectory(traj_guess, tail_length=7, dt=dt)
input("Press Enter to continue...")




#######################################################################################
########################## --- MPC controlled simulation --- ##########################
#######################################################################################

# creater solver based on MPCC problem
IP_solver, x_min, x_max, h_min, h_max = controller.get_ip_solver(N, dt, ns, nu)

# initial state car and initial guess to warm-start the solver
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0.37])
initial_guess = ca.veccat(traj_guess.flatten(), inputs_guess.flatten(), np.zeros((N+1)*ns))

# matrix to save progress of the disturance estimates and state predictions
current_estimate_vector = np.zeros(steps) # saves current disturbance estimates over time
actual_disturbance_vector = np.zeros(steps) # saves the actual disturbance over time
next_pred = np.zeros((steps, ns)) # saves the next step prediction of MPC over time
KF_error_vector = np.zeros(steps) # saves one-step predicition error measured by the Kalman filter

# factor to get integer multple of periods in one lap
period_factor = 10*np.pi/tracklength

# calculating trajectory
for i in range(steps):
    if i < 220:
        parameters = ca.veccat(prev_state, np.zeros(len(estimator.disturbance_vector)))
    else:
        parameters = ca.veccat(prev_state, estimator.disturbance_vector)

    opt_states, opt_inputs, opt_d = controller.mpc_controller(IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu)
    # opt_states, opt_inputs = controller.mpc_controller(IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu)

    # calculating next state and retrieving next applied input
    u0 = opt_inputs[0, :]
    new_state = prev_state + car.state_update_rk4(prev_state, u0, dt, True)

    # incorporating model mismatch
    new_state[5] = new_state[5] - ca.cos(prev_state[theta_index] * period_factor) * 0.7

    # use Kalman filter for update of disturbance function estimate
    disturbance_vector, current_estimate, KF_error = estimator.estimate(prev_state, u0, new_state, dt)
    KF_error_vector[i] = KF_error

    # save current disturbance estimates and actual disturbance
    current_estimate_vector[i] = current_estimate
    actual_disturbance_vector[i] = - ca.cos(prev_state[theta_index] * period_factor) * 0.7

    # save next step prediction and actual state at next step
    next_pred[i] = opt_states[1, :].flatten()

    # updating the "new" previous state and the actual trajectory
    prev_state = new_state
    traj[i] = new_state

    print("Step: ", i)

    # updating initial guess
    uN = opt_inputs[-1, :]
    initial_guess[: N*ns] = opt_states[1:, :].flatten()
    initial_guess[N*ns: (N+1)*ns] = opt_states[-1, :] + car.state_update_rk4(opt_states[-1, :], uN, dt, True)
    initial_guess[(N+1)*ns: (N+1)*ns+(N-1) * nu] = opt_inputs[1:, :].flatten()
    initial_guess[(N + 1) * ns + (N-1)* nu: (N + 1) * ns + N* nu] = uN
    initial_guess[(N + 1) * ns + N* nu:] = opt_d.flatten()





######################################################################
########################## --- Plotting --- ##########################
######################################################################

# plotting and animating the trajectory   
plot_trajectory(traj, 1)
animate_trajectory(traj, tail_length=5, dt=dt)

# # plotting the disturbance estimation over different laps
plt.plot(actual_disturbance_vector, color='blue')
plt.plot(current_estimate_vector, color='red')
plt.title("Disturbance estimation over time")
plt.show()


# plot difference between trajectory and prediction
plt.figure(figsize=(10, 7))
plt.plot(traj[:, 5] - next_pred[:, 5], color='blue')
plt.title("Difference between trajectory and prediction")
plt.legend()
plt.show()


# plot one-step prediction error in Kalman filter
plt.figure(figsize=(10, 7))
plt.plot(KF_error_vector, color='green')
plt.title("One-Step Prediction Error in Kalman Filter")
plt.legend()
plt.show()


# plotting true mismatch vs estimated mismatch
thetas = np.linspace(0, tracklength, 1000)
thetas = thetas[:-1]
disturbance_function = - ca.cos(thetas * period_factor) * 0.7
estimated_function = estimator.function(thetas)
plt.plot(thetas, disturbance_function, color='blue')
plt.plot(thetas, estimated_function, color='red')
plt.title("Final disturbance function estimate")
plt.show()

plot_trajectory(traj, 1)

