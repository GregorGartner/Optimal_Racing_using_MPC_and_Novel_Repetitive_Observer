import casadi as ca
import numpy as np
from mpcc import MPC
from car_model import CarModel
from helper_functions import plot_trajectory, animate_trajectory, get_demo_track_spline, \
    get_initial_guess, plot_track_spline, plot_track, track_pos, plot_track1, plot_all_states, animate_predictions
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from gaussian_process import GaussianProcess
from time import perf_counter


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
car2 = CarModel(lr, lf, m, I*0.8, Df*0.8, Cf*0.8, Bf*0.8, Dr*0.8, Cr*0.8, Br*0.8, Cm1*0.8, Cm2*0.8, Cd*0.8, Croll*0.8)

### --- initialize Kalman filter --- ###
num_fitting_points = 200 # step size for track discretization
estimator = KalmanFilter(car2, tracklength=tracklength, num_fitting_points=num_fitting_points, KF_type='linear', P=1, R=0.1)

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
controller = MPC(Q1, Q2, R1, R2, R3, target_speed, N, car2, estimator, dt)



### --- Initializing the Simulation --- ###
steps = 700 # number of simulatoin steps
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0]) # initial state

traj = np.zeros((steps, ns)) # array to save trajectory

# initial guess for the solver including trajectory and inputs
traj_guess = np.zeros((N+1, ns)) # calculated to warm-start the solver
traj_guess[0] = prev_state # first state of initial guess for solver
inputs_guess = np.zeros((N, nu)) # inputs to warm-start the solver
initial_guess = np.zeros(((N + 1) * ns + N * nu + (N + 1) * ns)) # concatenates traj and input guess up to MPC horizon

theta_index = 8 # index of progress parameter theta in state vector

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
        u = np.array([-0.5, 1.1, 0])
    elif i < 65:
        u = np.array([0.8, -1.5, 0.0])
    elif i < 72:
        u = np.array([-0.3, -1.2, 0.0])
    elif i < 84:
        u = np.array([0.0, 0.0, 0.0])
    elif i < 100:
        u = np.array([0.0, 2.0, 0.0])
    elif i < 100:
        u = np.array([0.0, 0.0, 0.0])



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


    
# print("--- Plotting initial guess for solver ---")
# plot_trajectory(traj_guess, 1)
# animate_trajectory(traj_guess, tail_length=7, dt=dt)
# input("Press Enter to continue...")




#######################################################################################
########################## --- MPC controlled simulation --- ##########################
#######################################################################################

# creater solver based on MPCC problem
rejection = False # turn on disturbance rejection
IP_solver, x_min, x_max, h_min, h_max = controller.get_ip_solver(N, dt, ns, rejection, nu)

# initial state car and initial guess to warm-start the solver
prev_state = np.array([0.48106088, -1.05, 0, 1.18079094, 0, 0, 0.4, 0, 0.37])
if rejection:
    initial_guess = ca.veccat(traj_guess.flatten(), inputs_guess.flatten(), np.zeros((N+1)*ns))
else:
    initial_guess = ca.veccat(traj_guess.flatten(), inputs_guess.flatten())

# matrix to save progress of the disturance estimates and state predictions
current_estimate_vector = np.zeros((steps, 3)) # saves current disturbance estimates for v_x, v_y and omega over time
#actual_disturbance_vector = np.zeros((steps, ns)) # saves the actual disturbance over time
next_pred = np.zeros((steps, ns)) # saves the next step prediction of MPC over time
KF_error_vector = np.zeros((steps, 3)) # saves one-step predicition error of Kalman filter for V_x, v_y and omega
time = np.zeros(steps) # saves the time needed to solve the optimization problem
predictions = np.zeros((steps, N+1, ns)) # saves all predictions over time
measurments = np.zeros((steps, 3)) # saves the measurements of the disturbance over time

# factor to get integer multple of periods in one lap
period_factor = 10*np.pi/tracklength

# calculating trajectory
for i in range(steps):
    
    # turn on disturbance rejection after 220 steps (roughly one lap)
    if i < 200:
        learning_phase = False
    else:
        learning_phase = True

    # solve optimization problem and measure time
    t0 = perf_counter()
    opt_states, opt_inputs, opt_d = controller.mpc_controller(IP_solver, x_min, x_max, h_min, h_max, initial_guess, prev_state, N, ns, nu, learning_phase)
    t1 = perf_counter()
    print("Finished compiling IPOPT solver in " + str(t1 - t0) + " seconds!")
    time[i] = t1 - t0 
    predictions[i] = opt_states

    # calculating next state and retrieving next applied input
    u0 = opt_inputs[0, :]
    new_state = prev_state + car.state_update_rk4(prev_state, u0, dt, True)

    # incorporating model mismatch
    #- ca.cos(prev_state[theta_index] * period_factor) * 0.01,
    # actual_disturbance_vector[i] = np.array([0.0,
    #                                          0.0,
    #                                          0.0,
    #                                          - ca.cos(prev_state[theta_index] * period_factor) * 0.2, 
    #                                          - ca.cos(prev_state[theta_index] * period_factor) * 0.1,
    #                                          - ca.cos(prev_state[theta_index] * period_factor) * 0.7,
    #                                          0.0,
    #                                          0.0,
    #                                          0.0])
    # new_state = new_state - actual_disturbance_vector[i]


    # use Kalman filter for update of disturbance function estimate
    disturbance_vector, current_estimate, KF_error, d_mes = estimator.estimate(prev_state, u0, new_state, dt)
    KF_error_vector[i] = KF_error
    measurments[i] = d_mes

    # save current disturbance estimates
    current_estimate_vector[i] = current_estimate

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
    if opt_d is not None:
        initial_guess[(N + 1) * ns + N* nu:] = opt_d.flatten()





######################################################################
########################## --- Plotting --- ##########################
######################################################################

#avg = np.average(time)
#print("Averga solving time was: " + str(avg))


fig, axs = plt.subplots(2, 4, figsize=(12, 7))
titles = ["x position", "y position", "yaw angle", "x velocity", "y velocity", "yaw rate", "torque", "steer"]
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].plot(traj[:, i], color='red', label="actual")
    axs[row, col].plot(next_pred[:, i], color='blue', label="predicted")
    axs[row, col].set_title(titles[i])
    axs[row, col].legend()
plt.tight_layout()
plt.show()


print("Average time: " + str(np.average(time)) + " seconds.")

# plotting and animating the trajectory   
plot_trajectory(traj, 1)
animate_trajectory(traj, tail_length=5, dt=dt)
animate_predictions(traj, predictions, dt=dt)


# plotting the disturbance estimation of the KF at the current thetas over time
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
axs[0].plot(current_estimate_vector[:, 0], color='red')
axs[0].set_title("v_x disturbance estimate over time")
axs[1].plot(current_estimate_vector[:, 1], color='red')
axs[1].set_title("v_y disturbance estimate over time")
axs[2].plot(current_estimate_vector[:, 2], color='red')
axs[2].set_title("omega disturbance estimate over time")
plt.tight_layout()
plt.show()

# plotting the disturbance estimation of the KF at the current thetas over time
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
axs[0].plot(KF_error_vector[:, 0], color='blue')
axs[0].set_title("v_x KF_error")
axs[1].plot(KF_error_vector[:, 1], color='blue')
axs[1].set_title("v_y KF_error")
axs[2].plot(KF_error_vector[:, 2], color='blue')
axs[2].set_title("omega KF_erro")
plt.tight_layout()
plt.show()


lap1 = (traj[:, 8] // tracklength) == 0
lap2 = (traj[:, 8] // tracklength) == 1
lap3 = (traj[:, 8] // tracklength) == 2
lap4 = (traj[:, 8] // tracklength) == 3




args = np.arange(0, tracklength, tracklength/10000)
f_values = estimator.function(args)
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
titles = ["v_x disturbance function", "v_y disturbance function", "omega disturbance function"]
for i in range(3):
    axs[i].plot(args, f_values[:, i], color='blue')
    axs[i].scatter((traj[:, 8] % tracklength)[lap1], measurments[:, i][lap1], color='red', s=6, label="lap1")
    axs[i].scatter((traj[:, 8] % tracklength)[lap2], measurments[:, i][lap2], color='magenta', s=6, label="lap2")
    axs[i].scatter((traj[:, 8] % tracklength)[lap3], measurments[:, i][lap3], color='green', s=6, label="lap3")
    axs[i].scatter((traj[:, 8] % tracklength)[lap4], measurments[:, i][lap4], color='cyan', s=6, label="lap4")
    axs[i].set_title(titles[i])
plt.tight_layout()
fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Laps")
plt.show()





# # plot difference between trajectory and prediction
# plt.figure(figsize=(10, 7))
# plt.plot(traj[:, 5] - next_pred[:, 5], color='blue')
# plt.title("Difference between trajectory and prediction")
# plt.legend()
# plt.show()



