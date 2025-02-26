import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from mpcc import MPC
from car_model import CarModel
from helper_functions import plot_trajectory, animate_trajectory, get_demo_track_spline, \
    get_initial_guess, plot_track_spline, plot_track, track_pos, plot_track1, plot_all_states, animate_predictions
from kalman_filter import KalmanFilter
from time import perf_counter
import pandas as pd
import os
import shutil

_ ,_ ,_ ,_ , tracklength = get_demo_track_spline()

# Define the parameters used in the folder name
steps = 3000
N = 20
num_inducing_points = 200
KF_type = "bspline"
R = 0.5
R1, R2, R3 = 2.0, 10.0, 30.2

# Reconstruct the folder path dynamically
folder_name = f"Simulation_Steps{steps}_Horizon{N}_InducingPoints{num_inducing_points}_Kernel{KF_type}_R{R}_R1R2R3_{R1}_{R2}_{R3}"
simulation_folder = os.path.join('simulation_workspace', 'simulations', folder_name)

# Check if the folder exists
if not os.path.exists(simulation_folder):
    raise FileNotFoundError(f"The folder '{simulation_folder}' does not exist.")

# Load trajectory data
trajectory_file_path = os.path.join(simulation_folder, 'trajectory.csv')
traj = pd.read_csv(trajectory_file_path).to_numpy()


inputs_file_path = os.path.join(simulation_folder, 'inputs.csv')
inputs = pd.read_csv(inputs_file_path).to_numpy()

#plot_trajectory(traj)


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
car2 = CarModel(lr, lf, m, I*0.9, Df*0.9, Cf*0.9, Bf*0.9, Dr*0.9, Cr*0.9, Br*0.9, Cm1*0.9, Cm2*0.9, Cd*0.9, Croll*0.9)
### --- initialize Kalman filter --- ###
dt = 1/30
num_inducing_points = 200 # step size for track discretization
R = [100.0, 10.0, 10.0] # measurement noise in KF
P = 10 # variance of initial estimate
KF_type = 'exponential'
estimator = KalmanFilter(car2, tracklength=tracklength, num_inducing_points=num_inducing_points, KF_type=KF_type, P=P, R=R)
disturbance_measurements = np.zeros((traj.shape[0], 3))
KF_error_evolution = np.zeros((traj.shape[0], 3))

assert traj.shape[0] == inputs.shape[0]

current_lap = 0
lap_index = 0
args = np.arange(0, tracklength, tracklength/1000)

for i in range(traj.shape[0]-1):
    
    prev_state = traj[i, :]
    new_state = traj[i+1, :]
    u0 = inputs[i, :]
    disturbance_vector, current_estimate, KF_error, d_mes = estimator.estimate(prev_state, u0, new_state, dt)
    disturbance_measurements[i, :] = d_mes
    KF_error_evolution[i, :] = KF_error

    if current_lap != np.floor(traj[i, 8] // tracklength):
        
        f_values = estimator.function(args)
        fig, axs = plt.subplots(3, 1, figsize=(12, 7))
        titles = ["v_x disturbance function", "v_y disturbance function", "omega disturbance function"]
        for subplot in range(3):
            axs[subplot].plot(args, f_values[:, subplot], color='blue')
            axs[subplot].scatter((traj[:lap_index, 8] % tracklength), disturbance_measurements[:lap_index, subplot], color='red', s=10)
            axs[subplot].scatter((traj[lap_index:i, 8] % tracklength), disturbance_measurements[lap_index:i, subplot], color='green', s=10)
            axs[subplot].set_title(titles[subplot])
        plt.tight_layout()
        #plt.pause(5)
        plt.show()
        
        lap_index = i
        current_lap = np.floor(traj[i, 8] // tracklength)


fig, axs = plt.subplots(3, 1, figsize=(12, 7))
#titles = ["v_x disturbance function", "v_y disturbance function", "omega disturbance function"]
for subplot in range(3):
    axs[subplot].plot(KF_error_evolution[:, subplot])
    #axs[subplot].set_title(titles[subplot])
plt.tight_layout()
#plt.pause(5)
plt.show()
