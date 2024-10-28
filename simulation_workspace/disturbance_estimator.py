import numpy as np
import matplotlib.pyplot as plt
import random
from math import floor


def find_interpolation_points(thetas, theta):
    idx = np.searchsorted(thetas, theta)

    if idx == 0:
        lower_idx = 0
        higher_idx = 1
    elif idx == len(thetas):
        lower_idx = len(thetas) - 2
        higher_idx = len(thetas) - 1
    else:
        lower_idx = idx - 1
        higher_idx = idx

    lower_theta = thetas[lower_idx]
    higher_theta = thetas[higher_idx]

    return lower_idx, higher_idx


# Defining the disturbance_function function
thetas = np.linspace(0, 24, 1000)
thetas = thetas[:-1]
disturbance_function = 1.5 * np.sin(thetas/1.5) + 1/3 * \
    thetas - 1/20 * (thetas - 10) ** 2

# # Plotting the disturbance_function function
# plt.plot(thetas, disturbance_function)
# plt.show()
# input("Press Enter to continue...")

# Initializing disturbance_function vector
first_column = np.arange(0, 25.0, 0.5)
second_column = np.random.rand(50)
d_est_vec = np.column_stack((first_column, second_column))
# d_est_vec = np.array([[1.0, 2.1], [2.1, 1.0], [3.1, 1.1], [4.1, 4.1], [5.1, 0.1], [7.1, 3.1], [8.0, 4.0], [10.0, 2.0], [11.0, 2.0], [12.0, 2.0], [
#                     13.3, 1.3], [14.1, 5.1], [15.2, 5.2], [17.2, 3.2], [18.2, 3.2], [20.2, 5.2], [21.2, 4.2], [22.3, 5.3], [23.3, 1.3], [24.3, 3.3]])



# Plotting the disturbance_function function
plt.plot(thetas, disturbance_function)
plt.scatter(d_est_vec[:, 0], d_est_vec[:, 1])
plt.show()



def luenberger_observer(d_est_vec, disturbance_function):
    for i in range(1000):
        # generate random function argument (theta)
        random_index = random.randint(0, 999)
        random_arg = (random_index / 999) * 25

        # get thetas of interpolation points
        int_points = d_est_vec[:, 0]

        # get next higher and lower interpolation points
        lower_idx, higher_idx = find_interpolation_points(int_points, random_arg)
        ip1 = d_est_vec[lower_idx, :]
        ip2 = d_est_vec[higher_idx, :]

        # estimate disturbance_function from linear interpolation
        d_est = ip2[1] * (random_arg - ip1[0]) / (ip2[0] - ip1[0]) + \
            ip1[1] * (ip2[0] - random_arg) / (ip2[0] - ip1[0])

        # "measured" disturbance_function
        noise = np.random.normal(0, 1)
        d_mes = disturbance_function[random_index] + noise

        # update disturbance_function estimate
        d_est_new = d_est + 0.2 * (d_mes - d_est)

        # Find the closer interpolation point
        if abs(random_arg - ip1[0]) < abs(random_arg - ip2[0]):
            closer_idx = lower_idx
        else:
            closer_idx = higher_idx

        # Update the closer interpolation point
        d_est_vec[closer_idx, :] = [random_arg, d_est_new]

    return d_est_vec


#d_est_vec = luenberger_observer(d_est_vec, disturbance_function, thetas)

# Plotting the disturbance_function function
# plt.plot(thetas, disturbance_function)
# plt.scatter(d_est_vec[:, 0], d_est_vec[:, 1])
# plt.show()



# Kalman filter
kalman_dist_est = np.random.rand(25)



def kalman_filter_lin(kalman_dist_est, disturbance_function):
    P = 1 # initial state variance
    R = 0.3 # measurement noise variance

    for i in range(75):
        # generate random function argument (theta)
        random_index = random.randint(0, 999)
        random_arg = (random_index / 1000) * 24

        # "measured" disturbance_function
        noise = np.random.normal(0, 0.2)
        d_mes = disturbance_function[random_index] + noise

        # measurement matrix with shape of state vector (inner product is scalar)
        H_k = np.zeros(kalman_dist_est.shape)

        idx1 = floor(random_arg)
        idx2 = idx1 + 1
        H_k[idx1] = (idx2 - random_arg) / (idx2 - idx1)
        H_k[idx2] = (random_arg - idx1) / (idx2 - idx1)

        K = P * H_k.T / (H_k @ (P * H_k) + R)

        kalman_dist_est = kalman_dist_est + K * (d_mes - H_k @ kalman_dist_est)

        x_values = np.arange(len(kalman_dist_est))

    return kalman_dist_est, x_values


# kalman_dist_est, x_values = kalman_filter_lin(kalman_dist_est, disturbance_function)

progress = np.zeros((3, len(thetas)))

def kalman_filter_polyn(kalman_dist_est, disturbance_function, progress):
    P = 1 # initial state variance
    R = 0.3 # measurement noise variance

    for i in range(1000):
        # generate random function argument (theta)
        random_index = random.randint(0, 998)
        random_arg = (random_index / 1000) * 24

        # "measured" disturbance_function
        noise = np.random.normal(0, 0.3)
        d_mes = disturbance_function[random_index] + noise

        # measurement matrix with shape of state vector (inner product is scalar)
        H_k = np.zeros(kalman_dist_est.shape)

        idx1 = floor(random_arg)
        if idx1 == 0:
            idx1 = 1
        elif idx1 == 23:
            idx1 = 22
        idx0 = idx1 - 1
        idx2 = idx1 + 1
        idx3 = idx1 + 2

        H_k[idx0] = (random_arg - idx1) * (random_arg - idx2) * (random_arg - idx3) / \
            ((idx0 - idx1) * (idx0 - idx2) * (idx0 - idx3))
        H_k[idx1] = (random_arg - idx0) * (random_arg - idx2) * (random_arg - idx3) / \
            ((idx1 - idx0) * (idx1 - idx2) * (idx1 - idx3))
        H_k[idx2] = (random_arg - idx0) * (random_arg - idx1) * (random_arg - idx3) / \
            ((idx2 - idx0) * (idx2 - idx1) * (idx2 - idx3))
        H_k[idx3] = (random_arg - idx0) * (random_arg - idx1) * (random_arg - idx2) / \
            ((idx3 - idx0) * (idx3 - idx1) * (idx3 - idx2))

        K = P * H_k.T / (H_k @ (P * H_k) + R)

        kalman_dist_est = kalman_dist_est + K * (d_mes - H_k @ kalman_dist_est)

        x_values = np.arange(len(kalman_dist_est))

        if (i+1)%333 == 0:
            for index, theta in enumerate(thetas):
                H_k = np.zeros(kalman_dist_est.shape)

                idx1 = floor(theta)
                if idx1 == 0:
                    idx1 = 1
                elif idx1 == 23:
                    idx1 = 22
                idx0 = idx1 - 1
                idx2 = idx1 + 1
                idx3 = idx1 + 2

                H_k[idx0] = (theta - idx1) * (theta - idx2) * (theta - idx3) / \
                    ((idx0 - idx1) * (idx0 - idx2) * (idx0 - idx3))
                H_k[idx1] = (theta - idx0) * (theta - idx2) * (theta - idx3) / \
                    ((idx1 - idx0) * (idx1 - idx2) * (idx1 - idx3))
                H_k[idx2] = (theta - idx0) * (theta - idx1) * (theta - idx3) / \
                    ((idx2 - idx0) * (idx2 - idx1) * (idx2 - idx3))
                H_k[idx3] = (theta - idx0) * (theta - idx1) * (theta - idx2) / \
                    ((idx3 - idx0) * (idx3 - idx1) * (idx3 - idx2))
                
                disturbance_value_theta =  H_k @ kalman_dist_est

                progress[((i+1)//333) - 1, index] = disturbance_value_theta

    return kalman_dist_est, x_values, progress




    
kalman_dist_est, x_values, progress = kalman_filter_polyn(kalman_dist_est, disturbance_function, progress)

plt.plot(thetas, disturbance_function, linewidth=3, color='blue', label='Disturbance Function')
#plt.scatter(x_values, kalman_dist_est)
plt.plot(thetas, progress[0], linewidth=2, color='red', label='Progress 0')
plt.plot(thetas, progress[1], linewidth=2, color='green', label='Progress 1')
plt.plot(thetas, progress[2], linewidth=2, color='orange', label='Progress 2')
plt.show()