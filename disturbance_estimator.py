import numpy as np
import matplotlib.pyplot as plt
import random


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


# Defining the disturbance function
thetas = np.linspace(1, 25, 1000)
disturbance = 1.5 * np.sin(thetas/1.5) + 1/3 * \
    thetas - 1/20 * (thetas - 10) ** 2

# # Plotting the disturbance function
# plt.plot(thetas, disturbance)
# plt.show()
# input("Press Enter to continue...")

# Initializing disturbance vector
first_column = np.arange(0, 25.0, 0.5)
second_column = np.random.rand(50)
d_est_vec = np.column_stack((first_column, second_column))
# d_est_vec = np.array([[1.0, 2.1], [2.1, 1.0], [3.1, 1.1], [4.1, 4.1], [5.1, 0.1], [7.1, 3.1], [8.0, 4.0], [10.0, 2.0], [11.0, 2.0], [12.0, 2.0], [
#                     13.3, 1.3], [14.1, 5.1], [15.2, 5.2], [17.2, 3.2], [18.2, 3.2], [20.2, 5.2], [21.2, 4.2], [22.3, 5.3], [23.3, 1.3], [24.3, 3.3]])



# Plotting the disturbance function
plt.plot(thetas, disturbance)
plt.scatter(d_est_vec[:, 0], d_est_vec[:, 1])
plt.show()




for i in range(100000):
    # generate random function argument (theta)
    random_index = random.randint(0, 999)
    random_arg = (random_index / 999) * 25

    # get thetas of interpolation points
    int_points = d_est_vec[:, 0]

    # get next higher and lower interpolation points
    lower_idx, higher_idx = find_interpolation_points(int_points, random_arg)
    ip1 = d_est_vec[lower_idx, :]
    ip2 = d_est_vec[higher_idx, :]

    # estimate disturbance from linear interpolation
    d_est = ip2[1] * (random_arg - ip1[0]) / (ip2[0] - ip1[0]) + \
        ip1[1] * (ip2[0] - random_arg) / (ip2[0] - ip1[0])

    # "measured" disturbance
    d_mes = disturbance[random_index]

    # update disturbance estimate
    d_est_new = d_est + 0.2 * (d_mes - d_est)

    # Find the closer interpolation point
    if abs(random_arg - ip1[0]) < abs(random_arg - ip2[0]):
        closer_idx = lower_idx
    else:
        closer_idx = higher_idx

    # Update the closer interpolation point
    d_est_vec[closer_idx, :] = [random_arg, d_est_new]


# Plotting the disturbance function
plt.plot(thetas, disturbance)
plt.scatter(d_est_vec[:, 0], d_est_vec[:, 1])
plt.show()
