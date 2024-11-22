import numpy as np
from helper_functions import plot_trajectory, plot_track, track_pos
import matplotlib.pyplot as plt
import casadi as ca


# Define the mean and standard deviation
mean = 0
std_dev = 0.2

# Generate x values
x = np.linspace(-1.5, 1.5, 1000)

# Calculate the Gaussian (normal) distribution
gaussian = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
periodic = 0.01 * np.exp(-0.5 * (np.sin(x*np.pi)/0.4)**2)

# Plot the Gaussian
plt.figure(figsize=(4.5, 2.5))
#plt.plot(x, gaussian)
plt.plot(x, periodic)
plt.title('Periodic Kernel')
plt.yticks([])
plt.xticks([-1, 0, 1])
plt.ylim(-0.0001, 0.015)
#plt.grid(True)
plt.show()


# Define the interpolation points
v = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

# Generate x values limited to -3 to 3
x_limited = np.linspace(-1.2, 1.2, 1000)

# Generate y values for linear interpolation
y = np.interp(x_limited, np.linspace(-1, 1, len(v)), v)

# Plot the linear interpolation
plt.figure(figsize=(4.5, 2.5))
plt.plot(x_limited, y)
plt.title('Linear Interpolation')
plt.yticks([])
plt.ylim(-0.02, 1.5)  # Set y-axis limits to 0 to 2
#plt.grid(True)
plt.show()




# Define control points for the B-spline
control_points1 = np.zeros(11)
#control_points1[5] = 1.0
control_points1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
grid_points = np.array([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
                        

# Define the B-spline interpolants
bspline1 = ca.interpolant('bspline1', 'bspline', [grid_points], control_points1)

# Generate points along the spline to evaluate and plot
t_values = np.linspace(-1.25, 1.25, 200)  # Parameter values
spline_points1 = np.array([bspline1(t).full().flatten() for t in t_values])
# Plot the B-spline
plt.figure(figsize=(4.5, 2.5))
plt.plot(t_values, spline_points1)
plt.yticks([])
plt.ylim(-0.2, 1.5)
plt.title('B-Spline Interpolation')
#plt.ylim(-0.02, 2)  # Set y-axis limits to 0 to 2
#plt.grid(True)
plt.show()


