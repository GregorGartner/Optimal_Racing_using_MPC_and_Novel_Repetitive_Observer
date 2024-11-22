import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define control points for the B-spline
control_points1 = np.zeros(11)
control_points2 = np.zeros(11)
control_points3 = np.zeros(11)
control_points1[3] = 1.0
control_points2[7] = 1.0
control_points3[3] = 2.0
control_points3[7] = 5.0
grid_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                        

# Define the B-spline interpolants
bspline1 = ca.interpolant('bspline1', 'bspline', [grid_points], control_points1)
bspline2 = ca.interpolant('bspline2', 'bspline', [grid_points], control_points2)
bspline3 = ca.interpolant('bspline3', 'bspline', [grid_points], control_points3)

# Generate points along the spline to evaluate and plot
t_values = np.linspace(0, 10, 200)  # Parameter values
spline_points1 = np.array([bspline1(t).full().flatten() for t in t_values])
spline_points2 = np.array([bspline2(t).full().flatten() for t in t_values])
spline_points3 = np.array([bspline3(t).full().flatten() for t in t_values])




# Plot the B-spline and its components
fig, axs = plt.subplots(2, 2, figsize=(9, 6))

# Define common limits for all subplots
y_min = min(np.min(5 * spline_points2), np.min(2 * spline_points1), np.min(2 * spline_points1 + 5 * spline_points2), np.min(spline_points3))
y_max = max(np.max(5 * spline_points2), np.max(2 * spline_points1), np.max(2 * spline_points1 + 5 * spline_points2), np.max(spline_points3))

# First subplot
axs[0, 0].plot(t_values, spline_points1, '-', label='Spline1 with d[3] = 1.0')
axs[0, 0].set_title('Spline1 with d[3] = 1.0')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_ylim([y_min, y_max])

# Second subplot
axs[0, 1].plot(t_values, spline_points2, '-', label='Spline2 with d[7] = 1.0')
axs[0, 1].set_title('Spline2 with d[7] = 1.0')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].legend()
axs[0, 1].grid()
axs[0, 1].set_ylim([y_min, y_max])

# Third subplot
axs[1, 0].plot(t_values, 2 * spline_points1 + 5 * spline_points2, '-', label='2 * Spline1 + 5 * Spline2')
axs[1, 0].set_title('2 * Spline1 + 5 * Spline2')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_ylim([y_min, y_max])

# Fourth subplot
axs[1, 1].plot(t_values, spline_points3, '-', label='Spline3 with d[3] = 2.0 and d[7] = 5.0')
axs[1, 1].set_title('B-spline Curve 3')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].legend()
axs[1, 1].grid()
axs[1, 1].set_ylim([y_min, y_max])

plt.suptitle('B-spline Interpolants in CasADi')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
