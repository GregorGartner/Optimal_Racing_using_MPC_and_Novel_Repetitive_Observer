import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

trackLength = 11.568244517641709
discretization = 0.25

def get_coeff_spline(trackLength, discretization):
    wrap_length = trackLength / discretization
    n_indices = int(ca.floor(trackLength / discretization))

    # Example control points and knot vector
    control_points = np.zeros(n_indices+1)
    control_points[int(ca.floor(n_indices/2))] = 1.0
    grid_points = np.linspace(-n_indices/2, n_indices/2, n_indices+1)

    # Create the B-spline interpolant
    x_spline = ca.interpolant('LUT', 'bspline', [grid_points], control_points, {'degree': [3]})

    # Define the symbolic variable for the interpolation parameter
    t_sym = ca.MX.sym('t')
    # Use the modulo operation to wrap around the parameter values
    x_exp = x_spline(ca.fmod(t_sym + int(ca.floor(n_indices/2)), wrap_length) - int(ca.floor(n_indices/2)))
    # Create a CasADi function to evaluate the periodic B-spline
    x_fun = ca.Function('f', [t_sym], [x_exp])

    return x_fun

coeff_spline = get_coeff_spline(trackLength, discretization)
plot_points = np.linspace(0, 5, 1000)


def plot_spline(spline, plot_points):
    # Evaluate the spline at the plot points
    spline_values = np.array([spline(t) for t in plot_points]).flatten()

    # Plot the spline
    plt.plot(plot_points, spline_values, label='B-spline')
    plt.xlabel('t')
    plt.ylabel('Spline value')
    plt.title('Periodic B-spline')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_spline(coeff_spline, plot_points)

print(coeff_spline(0))
print(coeff_spline(0.5))
print(coeff_spline(1.5))
print(coeff_spline(trackLength / discretization))
print(coeff_spline(trackLength / discretization + 0.5))
print(coeff_spline(trackLength / discretization - 0.5))

