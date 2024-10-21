import casadi as ca
from time import perf_counter
import numpy as np
import yaml
import matplotlib.pyplot as plt

# # Declare variables
# x = ca.SX.sym("x")

# # Form the NLP
# f = -(x-5)**2 # objective
# g = x[0]+x[1]-10  



# sol = IP_solver(lbg=0)

# # Print solution
# print("-----")
# print("objective at solution = ", sol["f"])
# print("primal solution = ", sol["x"])
# print("dual solution (x) = ", sol["lam_x"])
# print("dual solution (g) = ", sol["lam_g"])


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


x_spline, y_spline, _, _ = get_demo_track_spline()


# Generate parameter values for plotting
theta = np.linspace(0, 12, 100)  # 100 points from 0 to 4

# Evaluate the spline functions
x_vals = x_spline(theta)
y_vals = y_spline(theta)

# Step 3: Plot the splines
plt.figure()
plt.plot(x_vals, y_vals, label='Spline Curve', color='blue')
plt.title('Spline Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.legend()
plt.show()

#plt.plt(x_spline, y_spline)
#plt.show()








