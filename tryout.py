import casadi as ca
import numpy as np





y = np.array([4.0,3.1,2.0,1.0])
grid = np.array([1.0,2.0,3.1,4.0])

y_spline = ca.interpolant('LUT', 'bspline', [grid] ,y)

print(y_spline)