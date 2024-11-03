import numpy as np
import casadi as ca

# Defining the disturbance_function function
thetas = np.linspace(0, 24, 1000)
thetas = thetas[:-1]
disturbance_function = 1.5 * np.sin(thetas/1.5) + 1/3 * \
    thetas - 1/20 * (thetas - 10) ** 2


# Kalman filter
disturbance_vector = np.random.rand(25)


class KalmanFilterSpline():
    def __init__(self, car, disturbance_vector, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.disturbance_vector = disturbance_vector
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.l = 11.568244517641709 # track length

    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        arg = arg - self.l * ca.floor(arg / self.l)
        idx = arg/0.25
        idx2 = int(ca.floor(idx))
        if idx2 == 0:
            idx1 = -1
            idx3 = 1
            idx4 = 2
        elif idx2 == 45:
            idx1 = -3
            idx2 = -2
            idx3 = -1
            idx4 = 0
        elif idx2 == 46:
            idx1 = -2
            idx2 = -1
            idx3 = 0
            idx4 = 1
        else:
            idx1 = idx2 - 1
            idx3 = idx2 + 1
            idx4 = idx3 + 1
        
        # Create the y array
        y = np.array([self.disturbance_vector[idx1], self.disturbance_vector[idx2],
                    self.disturbance_vector[idx3], self.disturbance_vector[idx4]])

        # Create the grid points as a list of lists
        grid_points = np.array([float(idx1), float(idx2), float(idx3), float(idx4)])

        #grid_points = np.array([1.0,2.0,3.1,4.0])

        # Create the B-spline interpolant
        y_spline = ca.interpolant('LUT', 'bspline', [grid_points], y)

        # weighting for point update according to distance of position to interpolation point
        K = np.zeros(47)
        K[idx1] = (idx - idx2) * (idx - idx3) * (idx - idx4) / \
            ((idx1 - idx2) * (idx1 - idx3) * (idx1 - idx4))
        K[idx2] = (idx - idx1) * (idx - idx3) * (idx - idx4) / \
            ((idx2 - idx1) * (idx2 - idx3) * (idx2 - idx4))
        K[idx3] = (idx - idx1) * (idx - idx2) * (idx - idx4) / \
            ((idx3 - idx1) * (idx3 - idx2) * (idx3 - idx4))
        K[idx4] = (idx - idx1) * (idx - idx2) * (idx - idx3) / \
            ((idx4 - idx1) * (idx4 - idx2) * (idx4 - idx3))
        
        # tuning factor since actual Kalman filter not applicable
        tune = 0.3

        self.disturbance_vector = self.disturbance_vector + tune * K * (d_mes - float(y_spline(idx)))

        return self.disturbance_vector







# Plotting the disturbance_function function
# plt.plot(thetas, disturbance_function)
# plt.scatter(d_est_vec[:, 0], d_est_vec[:, 1])
# plt.show()



arr = np.array([-0.01970144, -0.01994053, -0.01735735, -0.01351775, -0.0087422 ,
       -0.00352694,  0.00198089,  0.00737082,  0.0124135 ,  0.0167789 ,
        0.0200589 ,  0.02191159,  0.02180851,  0.01956136,  0.01593869,
        0.01147354,  0.00633809,  0.00077836, -0.00498705, -0.0100442 ,
       -0.0147332 , -0.01844875, -0.02149558, -0.02208161, -0.02093207,
       -0.01795775, -0.01408175, -0.0087681 , -0.0024594 ,  0.00343734,
        0.00858253,  0.01271214,  0.01621189,  0.01880859,  0.02023872,
        0.02035309,  0.01904276,  0.01634352,  0.01199343,  0.00818395,
       -0.00157512, -0.00120727, -0.00907375, -0.0071243 , -0.01993629,
       -0.01553303, -0.01487925])