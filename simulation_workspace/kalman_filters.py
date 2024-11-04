import numpy as np
import matplotlib.pyplot as plt
import random
import casadi as ca

# trackLength: 11.568244517641709

class KalmanFilterLinear():
    def __init__(self, car, disturbance_vector, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.disturbance_vector = disturbance_vector
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.trackLength = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx1 = int(ca.floor(idx))
        if idx1 == self.n_points:
            idx1 = -1
            idx2 = 0
        else:
            idx2 = idx1 + 1

        H_k = np.zeros(self.n_points + 1)
        H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
        H_k[idx2] = (idx - idx1) / (idx2 - idx1)

        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)

        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        return self.disturbance_vector

    

    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            idx1 = int(ca.floor(idx))
            if idx1 == self.n_points:
                idx2 = 0
            else:
                idx2 = idx1 + 1

            H_k = np.zeros(self.n_points + 1)
            H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
            H_k[idx2] = (idx - idx1) / (idx2 - idx1)

            d_function[index] = H_k @ self.disturbance_vector
        
        return d_function
    


class KalmanFilterPolynomial():
    def __init__(self, car, disturbance_vector, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.disturbance_vector = disturbance_vector
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.trackLength = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        if arg > 11.568244517641709*2:
            lap2= 1
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx2 = int(ca.floor(idx))
        if idx2 == 0:
            idx1 = -1
            idx3 = 1
            idx4 = 2
        elif idx2 == self.n_points - 1:
            idx1 = -3
            idx2 = -2
            idx3 = -1
            idx4 = 0
        elif idx2 == self.n_points:
            idx1 = -2
            idx2 = -1
            idx3 = 0
            idx4 = 1
        else:
            idx1 = idx2 - 1
            idx3 = idx2 + 1
            idx4 = idx3 + 1

        H_k = np.zeros(self.n_points + 1)
        H_k[idx1] = (idx - idx2) * (idx - idx3) * (idx - idx4) / \
            ((idx1 - idx2) * (idx1 - idx3) * (idx1 - idx4))
        H_k[idx2] = (idx - idx1) * (idx - idx3) * (idx - idx4) / \
            ((idx2 - idx1) * (idx2 - idx3) * (idx2 - idx4))
        H_k[idx3] = (idx - idx1) * (idx - idx2) * (idx - idx4) / \
            ((idx3 - idx1) * (idx3 - idx2) * (idx3 - idx4))
        H_k[idx4] = (idx - idx1) * (idx - idx2) * (idx - idx3) / \
            ((idx4 - idx1) * (idx4 - idx2) * (idx4 - idx3))

        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)

        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        return self.disturbance_vector
    




    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            idx2 = int(ca.floor(idx))
            if idx2 == 0:
                idx1 = self.n_points
                idx3 = 1
                idx4 = 2
            elif idx2 == self.n_points - 1:
                idx1 = self.n_points - 2
                idx3 = self.n_points
                idx4 = 0
            elif idx2 == self.n_points:
                idx1 = self.n_points - 1
                idx3 = 0
                idx4 = 1
            else:
                idx1 = idx2 - 1
                idx3 = idx2 + 1
                idx4 = idx3 + 1

            H_k = np.zeros(self.n_points + 1)
            H_k[idx1] = (idx - idx2) * (idx - idx3) * (idx - idx4) / \
                ((idx1 - idx2) * (idx1 - idx3) * (idx1 - idx4))
            H_k[idx2] = (idx - idx1) * (idx - idx3) * (idx - idx4) / \
                ((idx2 - idx1) * (idx2 - idx3) * (idx2 - idx4))
            H_k[idx3] = (idx - idx1) * (idx - idx2) * (idx - idx4) / \
                ((idx3 - idx1) * (idx3 - idx2) * (idx3 - idx4))
            H_k[idx4] = (idx - idx1) * (idx - idx2) * (idx - idx3) / \
                ((idx4 - idx1) * (idx4 - idx2) * (idx4 - idx3))

            d_function[index] = H_k @ self.disturbance_vector
        
        return d_function
    


class KalmanFilterSpline():
    def __init__(self, car, disturbance_vector, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.disturbance_vector = disturbance_vector
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.trackLength = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))

    def get_coeff_spline(self):
        wrap_length = self.trackLength / self.discretization
        n_indices = int(ca.floor(self.trackLength / self.discretization))

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
        coeff_spline = ca.Function('f', [t_sym], [x_exp])

        return coeff_spline
    

    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        laps = ca.floor(arg / self.trackLength)
        arg = arg - self.trackLength * laps
        idx = arg/self.discretization

        coeff_spline = self.get_coeff_spline()

        self.P += self.P + self.Q

        H_k = np.zeros(self.n_points + 1)
        for index in range(len(H_k)):
            H_k[index] = coeff_spline(index - idx)
        
        # Create the y array
        y = self.disturbance_vector

        # Create the grid points as a list of lists
        grid_points = np.arange(self.n_points + 1, dtype=float)

        # Create the B-spline interpolant
        y_spline = ca.interpolant('spline', 'bspline', [grid_points], y)

        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)

        self.disturbance_vector = self.disturbance_vector + K * (d_mes - float(y_spline(idx)))

        self.P = (1 - K @ H_k) * self.P

        return self.disturbance_vector
    

    def function(self, thetas):
        d_function = np.zeros(thetas.shape)
        
        # Create the y array
        y = self.disturbance_vector

        # Create the grid points as a list of lists
        grid_points = np.arange(self.n_points + 1, dtype=float)

        # Create the B-spline interpolant
        y_spline = ca.interpolant('spline', 'bspline', [grid_points], y)


        for i, theta in enumerate(thetas):
            idx = theta/self.discretization
            d_function[i] = float(y_spline(idx))
        
        return d_function
    


class KalmanFilterLinearSpline():
    def __init__(self, car, disturbance_vector, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.disturbance_vector = disturbance_vector
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.n_points = int(ca.floor(tracklength / self.discretization))

        self.grid_points = np.arange(self.n_points + 1, dtype=float)
        self.y_spline = ca.interpolant('LUT', 'linear', [self.grid_points], self.disturbance_vector)



    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx1 = int(ca.floor(idx))
        if idx1 == self.n_points:
            idx1 = -1
            idx2 = 0
        else:
            idx2 = idx1 + 1

        H_k = np.zeros(self.n_points + 1)
        H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
        H_k[idx2] = (idx - idx1) / (idx2 - idx1)

        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - float(self.y_spline(idx)))

        self.y_spline = ca.interpolant('LUT', 'linear', [self.grid_points], self.disturbance_vector)

        return self.disturbance_vector

    

    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            idx1 = int(ca.floor(idx))
            if idx1 == self.n_points:
                idx2 = 0
            else:
                idx2 = idx1 + 1

            H_k = np.zeros(self.n_points + 1)
            H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
            H_k[idx2] = (idx - idx1) / (idx2 - idx1)

            d_function[index] = float(self.y_spline(idx))
        
        return d_function
