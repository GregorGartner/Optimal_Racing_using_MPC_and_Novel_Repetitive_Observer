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
        self.l = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        arg = arg - self.l * ca.floor(arg / self.l)
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
            if idx1 == 46:
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
        self.l = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        if arg > 11.568244517641709*2:
            lap2= 1
        arg = arg - self.l * ca.floor(arg / self.l)
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
        self.l = tracklength # track length
        self.discretization = 0.25

        self.n_points = int(ca.floor(tracklength / self.discretization))

    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[0] - prev_state[0] - self.car.state_update_rk4(prev_state, u0, dt, True)[0]
        arg = prev_state[8]
        laps = ca.floor(arg / self.l)
        arg = arg - self.l * laps
        idx = arg/self.discretization
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
            idx = idx - (self.n_points + 1)
        elif idx2 == self.n_points:
            idx1 = -2
            idx2 = -1
            idx3 = 0
            idx4 = 1
            idx = idx - (self.n_points + 1)
        else:
            idx1 = idx2 - 1
            idx3 = idx2 + 1
            idx4 = idx3 + 1
        
        # Create the y array
        y = np.array([self.disturbance_vector[idx1], self.disturbance_vector[idx2],
                    self.disturbance_vector[idx3], self.disturbance_vector[idx4]])

        # Create the grid points as a list of lists
        grid_points = np.array([float(idx1), float(idx2), float(idx3), float(idx4)])

        # Create the B-spline interpolant
        y_spline = ca.interpolant('LUT', 'bspline', [grid_points], y)

        # weighting for point update according to distance of position to interpolation point
        K = np.zeros(self.n_points + 1)
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
    

    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
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

            # Create the y array
            y = np.array([self.disturbance_vector[idx1], self.disturbance_vector[idx2],
                        self.disturbance_vector[idx3], self.disturbance_vector[idx4]])

            # Create the grid points as a list of lists
            grid_points = np.array([float(idx1), float(idx2), float(idx3), float(idx4)])

            # Create the B-spline interpolant
            y_spline = ca.interpolant('LUT', 'bspline', [grid_points], y)



            d_function[index] = float(y_spline(idx))
        
        return d_function