import numpy as np
import matplotlib.pyplot as plt
import random
import casadi as ca

# trackLength: 11.568244517641709

class KalmanFilter():
    def __init__(self, car, tracklength, discretization, KF_type, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength
        self.discretization = discretization
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points)

        self.P = P * np.eye(self.n_points) # initial state variance
        self.R = R # measurement noise variance
        self.Q = np.eye(self.n_points) # process noise variance
        self.bandwidth = 0.1 # bandwidth for exponential kernel

        wrap_length = self.trackLength / self.discretization # length of the track in indices (non integer value)
        n_indices = int(ca.floor(self.trackLength / self.discretization)) # integer value of highest index

        # create bspline and linear kernel
        if KF_type == 'bspline' or KF_type == 'linear':

            # Define the control points with all zeros but one at the center
            control_points = np.zeros(n_indices+1)
            control_points[int(ca.floor(n_indices/2))] = 1.0
            grid_points = np.linspace(-n_indices/2, n_indices/2, n_indices+1)

            # creating the splines with the activation at arg=0
            x_spline = ca.interpolant('LUT', KF_type, [grid_points], control_points)

            # Define the symbolic variable to create periodic function
            t_sym = ca.MX.sym('t')
            # Use the modulo operation to wrap around the parameter values
            x_exp = x_spline(ca.fmod(t_sym + int(ca.floor(n_indices/2)), wrap_length) - int(ca.floor(n_indices/2)))
            # Create a CasADi function from the periodic expression
            coeff_spline = ca.Function('f', [t_sym], [x_exp])

            # creating a vector function for values depending on distance of interpolation point to position of car
            idx_sym = ca.MX.sym('idx')
            H_k_sym = ca.MX.zeros(self.n_points)
            for index in range(self.n_points):
                H_k_sym[index] = coeff_spline(idx_sym - ca.MX(index))
            self.H_k_func = ca.Function('H_k_func', [idx_sym], [H_k_sym])
        
        # create exponential kernel
        elif KF_type == 'exponential':

            # create the matrix M and its inverse to have meaningful interpolation points
            self.track_index = tracklength / self.discretization
            M = np.zeros((self.n_points, self.n_points))
            for i in range(self.n_points):
                for j in range(self.n_points):
                    M[i, j] = ca.exp(- (ca.sin((ca.pi/self.track_index)*(i - j))**2) / self.bandwidth)
            self.M_inv = np.linalg.inv(M)

            # creating a vector function with periodic, squared exponential values depending on distance of interpolation point to position of car
            idx_sym = ca.MX.sym('idx')
            phi_sym = ca.MX.zeros(self.n_points)
            for index in range(self.n_points):
                phi_sym[index] = ca.exp(- (ca.sin((ca.pi/self.track_index)*(idx_sym - index))**2) / self.bandwidth)
            H_k_sym = self.M_inv @ phi_sym
            self.H_k_func = ca.Function('H_k_func', [idx_sym], [H_k_sym])
    



    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[5] - prev_state[5] - self.car.state_update_rk4(prev_state, u0, dt, True)[5]
        arg = prev_state[8]
        laps = ca.floor(arg / self.trackLength)
        arg = arg - self.trackLength * laps
        idx = arg/self.discretization

        # variance process update
        self.P = self.P + self.Q
        
        # Computing measurement matrix H for the Kalman gain
        H_k = np.array(self.H_k_func(idx).T).flatten()

        # Kalman gain computation and mean measurement update
        K = self.P @ H_k.T / (H_k @ (self.P @ H_k.T) + self.R)
        current_estimate = H_k @ self.disturbance_vector
        KF_error = d_mes - current_estimate
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - current_estimate)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        # Convert CasADi DM to numpy array
        #self.disturbance_vector = np.array(self.disturbance_vector)
        return self.disturbance_vector.flatten(), current_estimate, KF_error
    

    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for i, theta in enumerate(thetas):
            idx = theta/self.discretization
            H_k = np.array(self.H_k_func(idx).T).flatten()
            d_function[i] = H_k @ self.disturbance_vector
        
        return d_function