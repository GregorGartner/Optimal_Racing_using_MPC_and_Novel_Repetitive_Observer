import numpy as np
import matplotlib.pyplot as plt
import random
import casadi as ca

# trackLength: 11.568244517641709

class KalmanFilter():
    def __init__(self, car, tracklength, num_fitting_points, KF_type, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength
        self.n_points = num_fitting_points
        self.discretization = tracklength / num_fitting_points
        #self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros((self.n_points, 3))

        self.P = P * np.eye(self.n_points) # initial state variance
        self.R = R # measurement noise variance
        self.Q = np.eye(self.n_points) # process noise variance
        self.bandwidth = 0.0001 # bandwidth for exponential kernel

        #wrap_length = self.trackLength / self.discretization # length of the track in indices (non integer value)
        #n_indices = int(ca.floor(self.trackLength / self.discretization)) # integer value of highest index

        # create bspline and linear kernel
        if KF_type == 'bspline' or KF_type == 'linear':

            # Define the control points with all zeros but one at the center
            control_points = np.zeros(self.n_points+1)
            control_points[int(self.n_points/2)] = 1.0
            grid_points = np.linspace(-self.n_points/2, self.n_points/2, self.n_points+1)

            # creating the splines with the activation at arg=0
            x_spline = ca.interpolant('LUT', KF_type, [grid_points], control_points)

            # Define the symbolic variable to create periodic function
            t_sym = ca.MX.sym('t')
            # Use the modulo operation to wrap around the parameter values
            x_exp = x_spline(ca.fmod(t_sym + int(self.n_points/2), self.n_points) - int(self.n_points/2))
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

            # creating a vector function with periodic, squared exponential values depending on distance of interpolation point to position of car
            idx_sym = ca.MX.sym('idx')
            phi_sym = ca.MX.zeros(self.n_points)
            for index in range(self.n_points):
                phi_sym[index] = ca.exp(- (ca.sin((ca.pi/self.n_points)*(idx_sym - index))**2) / self.bandwidth)
            self.H_k_func = ca.Function('H_k_func', [idx_sym], [phi_sym])
    



    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[2:5] - prev_state[2:5] - self.car.state_update_rk4(prev_state, u0, dt, True)[2:5]
        arg = prev_state[8] # position of the car on the track
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
        self.disturbance_vector = self.disturbance_vector + np.outer(K, (d_mes - current_estimate))

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        # Convert CasADi DM to numpy array
        #self.disturbance_vector = np.array(self.disturbance_vector)
        return self.disturbance_vector.flatten(), current_estimate, KF_error, d_mes
    

    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros((len(thetas), 3))

        for i, theta in enumerate(thetas):
            idx = theta/self.discretization
            H_k = np.array(self.H_k_func(idx).T).flatten()
            d_function[i, :] = H_k @ self.disturbance_vector
        
        return d_function