import numpy as np
import matplotlib.pyplot as plt
import random
import casadi as ca

# trackLength: 11.568244517641709

class KalmanFilterLinear():
    def __init__(self, car, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points)


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[3] - prev_state[3] - self.car.state_update_rk4(prev_state, u0, dt, True)[3]
        arg = prev_state[8]
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx1 = int(ca.floor(idx))
        if idx1 == self.n_points - 1:
            idx1 = -1
            idx2 = 0
        else:
            idx2 = idx1 + 1

        # creating time varying measurement / interpolation matrix H_k
        H_k = np.zeros(self.n_points)
        H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
        H_k[idx2] = (idx - idx1) / (idx2 - idx1)

        # variance process update
        self.P += self.P + self.Q

        # Kalman gain computation and measurement update of mean of estimates
        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        return self.disturbance_vector

    
    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            idx1 = int(ca.floor(idx))
            if idx1 == self.n_points - 1:
                idx2 = 0
            else:
                idx2 = idx1 + 1

            # creating time-varying meausrement matrix to get function value for plotting
            H_k = np.zeros(self.n_points)
            H_k[idx1] = (idx2 - idx) / (idx2 - idx1)
            H_k[idx2] = (idx - idx1) / (idx2 - idx1)

            d_function[index] = H_k @ self.disturbance_vector
        
        return d_function
    


class KalmanFilterPolynomial():
    def __init__(self, car, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points)


    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[3] - prev_state[3] - self.car.state_update_rk4(prev_state, u0, dt, True)[3]
        arg = prev_state[8]
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx2 = int(ca.floor(idx))
        if idx2 == 0:
            idx1 = -1
            idx3 = 1
            idx4 = 2
        elif idx2 == self.n_points - 2:
            idx1 = -3
            idx2 = -2
            idx3 = -1
            idx4 = 0
        elif idx2 == self.n_points - 1:
            idx1 = -2
            idx2 = -1
            idx3 = 0
            idx4 = 1
        else:
            idx1 = idx2 - 1
            idx3 = idx2 + 1
            idx4 = idx3 + 1

        # creating time varying measurement / interpolation matrix H_k
        H_k = np.zeros(self.n_points)
        H_k[idx1] = (idx - idx2) * (idx - idx3) * (idx - idx4) / \
            ((idx1 - idx2) * (idx1 - idx3) * (idx1 - idx4))
        H_k[idx2] = (idx - idx1) * (idx - idx3) * (idx - idx4) / \
            ((idx2 - idx1) * (idx2 - idx3) * (idx2 - idx4))
        H_k[idx3] = (idx - idx1) * (idx - idx2) * (idx - idx4) / \
            ((idx3 - idx1) * (idx3 - idx2) * (idx3 - idx4))
        H_k[idx4] = (idx - idx1) * (idx - idx2) * (idx - idx3) / \
            ((idx4 - idx1) * (idx4 - idx2) * (idx4 - idx3))
        
        # variance process update
        self.P += self.P + self.Q

        # Kalman gain computation and measurement update of mean of estimates
        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        return self.disturbance_vector
    



    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            idx2 = int(ca.floor(idx))
            if idx2 == 0:
                idx1 = self.n_points - 1
                idx3 = 1
                idx4 = 2
            elif idx2 == self.n_points - 2:
                idx1 = self.n_points - 3
                idx3 = self.n_points - 1
                idx4 = 0
            elif idx2 == self.n_points - 1:
                idx1 = self.n_points - 2
                idx3 = 0
                idx4 = 1
            else:
                idx1 = idx2 - 1
                idx3 = idx2 + 1
                idx4 = idx3 + 1

            # creating time-varying meausrement matrix to get function value for plotting
            H_k = np.zeros(self.n_points)
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
    def __init__(self, car, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points)

        self.P = P * np.eye(self.n_points) # initial state variance

        # spline to determe coefficients for interpolation points depending on distance to point of interest
        #self.coeff_spline = self.get_coeff_spline()
        

    #def get_coeff_spline(self):
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

        idx_sym = ca.MX.sym('idx')
        H_k_sym = ca.MX.zeros(self.n_points)
        for index in range(self.n_points):
            H_k_sym[index] = coeff_spline(idx_sym - ca.MX(index))
        self.H_k_func = ca.Function('H_k_func', [idx_sym], [H_k_sym])

        #return coeff_spline
    

    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[5] - prev_state[5] - self.car.state_update_rk4(prev_state, u0, dt, True)[5]
        arg = prev_state[8]
        laps = ca.floor(arg / self.trackLength)
        arg = arg - self.trackLength * laps
        idx = arg/self.discretization

        # variance process update
        self.P += self.P + self.Q
        
        # Computing measurement matrix H for the Kalman gain
        H_k = np.array(self.H_k_func(idx).T).flatten()
        
        # Create the y array
        # y = self.disturbance_vector

        # Create the grid points as a list of lists
        # grid_points = np.arange(self.n_points, dtype=float)

        # compute spline for current disturbance estimates
        # y_spline = ca.interpolant('spline', 'bspline', [grid_points], y)

        # Kalman gain computation and mean measurement update
        K = self.P @ H_k.T / (H_k @ (self.P @ H_k.T) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        # Convert CasADi DM to numpy array
        #self.disturbance_vector = np.array(self.disturbance_vector)
        return self.disturbance_vector.flatten()
    

    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)
        
        # Create the y array
        # y = self.disturbance_vector

        # Create the grid points for spline interpolation
        # grid_points = np.arange(self.n_points, dtype=float)

        # Create the B-spline interpolant
        # y_spline = ca.interpolant('spline', 'bspline', [grid_points], y)


        for i, theta in enumerate(thetas):
            idx = theta/self.discretization
            H_k = np.array(self.H_k_func(idx).T)
            d_function[i] = H_k @ self.disturbance_vector
        
        return d_function
    


class KalmanFilterLinearSpline():
    def __init__(self, car, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points) # current disturbance estimates used for interpolation

        self.grid_points = np.arange(self.n_points, dtype=float)
        self.y_spline = ca.interpolant('LUT', 'linear', [self.grid_points], self.disturbance_vector)



    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[5] - prev_state[5] - self.car.state_update_rk4(prev_state, u0, dt, True)[5]
        arg = prev_state[8]
        arg = arg - self.trackLength * ca.floor(arg / self.trackLength)
        idx = arg / self.discretization
        idx1 = int(ca.floor(idx))
        if idx1 == self.n_points - 1:
            idx1 = -1
            idx2 = 0
        else:
            idx2 = idx1 + 1

        # Computing measurement matrix H for the Kalman gain
        H_k = np.zeros(self.n_points)
        H_k[idx1] = (idx2 - idx)
        H_k[idx2] = (idx - idx1)

        # Variance process update with dynamics matrix A = identity
        self.P += self.P + self.Q

        # Kalman gain computation and mean measurement update
        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - float(self.y_spline(idx)))

        # calculate new spline using updated disturbance estimate
        self.y_spline = ca.interpolant('LUT', 'linear', [self.grid_points], self.disturbance_vector)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        return self.disturbance_vector

    
    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for index, theta in enumerate(thetas):
            idx = theta/self.discretization
            d_function[index] = float(self.y_spline(idx))
        
        return d_function



class KalmanFilterExponential():
    def __init__(self, car, tracklength, P=1, R=0.3):
        # initializing estimator parameters
        self.car = car
        self.trackLength = tracklength # track length
        self.discretization = 0.25
        self.P = P # initial state variance
        self.R = R # measurement noise variance
        self.Q = 1.0 # process noise variance
        self.track_index = tracklength / self.discretization # unrounded highest index of track
        self.n_points = int(ca.ceil(tracklength / self.discretization))
        self.disturbance_vector = np.zeros(self.n_points)
        self.bandwidth = 0.05

        # define "masking" matrix M whose inverse determines the weights for each gaussian to interpolate the points
        M = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(self.n_points):
                M[i, j] = ca.exp(- (ca.sin((ca.pi/self.track_index)*(i - j))**2) / self.bandwidth)

        self.M_inv = np.linalg.inv(M)
    

    def estimate(self, prev_state, u0, new_state, dt):
        d_mes = new_state[5] - prev_state[5] - self.car.state_update_rk4(prev_state, u0, dt, True)[5]
        arg = prev_state[8]
        laps = ca.floor(arg / self.trackLength)
        arg = arg - self.trackLength * laps
        idx = arg/self.discretization

        # variance process update
        self.P += self.P + self.Q

        phi = np.zeros(self.n_points)
        for index in range(len(phi)):
            phi[index] = ca.exp(- (ca.sin((ca.pi/self.track_index)*(idx - index))**2) / self.bandwidth)

        H_k = phi.T @ self.M_inv

        # Kalman gain computation and mean measurement update
        K = self.P * H_k.T / (H_k @ (self.P * H_k) + self.R)
        self.disturbance_vector = self.disturbance_vector + K * (d_mes - H_k @ self.disturbance_vector)

        # variance measurement update
        self.P = (1 - K @ H_k) * self.P

        return self.disturbance_vector
    

    # return values of measured disturbance function as array for plotting
    def function(self, thetas):
        d_function = np.zeros(thetas.shape)

        for i, theta in enumerate(thetas):
            idx = theta/self.discretization

            phi = np.zeros(self.n_points)
            for index in range(len(phi)):
                phi[index] = ca.exp(- (ca.sin((ca.pi/self.track_index)*(idx - index))**2) / self.bandwidth)

            H_k = phi.T @ self.M_inv

            d_function[i] = H_k @ self.disturbance_vector
        
        return d_function