import os
import typing
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
#import GPy



class GaussianProcess():
    def __init__(self, tracklength, car):

        self.tracklength = tracklength
        self.car = car

        self.y_data_list = []
        self.x_data_list = []
        self.kernel = ExpSineSquared(length_scale=1, periodicity=tracklength)

    def update(self, prev_state, u0, new_state, dt, i):
        # retrieve measured data and current position of car along track
        d_mes = new_state[3] - prev_state[3] - self.car.state_update_rk4(prev_state, u0, dt, True)[3]
        arg = prev_state[8]

        # update data lists
        self.y_data_list.append(d_mes)
        self.x_data_list.append(arg)


        if i%200 == 0 and i > 0:
            # Convert lists to numpy arrays
            X = np.array(self.x_data_list)
            y = np.array(self.y_data_list)

            # Reshape X to be a 2D array
            if X.ndim == 1:
                X = X.reshape(-1, 1)


            # Fit Gaussian Process
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=0.1)
            gp.fit(X, y)

            # Make predictions
            X_pred = np.linspace(0, self.tracklength, 1000).reshape(-1, 1)
            y_pred, sigma = gp.predict(X_pred, return_std=True)

            # Plot the results
            plt.figure()
            plt.plot(X, y, 'r.', markersize=10, label='Observations')
            plt.plot(X_pred, y_pred, 'b-', label='Prediction')
            plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k', label='95% confidence interval')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.legend(loc='upper left')
            plt.show()

