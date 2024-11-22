import casadi as ca
import numpy as np




class CarModel():
    def __init__(self, lr, lf, m, I, Df, Cf, Bf, Dr, Cr, Br, Cm1, Cm2, Cd, Croll):
        # initializing car parameters
        self.lr = lr
        self.lf = lf
        self.m = m
        self.I = I
        self.Df = Df
        self.Cf = Cf
        self.Bf = Bf
        self.Dr = Dr
        self.Cr = Cr
        self.Br = Br
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cd = Cd
        self.Croll = Croll

    # dynamics function of the car
    def dynamics(self, x, u, numerical=False):

        # extract state information
        yaw = x[2]
        v_x = x[3]
        v_y = x[4]
        omega = x[5]
        acc = x[6]
        delta = x[7]

        # calculate forces and slipping angles
        Fx = (self.Cm1 - self.Cm2 * v_x) * acc - self.Cd * v_x * v_x - self.Croll
        alpha_f = - ca.arctan((omega * self.lf + v_y) / v_x) + delta
        alpha_r = ca.arctan((omega * self.lr - v_y) / v_x)
        Fr = self.Dr * ca.sin(self.Cr * ca.arctan(self.Br * alpha_r))
        Ff = self.Df * ca.sin(self.Cf * ca.arctan(self.Bf * alpha_f))

        # returning the differential equations governing the car dynamics
        if numerical:
            return np.array([
                v_x * np.cos(yaw) - v_y * np.sin(yaw),
                v_x * np.sin(yaw) + v_y * np.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * np.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * np.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * np.cos(delta) - Fr * self.lr),
                u[0],
                u[1],
                u[2]
            ])
        else:
            return ca.vertcat(
                v_x * ca.cos(yaw) - v_y * ca.sin(yaw),
                v_x * ca.sin(yaw) + v_y * ca.cos(yaw),
                omega,
                1 / self.m * (Fx - Ff * ca.sin(delta) + self.m * v_y * omega),
                1 / self.m * (Fr + Ff * ca.cos(delta) - self.m * v_x * omega),
                1 / self.I * (Ff * self.lf * ca.cos(delta) - Fr * self.lr),
                u[0],
                u[1],
                u[2]
            )


    # calculating one step of discrete dynamics using Runge-Kutta 4th order integration
    def state_update_rk4(self, x, u, dt, numerical=False):
        # Runge-Kutta 4th order integration
        k1 = self.dynamics(x, u, numerical)
        k2 = self.dynamics(x + dt / 2.0 * k1, u, numerical)
        k3 = self.dynamics(x + dt / 2.0 * k2, u, numerical)
        k4 = self.dynamics(x + dt * k3, u, numerical)

        # calculates weighted average of 4 different approximations
        out = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return out
    