import casadi as ca
import numpy as np
from time import perf_counter
from helper_functions import get_demo_track_spline


class MPC():
    def __init__(self, Q1, Q2, R1, R2, R3, target_speed, N, car, dt):
        self.Q1 = Q1
        self.Q2 = Q2
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.target_speed = target_speed
        self.N = N
        self.car = car
        self.dt = dt

    def get_ip_solver(self, N, dt, ns, nu):

        # state and input constraint values
        state_min = [-20.0, -20.0, -1000.0, 0.0, -2.0, -4.0, 0.0, -0.4, 0.0]
        state_max = [20.0, 20.0, 1000.0, 3.5, 2.0, 4.0, 0.7, 0.4, 1000.0]
        input_min = [-2.0, -15.0, 0.0]
        input_max = [2.0, 15.0, 3.5]


        # symbolic state variables
        states = ca.MX.sym('s', 9, N + 1)
        x_p = states[0, :]
        y_p = states[1, :]
        yaw = states[2, :]
        v_x = states[3, :]
        v_y = states[4, :]
        omega = states[5, :]
        acc = states[6, :]
        delta = states[7, :]
        theta = states[8, :]

        # symbolic input variables
        inputs = ca.MX.sym('u', 3, N)
        dacc = inputs[0, :]
        ddelta = inputs[1, :]
        dtheta = inputs[2, :]

        # initial state in params
        x0 = ca.MX.sym('x0', ns)

        # get track spline for track constraints
        x_spline, y_spline, dy_spline, dx_spline = get_demo_track_spline()

        # defining objective
        objective = 0
        for i in range(N):
            # adding quadratic input cost to objective
            objective += (self.R1 * dacc[i])**2 \
                + (self.R2 * ddelta[i])**2 \
                + (self.R3 * (dtheta[i] - self.target_speed))**2

        for i in range(N+1):
            # defining lag and contouring error
            eC = dy_spline(theta[i]) * (x_p[i] - x_spline(theta[i])) - \
                dx_spline(theta[i]) * (y_p[i] - y_spline(theta[i]))
            eL = -dx_spline(theta[i]) * (x_p[i] - x_spline(theta[i])) - \
                dy_spline(theta[i]) * (y_p[i] - y_spline(theta[i]))

            # adding quadratic stage cost to objective
            objective += (self.Q1 * eC)**2 \
                + (self.Q2 * eL)**2 \


            objective = 0.5 * objective

        # defining constraints
        constraints = []

        # initial state constraint
        constraints = ca.vertcat(constraints, states[:, 0] - x0)

        # constraint the states to follow car dynamics
        for i in range(N):
            constraints = ca.vertcat(
                constraints, states[:, i + 1] - states[:, i] - self.car.state_update_rk4(states[:, i], inputs[:, i], dt))

        # constraint the states to be within the track limits
        for i in range(1, N+1):
            constraints = ca.vertcat(constraints, ca.constpow(x_p[i] - x_spline(theta[i]), 2) + ca.constpow(y_p[i] - y_spline(theta[i]), 2))

        # initial state ns, dynamics N*ns, track limits N
        h_min = np.concatenate((np.zeros(ns), np.zeros(N*ns), np.zeros(N)))
        h_max = np.concatenate((np.zeros(ns), np.zeros(N*ns), 0.23 * 0.23 * np.ones(N))) # 0.23 * 0.23

        x_min = np.concatenate(
            (np.tile(state_min, N + 1), np.tile(input_min, N)))
        x_max = np.concatenate(
            (np.tile(state_max, N + 1), np.tile(input_max, N)))

        x = ca.veccat(states, inputs)
        parameters = ca.veccat(x0)

        print("Compiling IPOPT solver...")
        t0 = perf_counter()
        IP_nlp = {'x': x, 'f': objective, 'p': parameters, 'g': constraints}
        IP_solver = ca.nlpsol('S', 'ipopt', IP_nlp, {'ipopt': {'linear_solver': 'mumps' , 'max_iter': 100}}) #'linear_solver': 'ma57'
                              
        t1 = perf_counter()
        print("Finished compiling IPOPT solver in " + str(t1 - t0) + " seconds!")

        return IP_solver, x_min, x_max, h_min, h_max


    # solve MPC optimization problem
    def mpc_controller(self, IP_solver, x_min, x_max, h_min, h_max, initial_guess, parameters, N, ns, nu):
        sol = IP_solver(x0=initial_guess, p=parameters,
                        lbx=x_min, ubx=x_max, lbg=h_min, ubg=h_max)
        print("Optimization has ended!")
        # input("Press Enter to continue...")

        # Extract the solution and separate states and inputs
        sol_x = sol['x'].full().flatten()
        opt_states = sol_x[0:ns*(N+1)].reshape((N+1, ns))
        opt_inputs = sol_x[ns*(N+1):].reshape((N, nu))

        if IP_solver.stats()['return_status'] != 'Solve_Succeeded':
            print("---------------------------------" + IP_solver.stats()['return_status'] + "---------------------------------")
            #plot_trajectory(opt_states, 1)

        return opt_states, opt_inputs
