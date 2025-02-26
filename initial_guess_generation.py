# manual (hard coded) control
for i in range(N):
    if i < 0:
        u = np.array([0.8, 0, 0])
    elif i < 7:
        u = np.array([0.0, 0.0, 0])
    elif i < 13:
        u = np.array([0.0, 1.0, 0])
    elif i < 21:
        u = np.array([-0.05, 0.0, 0])
    elif i < 26: 
        u = np.array([0.0, -1.0, 0])
    elif i < 34:
        u = np.array([0.0, 0, 0])
    elif i < 50:
        u = np.array([-0.5, 1.1, 0])
    elif i < 65:
        u = np.array([0.8, -1.5, 0.0])
    elif i < 72:
        u = np.array([-0.3, -1.2, 0.0])
    elif i < 84:
        u = np.array([0.0, 0.0, 0.0])
    elif i < 100:
        u = np.array([0.0, 2.0, 0.0])
    elif i < 100:
        u = np.array([0.0, 0.0, 0.0])



    new_state = prev_state + car.state_update_rk4(prev_state, u, dt, True)


    # find closest point on center line to add progress to initial guess
    theta = ca.MX.sym('theta') # optimization variable
    spline_point = [x_spline(theta), y_spline(theta)]
    distance = ca.sqrt((spline_point[0] - new_state[0])**2 + (spline_point[1] - new_state[1])**2)
    nlp = {'x': theta, 'f': distance}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=0.5, lbx=0, ubx=8)
    theta_opt = sol['x'].full().flatten()[0]

    # Get the optimal theta and update progress parameter theta according to current position
    new_state[theta_index] = theta_opt

    traj_guess[i+1] = new_state
    inputs_guess[i] = u

    prev_state = new_state