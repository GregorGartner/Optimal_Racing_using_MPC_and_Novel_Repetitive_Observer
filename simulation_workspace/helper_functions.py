import numpy as np
import casadi as ca
import yaml
from track_object import Track
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm





def plot_trajectory(traj, final_traj):
    # Plot the track (assuming plot_track is defined elsewhere)
    ax = plot_track()
    
    x = traj[:, 0]
    y = traj[:, 1]

    # Normalize the trajectory index for colormap
    num_points = len(x)
    norm = plt.Normalize(vmin=0, vmax=num_points-1)
    cmap = cm.viridis  # You can change this to any other colormap

    # Plotting the trajectory with gradient color scheme
    scatter = plt.scatter(x, y, c=np.arange(num_points), cmap=cmap, norm=norm, s=10)
    
    # Adding colorbar and associating it with the scatter plot
    plt.colorbar(scatter, label="Progression (Time)")
    
    # Display options
    if final_traj == 0:
        plt.pause(5)
    else:
        plt.show()





# animate trajectory as video
def animate_trajectory(traj, tail_length=2, dt=0.1):
    """
    Animate a car trajectory with a dark blue current point and a light blue tail.

    Parameters:
    - x: Array of x coordinates.
    - y: Array of y coordinates.
    - tail_length: Number of points in the tail.
    - dt: Time between frames (in seconds).
    """

    x = traj[:, 0]
    y = traj[:, 1]
    
    # Calculate interval in milliseconds
    interval = 1000 * dt

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
    
    # Call plot_track with the axis object to plot the track on the same axis
    plot_track(ax)

    # Line that will represent the trajectory (initialize empty)
    line, = ax.plot([], [], lw=5, color="red")

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot at each frame
    def update(frame):
        current_idx = frame
        start_idx = max(0, current_idx - tail_length)

        # Plot the current point as dark blue and the previous points as light blue
        xdata = x[start_idx:current_idx+1]
        ydata = y[start_idx:current_idx+1]

        # Tail color gradient: last points as lighter
        line.set_data(xdata, ydata)
        # line.set_color("blue")
        line.set_color("red")
        line.set_linewidth(5)

        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(
        x), init_func=init, interval=interval, blit=True)

    # Show the animation
    plt.show()





# track spline
def get_demo_track_spline():
    track = yaml.load(open("DEMO_TRACK.yaml", "r"),
                      Loader=yaml.SafeLoader)["track"]
    trackLength = track["trackLength"]
    indices = [i for i in range(1500, 5430, 10)]
    x = np.array(track["xCoords"])[indices]
    y = np.array(track["yCoords"])[indices]
    dx = np.array(track["xRate"])[indices]
    dy = np.array(track["yRate"])[indices]
    t = np.array(track["arcLength"])[indices] - 0.5 * trackLength

    t_sym = ca.MX.sym('t')

    x_spline = ca.interpolant('LUT', 'bspline', [t], x)
    x_exp = x_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    x_fun = ca.Function('f', [t_sym], [x_exp])

    y_spline = ca.interpolant('LUT', 'bspline', [t], y)
    y_exp = y_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    y_fun = ca.Function('f', [t_sym], [y_exp])

    dx_spline = ca.interpolant('LUT', 'bspline', [t], dx)
    dx_exp = dx_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    dx_fun = ca.Function('f', [t_sym], [dx_exp])

    dy_spline = ca.interpolant('LUT', 'bspline', [t], dy)
    dy_exp = dy_spline(ca.fmod(t_sym + 0.5 * trackLength, trackLength))
    dy_fun = ca.Function('f', [t_sym], [dy_exp])

    return x_fun, y_fun, dx_fun, dy_fun, trackLength





# get initial guess to "warm start" the solver
def get_initial_guess(N, ns, nu):
    lap_states = np.load("lap1_states.npy")
    lap_inputs = np.load("lap1_inputs.npy")
    track_length = 0.3 * (26 + 4*np.pi)

    x_guess = np.zeros((N + 1) * ns + N * nu)
    # arranging the importet lap data in the initial guess
    for i in range(N):
        x_guess[i * ns: (i + 1) * ns] = lap_states[i]
        x_guess[(N + 1) * ns + i * nu: (N + 1) *
                ns + (i + 1) * nu] = lap_inputs[i]
    x_guess[N * ns: (N + 1) * ns] = lap_states[0]
    x_guess[(N + 1) * ns - 1] += track_length
    x_guess[(N + 1) * ns - 1 - 6] += 2 * np.pi

    for i in range(N+1):
        x_guess[i * ns + 8] -= track_length
        x_guess[i * ns + 2] -= 2 * np.pi
        x_guess[i * ns + 1] -= 1
    return x_guess





# creates plot of track spline w\o plt.show()
def plot_track_spline(ax = None):
    x_spline, y_spline, dx_spline, dy_spline, trackLength = get_demo_track_spline()

    range = ca.ceil(trackLength)

    # Generate parameter values for plotting
    theta = np.linspace(0, range, int(range * 100 / 12))  # 100 points from 0 to 4

    # Evaluate the spline functions
    x_vals = x_spline(theta)
    y_vals = y_spline(theta)

    if ax is None:
        # Step 3: Plot the splines
        plt.figure()
        plt.plot(x_vals, y_vals, label='Spline Curve', color='blue')
        plt.title('Spline Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.grid()
        plt.legend()
    
    else:
        # Plot the track spline on the provided axes
        ax.plot(x_vals, y_vals, label='Track', color='blue')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid()
        ax.legend()





def track_pos(t):
    trackobj = Track()
    a = t

    t = 0.3
    trackobj.add_line(3.5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, -np.pi / 2)
    trackobj.add_line(1 * t)
    trackobj.add_turn(t, -np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(2 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(5 * t)
    trackobj.add_turn(t, np.pi / 2)
    trackobj.add_line(4.5 * t)

    x, y = trackobj.point_at_arclength(a)
    x += 0.15
    y -= 1.05
    return x, y





# plotting track with center line and boundaries
def plot_track(ax = None):
    track_length = 0.3 * (26 + 4*np.pi)
    track_N = 600
    track_t = np.linspace(0, track_length, track_N)

    xtrack = np.zeros(track_N)
    ytrack = np.zeros(track_N)
    xrate = np.zeros(track_N)
    yrate = np.zeros(track_N)

    for ii in range(track_N):
        t = track_t[ii]
        xtrack[ii], ytrack[ii] = track_pos(t)

        eps = 0.001
        x_p, y_p = track_pos(t + eps)
        x_m, y_m = track_pos(t - eps)
        xrate[ii] = (x_p - x_m) / (2 * eps)
        yrate[ii] = (y_p - y_m) / (2 * eps)

    half_track_width = 0.23

    # plot the track and its boundaries
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(xtrack, ytrack, 'k--')
    ax.plot(xtrack + half_track_width * yrate, ytrack - half_track_width * xrate, 'k')
    ax.plot(xtrack - half_track_width * yrate, ytrack + half_track_width * xrate, 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax



# plotting track with center line and boundaries
def plot_track1(ax = None):
    x_spline, y_spline, dx_spline, dy_spline, tracklength = get_demo_track_spline()
    track_N = 600
    track_t = np.linspace(0, tracklength, track_N)

    xtrack = np.zeros(track_N)
    ytrack = np.zeros(track_N)
    xrate = np.zeros(track_N)
    yrate = np.zeros(track_N)

    for ii in range(track_N):
        t = track_t[ii]
        xtrack[ii] = x_spline(t)
        ytrack[ii] = y_spline(t)

        xrate[ii] = dx_spline(t)
        yrate[ii] = dy_spline(t)

    half_track_width = 0.23

    # plot the track and its boundaries
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(xtrack, ytrack, 'k--')
    ax.plot(xtrack + half_track_width * yrate, ytrack - half_track_width * xrate, 'k')
    ax.plot(xtrack - half_track_width * yrate, ytrack + half_track_width * xrate, 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax



def plot_all_states(traj, next_pred):
    # plot trajectories of all state components separately
    fig, axs = plt.subplots(3, 2, figsize=(10, 7))
    titles = ["x position", "y position", "yaw angle", "x velocity", "y velocity", "yaw rate"]
    for i in range(6):
        row = i // 2
        col = i % 2
        axs[row, col].plot(traj[:, i], color='red', label="actual")
        axs[row, col].plot(next_pred[:, i], color='blue', label="predicted")
        axs[row, col].set_title(titles[i])
        axs[row, col].legend()
    plt.tight_layout()
    plt.show()





# animate trajectory as video
def animate_predictions(traj, predictions, dt=0.1):

    # get x and y coordinates
    x = traj[:, 0]
    y = traj[:, 1]
    x_pred = predictions[:, :, 0]
    y_pred = predictions[:, :, 1]
    
    # Calculate interval in milliseconds
    interval = 1000 * dt

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
    
    # Call plot_track with the axis object to plot the track on the same axis
    plot_track(ax)

    # Line that will represent the trajectory (initialize empty)
    line, = ax.plot([], [], lw=5, color="red")
    current_pos, = ax.plot([], [], 'bo', markersize=10)

    # Initialize the animation
    def init():
        line.set_data([], [])
        current_pos.set_data([], [])
        return line, current_pos

    # Update function for the first animation
    def update1(frame):

        # Plot the current point as dark blue and the previous points as light blue
        xdata = x_pred[frame+1, : ]
        ydata = y_pred[frame+1, : ]

        # Update the line data
        line.set_data(xdata, ydata)
        line.set_color("red")
        line.set_linewidth(5)

        current_pos.set_data([x[frame]], [y[frame]])

        return line, current_pos

    # Update function for the second animation
    def update2(frame):
        # Update the line data
        current_pos.set_data([x[frame]], [y[frame]])
        return current_pos,

    assert len(x_pred) == len(y_pred), "x_pred and y_pred must have the same length"

    # Create the animation
    ani1 = FuncAnimation(fig, update1, frames=len(x), init_func=init, interval=interval, blit=True)

    # Show the animation
    plt.show()
