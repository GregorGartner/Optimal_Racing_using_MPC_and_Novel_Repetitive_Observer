import casadi as ca
from time import perf_counter
import numpy as np
import yaml
import matplotlib.pyplot as plt
from math import cos, sin, fabs
from track_object import Track


N = 30
ns = 9 


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


def track_params(t):
    trackarr = []

    for ti in t:
        xtrack, ytrack = track_pos(ti)

        eps = 0.001
        x_p, y_p = track_pos(ti + eps)
        x_m, y_m = track_pos(ti - eps)
        xrate = (x_p - x_m) / (2 * eps)
        yrate = (y_p - y_m) / (2 * eps)

        arcLength = ti
        tangentAngle = np.atan2(yrate, xrate)

        tracki = np.array([xtrack, ytrack, xrate, yrate, tangentAngle, arcLength])
        trackarr = np.concatenate((trackarr, tracki))

    return trackarr



def plot_track():
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xtrack, ytrack, 'k--')
    ax.plot(xtrack + half_track_width * yrate, ytrack - half_track_width * xrate, 'k')
    ax.plot(xtrack - half_track_width * yrate, ytrack + half_track_width * xrate, 'k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax



ax = plot_track()
plt.show()