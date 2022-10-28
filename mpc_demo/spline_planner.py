import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class SplinePlanner:
    def __init__(self, x, y, ds=0.1):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            ds (float, optional): _description_. Defaults to 0.1.
        """

        self.x_out = None
        self.y_out = None
        self.yaw_out = None
        self.k_out = None
        self.s_out = None

        # self.interpolate_poses(x, y, ds)
        self._interpolate_iterative(x, y, ds)

    def _arc_length(self, x, y):
        """_Calculate an estimate for the curve's arc length._
        The approximation is the cumulative chordal distance
        """

        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)

        s = [0]
        s.extend(np.cumsum(ds))

        return s

    def _interpolate_poses(self, x_in, y_in, ds):
        """_Interpolates the 2D pose (x, y, yaw)_"""
        s_in = self._arc_length(x_in, y_in)

        tck_x = interpolate.splrep(s_in, x_in)
        tck_y = interpolate.splrep(s_in, y_in)

        # curve will be sampled at every ds meters
        s_out = np.arange(0, s_in[-1], ds)

        # interpolate out x and y at s
        x_out = interpolate.splev(s_out, tck_x, der=0)
        y_out = interpolate.splev(s_out, tck_y, der=0)

        # # interpolate out the derivatives dx/ds and dy/ds
        dx_out = interpolate.splev(s_out, tck_x, der=1)
        dy_out = interpolate.splev(s_out, tck_y, der=1)

        # orientation of new waypoints in the range [-pi, pi]
        yaw_out = np.arctan2(dy_out, dx_out)

        self.s_out = s_out
        self.x_out = x_out
        self.y_out = y_out
        self.yaw_out = yaw_out

        return x_out, y_out, yaw_out, s_out

    def _interpolate_iterative(self, x_in, y_in, ds=0.1, max_iter=10):

        x_w = x_in
        y_w = y_in

        distance_thresh = 0.01  # [m]
        n_iter = 0

        # goal coordinates
        x_goal = x_in[-1]
        y_goal = y_in[-1]

        while True:

            if n_iter > max_iter:
                break

            n_iter += 1
            x_out, y_out, yaw_out, s_out = self._interpolate_poses(x_w, y_w, ds=0.1)

            dx = x_out[-1] - x_goal
            dy = y_out[-1] - y_goal
            d = math.sqrt(dx ** 2 + dy ** 2)

            print(f" Iteration: {n_iter}\n Distance: {d}")
            if d > distance_thresh:
                # too far, need to interpolate again
                x_w = list(x_out) + [x_goal]
                y_w = list(y_out) + [y_goal]
            else:
                break

        self.s_out = s_out
        self.x_out = x_out
        self.y_out = y_out
        self.yaw_out = yaw_out

        return x_out, y_out, yaw_out, s_out

    def get_poses(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.x_out, self.y_out, self.yaw_out


def normalize_angles(angles_arr):
    """_Normalize angles to [-pi, pi] interval _

    Args:
        angles_arr (np.array): _array of angles_

    Returns:
        _type_: _array of normalized angles_
    """

    angles = np.copy(angles_arr)

    while np.any(angles > np.pi):
        mask = angles > np.pi
        angles[mask] -= 2.0 * np.pi
    while np.any(angles < -np.pi):
        mask = angles < -np.pi
        angles[mask] += 2.0 * np.pi

    return angles


def main():

    # TODO: Add explanation
    R = 3.0
    dth = np.pi / 4.0
    th_w = dth * np.arange(0.0, 9.0, 1.0)

    x_w = R * np.cos(th_w)
    y_w = R * np.sin(th_w)

    planner = SplinePlanner(x_w, y_w)

    x_o, y_o, yaw_o = planner.get_poses()
    s_o = planner.s_out

    # for a circular path, s = theta . R holds
    # ground truth
    x_gt = R * np.cos(np.asarray(planner.s_out) / R)
    y_gt = R * np.sin(np.asarray(planner.s_out) / R)
    yaw_gt = normalize_angles(np.asarray(planner.s_out) / R + np.pi / 2)

    plt.figure(figsize=(10, 10))
    plt.plot(x_gt, y_gt, "b-", label="true")
    plt.plot(x_w, y_w, "xb", label="waypoints")
    plt.plot(x_o, y_o, "r--", label="SplinePlanner")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title("Circular path")
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.plot(s_o, x_o - x_gt, "-", label="x - x_gt")
    plt.plot(s_o, y_o - y_gt, "-", label="y - y_gt")
    plt.plot(s_o, yaw_o - yaw_gt, "-", label="yaw - yaw_gt")
    plt.grid(True)
    plt.xlabel("traveled distance [m]")
    plt.ylabel("Errors")
    plt.title("Error (SplinePlanner - GroundTruth)")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
