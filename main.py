"""
System model: differential drive robot
Problem: point stabilization
"""

import math

from mpc_demo.casadi_solver import Simulation


def main():

    # Horizon specs
    N = 10  # number of look ahead steps
    time_step = 0.2

    # Initial state
    x_init = 0
    y_init = 0
    theta_init = 0

    # Target state
    x_target = 10
    y_target = 10
    theta_target = -math.pi / 2

    theta_target = theta_target % (2 * math.pi)

    initial_pose = [x_init, y_init, theta_init]
    target_pose = [x_target, y_target, theta_target]

    sim = Simulation(n_la=N, dt=time_step)
    sim.run_point_stabilization(initial_pose, target_pose)


if __name__ == "__main__":

    main()
