"""
System model: differential drive robot
Problem: point stabilization
"""

from time import time

import casadi as ca
import numpy as np
# from casadi import sin, cos, pi

from mpc_demo.simulation_code import simulate
from mpc_demo.casadi_solver import RobotModel, Solver


def to_array(dm):
    return np.array(dm.full())


def is_goal_reached(current_state, target_state, tol=1e-1):
    return ca.norm_2(current_state - target_state) < tol


def update_state(st, con, dt):
    robot = RobotModel()
    # TODO: add noise
    st_next = robot.update_state(st, con, dt)

    return st_next


def update_initial_conditions(states, controls):

    states = ca.horzcat(states[:, 1:], ca.reshape(states[:, -1], -1, 1))
    controls = ca.horzcat(controls[:, 1:], ca.reshape(controls[:, -1], -1, 1))

    return states, controls


def run(initial_pose: list, target_pose: list, N: int, dt=0.2):

    n_states = 3
    n_controls = 2
    max_sim_time = 200  # simulation time

    current_state = ca.DM(initial_pose)
    target_state = ca.DM(target_pose)

    # initialize the decision variables
    # TODO: rename
    initial_states = ca.repmat(current_state, 1, N + 1)
    initial_controls = ca.DM.zeros((n_controls, N))

    # history variables
    # TODO: rename
    cat_states = to_array(initial_states)
    cat_controls = to_array(initial_controls[:, 0])
    times = np.array([[0]])

    # simulation state
    mpc_iter = 0

    solver = Solver()

    # start a chronometer to time the whole loop
    main_loop = time()  # return time in sec

    goal_reached = False

    while mpc_iter * dt < max_sim_time:
        if is_goal_reached(current_state, target_state):
            goal_reached = True
            break

        # start a chronometer to time one iteration
        t1 = time()

        initial_conditions = ca.vertcat(
            ca.reshape(initial_states, n_states * (N + 1), 1),
            ca.reshape(initial_controls, n_controls * N, 1)
        )
        parameters_vec = ca.vertcat(current_state, target_state)

        solver.reset()
        solver.solve(initial_conditions, parameters_vec)
        opt_states = solver._opt_states
        opt_controls = solver._opt_controls

        # update history variables (for simulation)
        cat_states = np.dstack((cat_states, to_array(opt_states)))
        cat_controls = np.vstack((cat_controls, to_array(opt_controls[:, 0])))
        times = np.vstack((times, time() - t1))

        # update the state of the vehicle using it's current pose and optimal control
        current_state = update_state(current_state, opt_controls[:, 0], dt)
        # update the initial conditions
        initial_states, initial_controls = update_initial_conditions(opt_states, opt_controls)

        mpc_iter += mpc_iter

    main_loop_time = time()
    ss_error = ca.norm_2(current_state - target_state)

    print("\n\n")
    if not goal_reached:
        print("Simulation timeout")
    print("Total time: ", main_loop_time - main_loop)
    print("avg iteration time: ", np.array(times).mean() * 1000, "ms")
    print("final error: ", ss_error)

    # show results
    simulate(
        cat_states,
        cat_controls,
        times,
        dt,
        N,
        np.asarray(initial_pose + target_pose),
        save=False,
    )


def main():

    # Horizon specs
    N = 10  # number of look ahead steps

    # Initial state
    x_init = 0
    y_init = 0
    theta_init = 0

    # Target state
    x_target = 10
    y_target = 10
    theta_target = -ca.pi / 2

    theta_target = theta_target % (2 * ca.pi)

    initial_pose = [x_init, y_init, theta_init]
    target_pose = [x_target, y_target, theta_target]

    run(initial_pose, target_pose, N)


if __name__ == "__main__":

    main()
