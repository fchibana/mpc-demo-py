"""
System model: differential drive robot
Problem: point stabilization
"""

from time import time

import casadi as ca
import numpy as np
# from casadi import sin, cos, pi

from mpc_demo.simulation_code import simulate
from mpc_demo.casadi_solver import RobotModel


def DM2Arr(dm):
    return np.array(dm.full())


def get_nlp_solver(N: int, dt: float, n_states: int = 3, n_controls: int = 2) -> ca.Function:
    """Set up NLP problem symbolically and returns the solver function.

    Args:
        N (int): number of lookahead steps
        dt (float): time step

    Returns:
        casadi.Function: a function that takes in the nlp args and returns the solution
    """

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = ca.SX.sym("X", n_states, N + 1)
    # matrix containing all control actions over all time steps (each column is an action vector)
    U = ca.SX.sym("U", n_controls, N)
    # column vector for storing initial state and target state
    P = ca.SX.sym("P", n_states + n_states)

    nlp_prob = {
        "x": ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        "f": cost_function(X, U, P),
        "g": constraint_equations(X, U, P, dt),
        "p": P,
    }

    solver_options = {
        "ipopt": {
            "max_iter": 2000,
            "print_level": 0,
            "acceptable_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-6,
        },
        "print_time": 0,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp_prob, solver_options)

    return solver


def cost_function(X, U, P):

    Q_x = 1
    Q_y = 1
    Q_theta = 0.01
    R_v = 0.5
    R_omega = 0.05

    Q = ca.diagcat(Q_x, Q_y, Q_theta)
    R = ca.diagcat(R_v, R_omega)

    n_states = X.shape[0]
    N = X.shape[1] - 1

    cost_fn = 0
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_ref = P[n_states:]

        state_loss = (st - st_ref).T @ Q @ (st - st_ref)
        control_loss = con.T @ R @ con
        cost_fn += state_loss + control_loss

    return cost_fn


def constraint_equations(X, U, P, dt):

    n_states = X.shape[0]
    N = X.shape[1] - 1

    g = X[:, 0] - P[:n_states]  # constraints in the equation

    robot = RobotModel()

    # runge kutta
    for k in range(N):
        st = X[:, k]
        con = U[:, k]

        st_next = X[:, k + 1]
        st_next_RK4 = robot.update_state(st, con, dt)
        g = ca.vertcat(g, st_next - st_next_RK4)

    return g


def initialize_nlp_args(N: int, n_states: int = 3, n_controls: int = 2):
    """
    Define lower and upper bounds for optimization variables and constraint functions.

    Args:
        N (int): _description_
    """

    # initialize to zero
    lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
    ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

    # lower bounds for x, y, and yaw, respectively
    lbx[0: n_states * (N + 1): n_states] = -ca.inf
    lbx[1: n_states * (N + 1): n_states] = -ca.inf
    lbx[2: n_states * (N + 1): n_states] = -ca.inf

    # upper bounds for x, y, and yaw, respectively
    ubx[0: n_states * (N + 1): n_states] = ca.inf
    ubx[1: n_states * (N + 1): n_states] = ca.inf
    ubx[2: n_states * (N + 1): n_states] = ca.inf

    # Control bounds
    v_max = 0.6
    v_min = -v_max
    omega_max = ca.pi / 4
    omega_min = -omega_max

    lbx[n_states * (N + 1):: n_controls] = v_min
    lbx[n_states * (N + 1) + 1:: n_controls] = omega_min
    ubx[n_states * (N + 1):: n_controls] = v_max
    ubx[n_states * (N + 1) + 1:: n_controls] = omega_max

    args = {
        "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
        "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
        "lbx": lbx,
        "ubx": ubx,
    }

    return args


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
    cat_states = DM2Arr(initial_states)
    cat_controls = DM2Arr(initial_controls[:, 0])
    times = np.array([[0]])

    # simulation state
    mpc_iter = 0

    nlp_solver = get_nlp_solver(N, dt)
    nlp_args = initialize_nlp_args(N, n_states, n_controls)

    # start a chronometer to time the whole loop
    main_loop = time()  # return time in sec

    goal_reached = False

    while mpc_iter * dt < max_sim_time:
        if is_goal_reached(current_state, target_state):
            goal_reached = True
            break

        # start a chronometer to time one iteration
        t1 = time()

        # update the dictionary with the initial values for the decision variables
        nlp_args["x0"] = ca.vertcat(
            ca.reshape(initial_states, n_states * (N + 1), 1),
            ca.reshape(initial_controls, n_controls * N, 1)
        )
        # update dictionary with the vector of parameters
        nlp_args["p"] = ca.vertcat(current_state, target_state)

        solution = nlp_solver(
            x0=nlp_args["x0"],
            p=nlp_args["p"],
            lbx=nlp_args["lbx"],
            ubx=nlp_args["ubx"],
            lbg=nlp_args["lbg"],
            ubg=nlp_args["ubg"],
        )

        # retrieve optimal states and controls from the solution
        opt_states = ca.reshape(solution["x"][: n_states * (N + 1)], n_states, N + 1)
        opt_controls = ca.reshape(solution["x"][n_states * (N + 1):], n_controls, N)

        # update history variables (for simulation)
        cat_states = np.dstack((cat_states, DM2Arr(opt_states)))
        cat_controls = np.vstack((cat_controls, DM2Arr(opt_controls[:, 0])))
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
