"""
System model: differential drive robot
Problem: point stabilization
"""

from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from mpc_demo.simulation_code import simulate


def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + k1 * dt / 2, u)
    k3 = f(x + k2 * dt / 2, u)
    k4 = f(x + k3 * dt, u)
    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return x_next


def DM2Arr(dm):
    return np.array(dm.full())


def diff_drive(states, controls):

    theta = states[2]
    v = controls[0]
    omega = controls[1]

    rhs = ca.vertcat(v * cos(theta), v * sin(theta), omega)
    f = ca.Function("f", [states, controls], [rhs])

    return f


# def mecanum_whell(states, controls):
#     """ Mecanum wheel transfer function which can be found here:
#     https://www.researchgate.net/publication/334319114_Model_Predictive_Control_for_a_Mecanum-wheeled_robot_in_Dynamical_Environments


#     Returns:
#         [type]: [description]
#     """

#     # Robot specs
#     # rob_diam = 0.3  # diameter of the robot
#     wheel_radius = 1  # wheel radius
#     Lx = 0.3  # L in J Matrix (half robot x-axis length)
#     Ly = 0.3  # l in J Matrix (half robot y-axis length)

#     # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
#     rot_3d_z = ca.vertcat(ca.horzcat(cos(theta), -sin(theta), 0),
#                           ca.horzcat(sin(theta), cos(theta), 0),
#                           ca.horzcat(0, 0, 1)
#                           )
#    J = (wheel_radius / 4) * ca.DM(
#        [[1, 1, 1, 1],
#        [-1, 1, 1, -1],
#        [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)]])
#     # RHS = states + J @ controls * step_horizon  # Euler discretization
#     RHS = rot_3d_z @ J @ controls
#     # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
#     f = ca.Function('f', [states, controls], [RHS])

#     return f


def get_nlp_solver(N: int, dt: float) -> ca.Function:
    """Set up NLP problem symbolically and returns the solver function.

    Args:
        N (int): number of lookahead steps
        dt (float): time step

    Returns:
        casadi.Function: a function that takes in the nlp args and returns the solution
    """

    # state symbolic variables
    state = ca.vertcat(ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("theta"))
    n_states = state.numel()

    # control symbolic variables
    control = ca.vertcat(ca.SX.sym("v"), ca.SX.sym("omega"))
    n_controls = control.numel()

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = ca.SX.sym("X", n_states, N + 1)
    # matrix containing all control actions over all time steps (each column is an action vector)
    U = ca.SX.sym("U", n_controls, N)
    # column vector for storing initial state and target state
    P = ca.SX.sym("P", n_states + n_states)

    # kinematics
    kinematical_func = diff_drive(state, control)

    nlp_prob = {
        "x": ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        "f": cost_function(n_states, N, X, U, P),
        "g": constraint_equations(n_states, N, dt, X, U, P, kinematical_func),
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


def cost_function(n_states, N, X, U, P):

    Q_x = 1
    Q_y = 1
    Q_theta = 0.01
    R_v = 0.5
    R_omega = 0.05

    Q = ca.diagcat(Q_x, Q_y, Q_theta)
    R = ca.diagcat(R_v, R_omega)

    cost_fn = 0
    for k in range(N):
        st = X[:, k]
        con = U[:, k]

        state_loss = (st - P[n_states:]).T @ Q @ (st - P[n_states:])
        control_loss = con.T @ R @ con
        cost_fn += state_loss + control_loss

    return cost_fn


def constraint_equations(n_states, N, step_horizon, X, U, P, f):
    g = X[:, 0] - P[:n_states]  # constraints in the equation

    # runge kutta
    for k in range(N):
        st = X[:, k]
        con = U[:, k]

        st_next = X[:, k + 1]
        st_next_RK4 = rk4(f, st, con, step_horizon)
        g = ca.vertcat(g, st_next - st_next_RK4)

    return g


def initialize_nlp_args(N: int):
    """
    Define lower and upper bounds for optimization variables and constraint functions.

    Args:
        N (int): _description_
    """

    n_states = 3
    n_controls = 2

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
    omega_max = pi / 4
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


def update_state(f, st, con, dt):
    # TODO: add noise
    st_next = rk4(f, st, con, dt)

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
    X0 = ca.repmat(current_state, 1, N + 1)
    u0 = ca.DM.zeros((n_controls, N))

    # history variables
    # TODO: rename
    cat_states = DM2Arr(X0)
    cat_controls = DM2Arr(u0[:, 0])
    times = np.array([[0]])

    # simulation state
    mpc_iter = 0

    nlp_solver = get_nlp_solver(N, dt)
    nlp_args = initialize_nlp_args(N)

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
            ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1)
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
        u = ca.reshape(solution["x"][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(solution["x"][: n_states * (N + 1)], n_states, N + 1)

        # update history variables (for simulation)
        cat_states = np.dstack((cat_states, DM2Arr(X0)))
        cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
        times = np.vstack((times, time() - t1))

        # update the state of the vehicle using it's current pose and optimal control
        # TODO: should add some noise to this step, since it simulates the actual kinematics

        # TODO: fix
        state = ca.vertcat(ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("theta"))
        control = ca.vertcat(ca.SX.sym("v"), ca.SX.sym("omega"))
        f_diff_drive = diff_drive(state, control)

        current_state = update_state(f_diff_drive, current_state, u[:, 0], dt)
        X0, u0 = update_initial_conditions(X0, u0)

        mpc_iter += mpc_iter
    main_loop_time = time()
    ss_error = ca.norm_2(current_state - target_state)

    print("\n\n")
    if not goal_reached:
        print("Simulations timeout")
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
    theta_target = -pi / 2

    theta_target = theta_target % (2 * pi)

    initial_pose = [x_init, y_init, theta_init]
    target_pose = [x_target, y_target, theta_target]

    run(initial_pose, target_pose, N)


if __name__ == "__main__":

    main()
