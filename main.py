"""
System model: differential drive robot
Problem: point stabilization
"""

from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from mpc_demo.simulation_code import simulate


# def shift_timestep(step_horizon, t0, state_init, u, f):
#     f_value = f(state_init, u[:, 0])
#     next_state = ca.DM.full(state_init + (step_horizon * f_value))
#
#     t0 = t0 + step_horizon
#     u0 = ca.horzcat(
#         u[:, 1:],
#         ca.reshape(u[:, -1], -1, 1)
#     )
#
#     return t0, next_state, u0


def shift_time_step_rk4(step_horizon, t0, st, u, f):
    """
    Advance the time, state, and control vectores to next step
    """
    # Kinematic equation (RK4)
    con = u[:, 0]
    st_next = rk4(f, st, con, step_horizon)

    t0 = t0 + step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

    return t0, st_next, u0


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
    f = ca.Function('f', [states, controls], [rhs])

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

        # state_loss = (st - P[n_states:]).T @ Q @ (st - P[n_states:])
        # control_loss = con.T @ R @ con
        # cost_fn += state_loss + control_loss

        st_next = X[:, k+1]
        st_next_RK4 = rk4(f, st, con, step_horizon)
        g = ca.vertcat(g, st_next - st_next_RK4)

    return g


def main():

    #
    # Input vars
    #

    # Loss function weights
    # State loss function
    # Q_x = 1
    # Q_y = 5
    # Q_theta = 0.1
    # Q_x = 1
    # Q_y = 10
    # Q_theta = 0.01

    # Control loss function
    # R_v = 0.5
    # R_omega = 0.05
    # R_v = 0.5
    # R_omega = 0.05

    # Horizon specs
    step_horizon = 0.2  # time between steps in seconds
    N = 10              # number of look ahead steps

    # Simulation specs
    sim_time = 200      # simulation time

    # Initial state
    x_init = 0
    y_init = 0
    theta_init = 0
    # Target state
    x_target = 10
    y_target = 10
    theta_target = - pi / 2
    theta_target = theta_target % (2*pi)

    # Control bounds
    # TODO: use turtlebot specs
    v_max = 0.6
    v_min = - v_max
    omega_max = pi / 4
    omega_min = - omega_max

    #
    # Symbolic
    #

    # state symbolic variables
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    n_states = states.numel()

    # control symbolic variables
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.numel()

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = ca.SX.sym('X', n_states, N + 1)
    # matrix containing all control actions over all time steps (each column is an action vector)
    U = ca.SX.sym('U', n_controls, N)

    # column vector for storing initial state and target state
    P = ca.SX.sym('P', n_states + n_states)

    # state weights matrix (Q_X, Q_Y, Q_THETA)
    # Q = ca.diagcat(Q_x, Q_y, Q_theta)
    # controls weights matrix
    # R = ca.diagcat(R_v, R_omega)

    f = diff_drive(states, controls)

    # cost_fn = 0  # cost function
    # g = X[:, 0] - P[:n_states]  # constraints in the equation

    # runge kutta
    # for k in range(N):

    #     st = X[:, k]
    #     con = U[:, k]

    #     state_loss = (st - P[n_states:]).T @ Q @ (st - P[n_states:])
    #     control_loss = con.T @ R @ con
    #     cost_fn += state_loss + control_loss

    #     st_next = X[:, k+1]
    #     st_next_RK4 = rk4(f, st, con, step_horizon)
    #     g = ca.vertcat(g, st_next - st_next_RK4)

    # Vector with optimization variables
    # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    opt_vars = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
    nlp_prob = {
        # 'f': cost_fn,
        "f": cost_function(n_states, N, X, U, P),
        'x': opt_vars,
        # 'g': g,
        "g": constraint_equations(n_states, N, step_horizon, X, U, P, f),
        'p': P
    }

    opts = {'ipopt': {'max_iter': 2000,
                      'print_level': 0,
                      'acceptable_tol': 1e-8,
                      'acceptable_obj_change_tol': 1e-6},
            'print_time': 0}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Define lower and upper bounds for optimization variables
    # and constraint functions

    lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
    ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

    lbx[0: n_states*(N+1): n_states] = -ca.inf     # x lower bound
    lbx[1: n_states*(N+1): n_states] = -ca.inf     # y lower bound
    lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

    ubx[0: n_states*(N+1): n_states] = ca.inf      # y upper bound
    ubx[1: n_states*(N+1): n_states] = ca.inf      # y upper bound
    ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

    lbx[n_states * (N + 1)::n_controls] = v_min
    lbx[n_states * (N + 1) + 1::n_controls] = omega_min
    ubx[n_states * (N + 1)::n_controls] = v_max
    ubx[n_states * (N + 1) + 1::n_controls] = omega_max

    args = {'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx}

    # Set up main loop

    t0 = 0
    current_state = ca.DM([x_init, y_init, theta_init])        # initial state
    target_state = ca.DM([x_target, y_target, theta_target])  # target state

    # t = ca.DM(t0)

    u0 = ca.DM.zeros((n_controls, N))          # initial control
    X0 = ca.repmat(current_state, 1, N+1)         # initial state full

    mpc_iter = 0
    cat_states = DM2Arr(X0)
    cat_controls = DM2Arr(u0[:, 0])
    times = np.array([[0]])

    main_loop = time()  # return time in sec

    while (ca.norm_2(current_state - target_state) > 1e-1) and (mpc_iter * step_horizon < sim_time):

        t1 = time()

        # current and target states
        args['p'] = ca.vertcat(current_state, target_state)
        # optimization variable current state
        args['x0'] = ca.vertcat(ca.reshape(X0, n_states*(N+1), 1), ca.reshape(u0, n_controls*N, 1))

        sol = solver(x0=args['x0'],
                     lbx=args['lbx'],
                     ubx=args['ubx'],
                     lbg=args['lbg'],
                     ubg=args['ubg'],
                     p=args['p'])
        
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((cat_states, DM2Arr(X0)))
        cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
        # t = np.vstack((t, t0))
        times = np.vstack((times, time()-t1))

        t0, current_state, u0 = shift_time_step_rk4(step_horizon, t0, current_state, u, f)

        X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))

        # xx ...
        # t2 = time()
        # print(mpc_iter)
        # print(t2-t1)
        # times = np.vstack((times, t2-t1))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(current_state - target_state)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, step_horizon, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)

###############################################################################


if __name__ == '__main__':

    main()
