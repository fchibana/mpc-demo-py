import math
import sys
from enum import auto
from enum import Enum

import casadi as ca


class ProblemType(Enum):
    PS = auto()  # point stabilization
    TT = auto()  # trajectory tracking


class Solver:
    def __init__(self, type="point_stabilization") -> None:

        # number of states (x, y, yaw)
        self._n_states = 3
        # number of controls (v, omega)
        self._n_controls = 2

        # look-ahead number
        self._n_la = 10
        self._dt = 0.2

        if type == "point_stabilization":
            self._type = ProblemType.PS

            self._weight_x = 1
            self._weight_y = 1
            self._weight_yaw = 0.01
            self._weight_v = 0.5
            self._weight_omega = 0.05
        else:
            # trajectory tracking
            self._type = ProblemType.TT

            self._weight_x = 0.99
            self._weight_y = 0.99
            self._weight_yaw = 0.1
            self._weight_v = 0.05
            self._weight_omega = 0.05

        self.reset()
        self._solver = self._get_nlp_solver()
        self._bounds = self._get_bounds()

    def reset(self) -> None:
        self._opt_states = None
        self._opt_controls = None

    def solve(self, x0, p) -> None:

        solution = self._solver(
            x0=x0,
            p=p,
            lbx=self._bounds["lbx"],
            ubx=self._bounds["ubx"],
            lbg=self._bounds["lbg"],
            ubg=self._bounds["ubg"],
        )

        self._opt_states = ca.reshape(
            solution["x"][: self._n_states * (self._n_la + 1)],
            self._n_states,
            self._n_la + 1,
        )
        self._opt_controls = ca.reshape(
            solution["x"][self._n_states * (self._n_la + 1) :],
            self._n_controls,
            self._n_la,
        )

        # std::tuple<StMat, ConMat> mpc_control(
        # StVec st_i, StMat X_i, ConMat U_i, StMat X_ref) {
        #   // Tested 210405

        #   using namespace casadi;

        #   // Get solution to NLP problem
        #   Function solver{npl_solution()};
        #   std::map<std::string, DM> mpc_args{get_mpc_args(st_i, X_i, U_i, X_ref)};
        #   std::vector<double> z_opt(solver(mpc_args).at("x"));

        #   int j = 0;
        #   StMat oX = StMat::Zero();
        #   for (int k = 0; k < N_la + 1; ++k) {
        #     for (int s = 0; s < n_states; ++s) {
        #       oX(s, k) = z_opt.at(j);
        #       j++;
        #     }
        #   }

        #   // Optimized control matrix
        #   ConMat oU = ConMat::Zero();
        #   for (int k = 0; k < N_la; ++k) {
        #     for (int s = 0; s < n_controls; ++s) {
        #       oU(s, k) = z_opt.at(j);
        #       j++;
        #     }
        #   }

        #   casadi_assert(j == n_states * (N_la + 1) + n_controls * N_la, "");

        #   return {oX, oU};
        # }

        # x = np.zeros((self._n_states, self._n_la + 1))
        # u = np.zeros((self._n_controls, self._n_la))

        # x_ref = np.zeros((self._n_states, self._n_la + 1))
        # # x_ref[:, 0] = x
        # x_ref[1, 1] = 1.0

        # args = self._get_nlp_args(x, u, x_ref)
        # print(args)

        # solver = self._get_nlp_solver()
        # res = solver(
        #     x0=args["x0"],
        #     lbx=args["lbx"],
        #     ubx=args["ubx"],
        #     lbg=args["lbg"],
        #     ubg=args["ubg"],
        #     p=args["p"],
        # )
        # print(res)

    # def _get_nlp_solver(self) -> ca.Function:
    def _get_nlp_solver(self) -> ca.Function:

        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym("X", self._n_states, self._n_la + 1)
        # matrix containing all controls over all time steps (each column is an control vector)
        U = ca.SX.sym("U", self._n_controls, self._n_la)

        # parameters vector
        if self._type is ProblemType.PS:
            # column vector for storing initial state and target state
            P = ca.SX.sym("P", self._n_states + self._n_states)
        elif self._type is ProblemType.TT:
            # robot's initial pose + reference poses along the path
            P = ca.SX.sym("P", self._n_states * (self._n_la + 1))
        else:
            print("error")
            sys.exit(1)

        nlp_prob = {
            "x": ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
            "p": P,
            "f": self._cost_function(X, U, P),
            "g": self._constraint_equations(X, U, P),
        }

        # solver options
        solver_options = {
            "ipopt": {
                "max_iter": 2000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }

        return ca.nlpsol("solver", "ipopt", nlp_prob, solver_options)

    def _cost_function(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """Compute the cost function in terms of the symbolic variable.

        Note that the cost function does not depend on x[:, 0]. This vector is
        determined by the
        initial constraint x[:, 0] = st_ini, the current pose of the vehicle.

        Also note that the reference control is zero. This means we want to solve the
        optimization
        problem with the lowest speed possible.

        Args:
            x (ca.SX): symbolic state matrix
            u (ca.SX): symbolic control matrix
            p (ca.SX): symbolic paramters vector

        Returns:
            ca.SX: symbolic scalar representing the cost function
        """
        # state and control weights matrices
        q = ca.diagcat(self._weight_x, self._weight_y, self._weight_yaw)
        r = ca.diagcat(self._weight_v, self._weight_omega)

        cost_fn = 0

        if self._type is ProblemType.PS:
            for k in range(self._n_la):
                st = x[:, k]
                con = u[:, k]
                st_ref = p[self._n_states :]

                state_loss = (st - st_ref).T @ q @ (st - st_ref)
                control_loss = con.T @ r @ con
                cost_fn += state_loss + control_loss

        elif self._type is ProblemType.TT:
            for k in range(self._n_la + 1):
                st = x[:, k]
                st_ref = p[k * self._n_states : (k + 1) * self._n_states]
                assert st.shape == st_ref.shape

                # state cost function: weighted squared difference between estimated and
                # reference poses
                if k != 0:
                    cost_fn += (st - st_ref).T @ q @ (st - st_ref)

                # control cost function: weighted weighted norm of the estimated control
                if k < self._n_la:
                    con = u[:, k]
                    cost_fn = con.T @ r @ con
        else:
            print("error")
            sys.exit(1)

        return cost_fn

    def _constraint_equations(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """Compute the constraint equations in terms of the symbolic variables.

        In total there are n_states * (n_la + 1) constraints, divided into two types:
        1. initial constraint: x[:, 0] = st_ini, the current pose of the vehicle;
        2. kinematic constraints: x[:, i+1] = f(x[:, i], u[:, i]), for i = 0, ...,
        n_la-1, where
           f(.) is the discretized evolution equation that describes the pose at the
           next step given
           the current pose and velocity control.

        Args:
            x (ca.SX): symbolic state matrix
            u (ca.SX): symbolic control matrix
            p (ca.SX): symbolic paramters vector

        Returns:
            ca.SX: symbolic vector representing the constraint equations
        """

        # initial constraint
        g = x[:, 0] - p[: self._n_states]

        robot = RobotModel()

        # kinematic constraints
        for k in range(self._n_la):
            st = x[:, k]
            con = u[:, k]
            st_next = x[:, k + 1]
            st_next_rk = robot.update_state(st, con, dt=self._dt)
            g = ca.vertcat(g, st_next - st_next_rk)
        return g

    def _get_bounds(self):
        # initialize to zero
        lbx = ca.DM.zeros(
            (self._n_states * (self._n_la + 1) + self._n_controls * self._n_la, 1)
        )
        ubx = ca.DM.zeros(
            (self._n_states * (self._n_la + 1) + self._n_controls * self._n_la, 1)
        )

        # lower bounds for x, y, and yaw, respectively
        lbx[0 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf
        lbx[1 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf
        lbx[2 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf

        # upper bounds for x, y, and yaw, respectively
        ubx[0 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf
        ubx[1 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf
        ubx[2 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf

        # Control bounds
        v_max = 0.6
        v_min = -v_max
        omega_max = ca.pi / 4
        omega_min = -omega_max

        lbx[self._n_states * (self._n_la + 1) :: self._n_controls] = v_min
        lbx[self._n_states * (self._n_la + 1) + 1 :: self._n_controls] = omega_min
        ubx[self._n_states * (self._n_la + 1) :: self._n_controls] = v_max
        ubx[self._n_states * (self._n_la + 1) + 1 :: self._n_controls] = omega_max

        args = {
            "lbg": ca.DM.zeros(
                (self._n_states * (self._n_la + 1), 1)
            ),  # constraints lower bound
            "ubg": ca.DM.zeros(
                (self._n_states * (self._n_la + 1), 1)
            ),  # constraints upper bound
            "lbx": lbx,
            "ubx": ubx,
        }

        return args

    def _get_nlp_args(self, x, u, x_ref):
        """
        Compute dictionary with MPC arguments used by the NLP solver.

        X_ref(:, 0) = st_i is not an optimization variable. st_i != X(:, 0)
        It is needed to compute P

        Args:
            X (_type_): _description_
            U (_type_): _description_
            X_ref (_type_): _description_
        """

        # initial conditions for decision variables (states + controls)
        x0 = ca.vertcat(
            ca.reshape(x, self._n_states * (self._n_la + 1), 1),
            ca.reshape(u, self._n_controls * self._n_la, 1),
        )

        # Initialize vectors with lower and upper bound for the optimization variables
        lbx = ca.DM.zeros(
            (self._n_states * (self._n_la + 1) + self._n_controls * self._n_la, 1)
        )
        ubx = ca.DM.zeros(
            (self._n_states * (self._n_la + 1) + self._n_controls * self._n_la, 1)
        )

        # lower bounds for the 2d pose (x, y, yaw)
        lbx[0 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf
        lbx[1 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf
        lbx[2 : self._n_states * (self._n_la + 1) : self._n_states] = -ca.inf

        # upper bounds for the 2d pose
        ubx[0 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf
        ubx[1 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf
        ubx[2 : self._n_states * (self._n_la + 1) : self._n_states] = ca.inf

        # Define some kinematic constants
        # TODO: define these some place else
        lin_vel_max = 2.0  # Max linear velocity [m/s]
        lin_vel_min = -lin_vel_max  # Min linear velocity [m/s]
        ang_vel_max = math.pi / 4.0  # Max angular velocity [rad/s]
        ang_vel_min = -ang_vel_max  # Min angular velocity [rad/s]
        v_min = lin_vel_min
        v_max = lin_vel_max
        omega_min = ang_vel_min
        omega_max = ang_vel_max

        # lower bounds for the controls (v, omega)
        lbx[self._n_states * (self._n_la + 1) :: self._n_controls] = v_min
        lbx[self._n_states * (self._n_la + 1) + 1 :: self._n_controls] = omega_min

        # upper bounds for the controls
        ubx[self._n_states * (self._n_la + 1) :: self._n_controls] = v_max
        ubx[self._n_states * (self._n_la + 1) + 1 :: self._n_controls] = omega_max

        p = ca.reshape(x_ref, self._n_states * (self._n_la + 1), 1)

        args = {
            "x0": x0,
            "lbx": lbx,
            "ubx": ubx,
            "p": p,
            "lbg": ca.DM.zeros((self._n_states * (self._n_la + 1), 1)),
            "ubg": ca.DM.zeros((self._n_states * (self._n_la + 1), 1)),
        }

        return args


class RobotModel:
    def __init__(self) -> None:
        self._type = "differential_drive"

    def update_state(self, st, con, dt):
        state = ca.vertcat(ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("theta"))
        control = ca.vertcat(ca.SX.sym("v"), ca.SX.sym("omega"))
        f = self._differential_drive(state, control)

        k1 = f(st, con)
        k2 = f(st + k1 * dt / 2.0, con)
        k3 = f(st + k2 * dt / 2.0, con)
        k4 = f(st + k3 * dt, con)

        st_next = st + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
        return st_next

    def _differential_drive(self, st, con):
        yaw = st[2]
        v = con[0]
        omega = con[1]

        # dx/dt, dy/dt, dyaw/dt
        rhs = ca.vertcat(v * ca.cos(yaw), v * ca.sin(yaw), omega)
        return ca.Function("f", [st, con], [rhs])

    def _mecanum_whell(self, states, controls):
        """Mecanum wheel transfer function which can be found here:
        https://www.researchgate.net/publication/334319114_Model_Predictive_Control_for_a_Mecanum-wheeled_robot_in_Dynamical_Environments

        Returns:
            [type]: [description]
        """

        theta = states[2]

        # Robot specs
        # rob_diam = 0.3  # diameter of the robot
        wheel_radius = 1  # wheel radius
        Lx = 0.3  # L in J Matrix (half robot x-axis length)
        Ly = 0.3  # l in J Matrix (half robot y-axis length)

        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
        rot_3d_z = ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
            ca.horzcat(ca.sin(theta), ca.cos(theta), 0),
            ca.horzcat(0, 0, 1),
        )
        J = (wheel_radius / 4) * ca.DM(
            [
                [1, 1, 1, 1],
                [-1, 1, 1, -1],
                [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)],
            ]
        )
        # RHS = states + J @ controls * step_horizon  # Euler discretization
        RHS = rot_3d_z @ J @ controls
        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        f = ca.Function("f", [states, controls], [RHS])

        return f
