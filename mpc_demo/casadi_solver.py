import math

import casadi as ca


# from mpc_demo.simulation import N_STATES, TIME_STEP
# from mpc_demo.simulation import N_CONTROLS
# from mpc_demo.simulation import N_LA


class Solver:
    def __init__(self) -> None:

        # number of states (x, y, yaw)
        self._n_states = 3
        # number of controls (v, omega)
        self._n_controls = 2

        # look-ahead number
        self._n_la = 1
        self._dt = 0.2

        # cost function weights
        self._weight_x = 0.99
        self._weight_y = 0.99
        self._weight_yaw = 0.1
        self._weight_v = 0.05
        self._weight_omega = 0.05

        # Q_x = 0.99
        # Q_y = 0.99
        # Q_yaw = 0.1
        # R_v = 0.05
        # R_omega = 0.05

    def solve(self):
        # std::tuple<StMat, ConMat> mpc_control(StVec st_i, StMat X_i, ConMat U_i, StMat X_ref) {
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

        solver = self._get_nlp_solver()
        # args = self._get_nlp_args()

    def _get_nlp_solver(self) -> ca.Function:
        """Set up NLP problem symbolically and returns the solver object.

        Returns:
            _type_: _description_
        """
        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', self._n_states, self._n_la + 1)
        # matrix containing all control actions over all time steps (each column is an control vector)
        U = ca.SX.sym('U', self._n_controls, self._n_la)
        # parameters vector (robot's initial pose + reference poses along the path)
        P = ca.SX.sym('P', self._n_states * (self._n_la + 1))

        nlp_prob = {
            'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
            "p": P,
            "f": self._cost_function(X, U, P),
            "g": self._constraint_equations(X, U, P)
        }
        print(nlp_prob)
        # solver options
        opts = {'ipopt': {'max_iter': 2000,
                          'print_level': 0,
                          'acceptable_tol': 1e-8,
                          'acceptable_obj_change_tol': 1e-6},
                'print_time': 0}

        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def _cost_function(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """ Compute the cost function in terms of the symbolic variable.

        Note that the cost function does not depend on x[:, 0]. This vector is determined by the
        initial constraint x[:, 0] = st_ini, the current pose of the vehicle.

        Also note that the reference control is zero. This means we want to solve the optimization
        problem with the lowest speed possible.

        Args:
            x (ca.SX): symbolic state matrix
            u (ca.SX): symbolic control matrix
            p (ca.SX): symbolic paramters vector

        Returns:
            ca.SX: symbolic scalar representing the cost function
        """
        # state and control weights matrice
        q = ca.diagcat(self._weight_x, self._weight_y, self._weight_yaw)
        # controls weights matrix
        r = ca.diagcat(self._weight_v, self._weight_omega)

        cost_fn = 0

        for k in range(self._n_la + 1):
            st = x[:, k]
            st_ref = p[k * self._n_states: (k + 1) * self._n_states]
            assert st.shape == st_ref.shape

            # state cost function: weighted squared difference between estimated and reference poses
            if k != 0:
                cost_fn += (st - st_ref).T @ q @ (st - st_ref)

            # control cost function: weighted weighted norm of the estimated control
            if k < self._n_la:
                con = u[:, k]
                cost_fn = con.T @ r @ con

        return cost_fn

    def _constraint_equations(self, x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
        """Compute the constraint equations in terms of the symbolic variables.

        In total there are n_states * (n_la + 1) constraints, divided into two types:
        1. initial constraint: x[:, 0] = st_ini, the current pose of the vehicle;
        2. kinematic constraints: x[:, i+1] = f(x[:, i], u[:, i]), for i = 0, ..., n_la-1, where
           f(.) is the discretized evolution equation that describes the pose at the next step given
           the current pose and velocity control.

        Args:
            x (ca.SX): symbolic state matrix
            u (ca.SX): symbolic control matrix
            p (ca.SX): symbolic paramters vector

        Returns:
            ca.SX: symbolic vector representing the constraint equations
        """
        # initial constraint
        g = x[:, 0] - p[:self._n_states]

        # # kinematic constraints
        for k in range(self._n_la):
            st_next = x[:, k + 1]
            st_next_rk = self._rk4(x[:, k], u[:, k], dt=self._dt)
            g = ca.vertcat(g, st_next - st_next_rk)

        return g

    def _rk4(self, state, control, dt):

        def differential_drive(state, control):
            yaw = state[2]
            v = control[0]
            omega = control[1]

            # dx/dt, dy/dt, dyaw/dt
            kinematics = ca.vertcat(v * math.cos(yaw), v * math.sin(yaw), omega)
            return kinematics

        k1 = differential_drive(state, control)
        k2 = differential_drive(state + k1 * dt / 2.0, control)
        k3 = differential_drive(state + k2 * dt / 2.0, control)
        k4 = differential_drive(state + k3 * dt, control)

        state_next = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
        return state_next


    def _get_nlp_args(self, st_i, X, U, X_ref):
        """
        Compute dictionary with MPC arguments used by the NLP solver.

        st_i is not an optimization variable. st_i != X(:, 0)
        It is needed to compute P   

        Args:
            st_i (_type_): _description_
            X (_type_): _description_
            U (_type_): _description_
            X_ref (_type_): _description_
        """

        # Initialize vectors with lower and upper bound fro the optimization variables
        lbx = ca.DM.zeros((self._n_states * (self._n_la+1) + self._n_controls * self._n_la, 1))
        ubx = ca.DM.zeros((self._n_states * (self._n_la + 1) + self._n_controls*self._n_la, 1))

        # lower bounds for the 2d pose (x, y, yaw)
        lbx[0: self._n_states*(self._n_la+1): self._n_states] = -ca.inf
        lbx[1: self._n_states*(self._n_la+1): self._n_states] = -ca.inf
        lbx[2: self._n_states*(self._n_la+1): self._n_states] = -ca.inf

        # upper bounds for the 2d pose
        ubx[0: self._n_states*(self._n_la+1): self._n_states] = ca.inf
        ubx[1: self._n_states*(self._n_la+1): self._n_states] = ca.inf
        ubx[2: self._n_states*(self._n_la+1): self._n_states] = ca.inf

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
        lbx[self._n_states*(self._n_la+1):: self._n_controls] = v_min
        lbx[self._n_states*(self._n_la+1) + 1:: self._n_controls] = omega_min

        # upper bounds for the controls
        ubx[self._n_states*(self._n_la+1):: self._n_controls] = v_max
        ubx[self._n_states*(self._n_la+1) + 1:: self._n_controls] = omega_max

        # u0 = ca.DM.zeros((n_controls, N))  # initial control
        # X0 = ca.repmat(ST_I, 1, N+1)         # initial state full
        #     args['x0'] = ca.vertcat(
        #         ca.reshape(X0, n_states*(N+1), 1),
        #         ca.reshape(u0, n_controls*N, 1)
        #     )

        #   DM x0; // opt vars
        #   for (int k = 0; k < N_la + 1; ++k) {
        #     for (int s = 0; s < n_states; ++s) {
        #       x0 = vertcat(x0, X(s, k));
        #     }
        #   }
        #   for (int k = 0; k < N_la; ++k) {
        #     for (int s = 0; s < n_controls; ++s) {
        #       x0 = vertcat(x0, U(s, k));
        #     }
        #   }

        #   DM p; // parameters (st_ini, X_ref)
        #   for (int s = 0; s < n_states; ++s) {
        # //    p = vertcat(p, X(s, 0));
        #     p = vertcat(p, st_i(s));
        #   }
        #   for (int k = 1; k < N_la + 1; ++k) {
        #     for (int s = 0; s < n_states; ++s) {
        #       p = vertcat(p, X_ref(s, k));
        #     }
        #   }

        #   std::map<std::string, DM> args;
        #   args["x0"] = x0;
        #   args["lbx"] = lbx;
        #   args["ubx"] = ubx;
        #   args["p"] = p;
        #   args["lbg"] = DM::zeros(n_states * (N_la + 1), 1);      // constraints lower bound
        #   args["ubg"] = DM::zeros(n_states * (N_la + 1), 1);      // constraints upper bound

        #   return args;
        # }

# void test_get_mpc_args() {
#   StVec st_i(1, 2, 3);

#   StMat X;
#   X << 1, 4, 7, 10,
#       2, 5, 8, 11,
#       3, 6, 9, 12;
#   ConMat U;
#   U << 13, 15, 17,
#       14, 16, 18;
#   StMat X_ref;
#   X_ref << 91, 94, 97, 100,
#       92, 95, 98, 101,
#       93, 96, 99, 102;

#   std::map<std::string, casadi::DM> args{get_mpc_args(st_i, X, U, X_ref)};

#   std::cerr << "x0: " << args["x0"] << '\n';
#   std::cerr << "lbx: " << args["lbx"] << '\n';
#   std::cerr << "ubx: " << args["ubx"] << '\n';
#   std::cerr << "lbg: " << args["lbg"] << '\n';
#   std::cerr << "ubg: " << args["ubg"] << '\n';
#   std::cerr << "p: " << args["p"] << '\n';
# }

# std::tuple<StMat, ConMat> mpc_control(StVec st_i, StMat X_i, ConMat U_i, StMat X_ref) {
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
