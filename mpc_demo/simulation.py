import math

import matplotlib.pyplot as plt

# import numpy as np

from mpc_demo.spline_planner import SplinePlanner

# Number of states (x, y, yaw)
N_STATES = 3
# Number of controls (v, omega)
N_CONTROLS = 2

# Control bounds
lin_vel_max = 2.0  # Max linear velocity [m/s]
lin_vel_min = -lin_vel_max  # Min linear velocity [m/s]
ang_vel_max = math.pi / 4.0  # Max angular velocity [rad/s]
ang_vel_min = -ang_vel_max  # Min angular velocity [rad/s]


# Horizon and simulation specs
N_LA = 2  # Number of look-ahead steps (waypoints inside the horizon)
TIME_STEP = 0.2  # Time interval between look-ahead steps [s]
LENGTH_STEP = lin_vel_max * TIME_STEP  # Trajectory bin [m]


class Simulation:
    def __init__(self, course_name: str) -> None:

        self.course = Course(course_name)

    def run(self) -> None:
        # StVec target(cx.back(), cy.back(), cyaw.back());
        # target = [self.course.x[-1], self.course.y[-1], self.course.yaw[-1]]

        # double t_i{0.0};
        # StVec st_i{st_ini};
        # ConVec con_i = ConVec::Zero();
        # ConMat U_i = ConMat::Zero();       // initial control matrix
        # StMat X_i;                         // initial states matrix
        # for (int k = 0; k < X_i.cols(); ++k) {
        #     X_i.col(k) = st_i;
        # }
        # t_i = 0.0
        # st_i = [self.course.x[0], self.course.y[0], self.course.yaw[0]]
        # con_i = [0, 0]
        # U_i = np.zeros((N_CONTROLS, N_LA))
        # X_i = np.zeros((N_STATES, N_LA + 1))
        # for i in range(N_LA + 1):
        #     X_i[:, i] = st_i

        # Eigen::Matrix2d vel_bounds;
        # vel_bounds << parameters::lin_vel_min, parameters::lin_vel_max,
        #     parameters::ang_vel_min, parameters::ang_vel_max;
        # double time_step{parameters::time_step};
        # Eigen::DiagonalMatrix<double, 2> r_mat(parameters::control_weight_v,
        #                                         parameters::control_weight_omega);
        # Eigen::DiagonalMatrix<double, 3> q_mat(parameters::state_weight_x,
        #                                         parameters::state_weight_y,
        #                                         parameters::state_weight_yaw);
        # int lookahead_nodes{parameters::number_lookahead_nodes};

        # cyaw = smooth_yaw(cyaw);

        # Vec t{t_i};
        # Vec x{st_i(0)};
        # Vec y{st_i(1)};
        # Vec yaw{st_i(2)};
        # Vec v{0};
        # Vec omega{0};

        # std::vector<std::vector<double>> obst_x, obst_y, obst_z;
        # std::tie(obst_x, obst_y, obst_z) = get_obstacles_plot(obst_poses);

        # int target_ind{calc_nearest_index(st_i, cx, cy, 0)};

        return

    def plot_results(self) -> None:
        # void plot_results(
        # Vec &t, Vec &cx, Vec &cy, Vec &x, Vec &y, Vec &v, Vec &omega) {

        # plt::close();
        # plt::plot(t, v, {{"label", "v [m/s]"}});
        # plt::plot(t, omega, {{"label", "omega [rad/s]"}});
        # plt::xlabel("t [s]");
        # plt::legend();
        # plt::grid(true);
        # plt::show();

        plt.plot(self.course.x, self.course.y, "--k", label="spline")
        plt.legend()
        plt.grid(True)
        plt.show()

        # plt::plot(cx, cy, "--k", {{"label", "spline"}});
        # plt::plot(x, y, "--r", {{"label", "trajectory"}});
        # plt::xlabel("t [s]");
        # plt::axis("equal");
        # plt::legend();
        # plt::grid(true);
        # plt::show();


class Course:
    def __init__(self, name: str, dl=0.1) -> None:
        self._name = name
        self._dl = dl

        self.x = None
        self.y = None
        self.yaw = None

        self.compute_course()

    def compute_course(self):

        if self._name == "straight":
            ax, ay = self.get_straight_course()

        planner = SplinePlanner(ax, ay, ds=self._dl)
        self.x, self.y, self.yaw = planner.get_poses()

    def get_straight_course(self):
        ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return ax, ay
