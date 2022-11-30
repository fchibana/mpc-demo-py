import argparse
import math
import random

from mpc_demo import Simulation


def main(args: argparse.Namespace):

    n = args.n
    time_step = args.dt

    x_init = args.xini
    y_init = args.yini
    yaw_init = args.yawini
    initial_pose = [x_init, y_init, yaw_init]

    if args.randtarget:
        x_target = random.uniform(-10.0, 10.0)
        y_target = random.uniform(-10.0, 10.0)
        yaw_target = random.uniform(-math.pi, math.pi)
    else:
        x_target = args.xtarget
        y_target = args.ytarget
        yaw_target = args.yawtarget
    target_pose = [x_target, y_target, yaw_target]

    sim = Simulation(n_la=n, dt=time_step)
    sim.run_point_stabilization(initial_pose, target_pose)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        # prog="mpc demo",
        description="simple demo of an mpc-based local planner (controller)"
    )
    parser.add_argument(
        "-n", type=int, help="prediction horizon", required=False, default=10
    )
    parser.add_argument(
        "-dt", type=float, help="time step", required=False, default=0.2
    )

    parser.add_argument(
        "-randtarget",
        type=bool,
        help="use random target pose",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-xini", type=float, help="initial pose x", required=False, default=0.0
    )
    parser.add_argument(
        "-yini", type=float, help="initial pose y", required=False, default=0.0
    )
    parser.add_argument(
        "-yawini", type=float, help="initial pose yaw", required=False, default=0.0
    )

    parser.add_argument(
        "-xtarget", type=float, help="target pose x", required=False, default=10.0
    )
    parser.add_argument(
        "-ytarget", type=float, help="target pose y", required=False, default=10.0
    )
    parser.add_argument(
        "-yawtarget", type=float, help="target pose yaw", required=False, default=0.0
    )

    main(parser.parse_args())
