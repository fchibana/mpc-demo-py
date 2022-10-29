import casadi as ca

from mpc_demo.casadi_solver import Solver


n_states = 3
n_la = 1

# X = ca.SX.sym("X", n_states, n_la + 1)

# print(X)

# P = ca.SX.sym("P", n_states * (n_la + 1))
# print(P)

# print(X[:, 0])
# print(P[: n_states])
# print(X[:, 0] - P[: n_states])

solver = Solver()

# solver._get_nlp_solver()

# X = ca.SX.sym("X", n_states, n_la + 1)
# U = ca.SX.sym("U", 2, n_la)
# P = ca.SX.sym("P", n_states * (n_la + 1))

# solver._cost_function(X, U, P)

# solver._constraint_equations(X, U, P)

solver.solve()
