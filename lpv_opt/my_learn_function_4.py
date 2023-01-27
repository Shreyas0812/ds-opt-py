import numpy as np
from casadi import *


# reconstruction, ready to
def my_learn_function(Data):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]
    P = SX.sym('y', 3, 3)
    eigs = eig_symbolic(P)
    g = vertcat(eigs[0] - 0.1, eigs[1] - 0.1, eigs[2] - 0.1, P[0,1] - P[1,0], P[0,2] - P[2,0], P[1,2] - P[2,1])
    nlp = {'x': P.reshape((1, 9)), 'f': object_function(P, x, xd, 0.0001), 'g': g}
    S = nlpsol('S', 'ipopt', nlp)
    r = S(x0=np.eye(3),
          lbg=[0.00, 0.00, 0.00, 0, 0, 0], ubg=[inf, inf, inf, 0, 0, 0])
    x_opt = r['x']
    print('x_opt: ', x_opt)

    return x_opt

def object_function(P, x, xd, w):
    J_total = 0
    for i in np.arange(len(x[0])):
        dlyap_dx, dlyap_dt = compute_Energy_Single(x[:, i], xd[:, i], P)
        norm_vx = sqrt(dlyap_dx[0]**2 + dlyap_dx[1]**2 + dlyap_dx[2] ** 2)
        norm_xd = sqrt(xd[:, i][0]**2 + xd[:, i][1]**2 + xd[:, i][2]**2)
        J = if_else(logic_or(norm_xd == 0, norm_vx == 0), 0, dlyap_dt / (norm_vx * norm_xd))
        J_total += if_else(dlyap_dt < 0, -w * J**2, J ** 2)

    return J_total


def compute_Energy_Single(x, xd, P):
    d_lyap = (P + P.T) @ x  # 3x3 @ 3x1 = 3x1 (derivative of lyap)
    V_dot = xd[0] * d_lyap[0] + xd[1] * d_lyap[1] + xd[2] * d_lyap[2]
    return d_lyap, V_dot
