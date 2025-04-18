from sbamp.ds_opt_py.lpv_opt.posterior_probs_gmm import posterior_probs_gmm
import numpy as np
import cvxpy as cp
from sbamp.ds_opt_py.phys_gmm_python.fit_gmm import fig_gmm
from sbamp.ds_opt_py.lpv_opt.my_learn_function import my_learn_function


def optimize_lpv_ds_from_data(Data, attractor, ctr_type, est_options, *args):
    M = len(Data)
    N = len(Data[0])
    M = int(M / 2)

    # Positions and Velocity Trajectories
    Xi_ref = Data[0:M, :]
    Xi_ref_dot = Data[M:, :]


    gmm = fig_gmm(Xi_ref, Xi_ref_dot, est_options)
    
    # Define Optimization Variables
    K = len(gmm.Priors)
    A_c = np.zeros((K, M, M))
    b_c = np.zeros((M, K))

    # should have switch ctr_type here
    if ctr_type == 0:
        helper = 1  # blank for later use
        symm_constr = 0

    P = my_learn_function(Data)
    symm_constr = args[1]

    h_k = posterior_probs_gmm(Xi_ref, gmm, 'norm')

    # Define Constraints and Assign Initial Values
    # 创建一个object 叫decision variable，which makes it
    A_vars = []
    b_vars = []
    Q_vars = []
    constrains = []
    # elif ctr_type == 1:
    #     P = cp.Variable((M, M), symmetric=True)
    #     constrains += [P >> np.eye(M)]

    for k in np.arange(K):
        if symm_constr:
            A_vars.append(cp.Variable((M, M), symmetric=True))
        else:
            A_vars.append(cp.Variable((M, M)))

        if k == 0:
            A_vars[k] = cp.Variable((M, M), symmetric=True)

        if ctr_type != 1:
            if M == 2:
                b_vars.append(cp.Variable((2, 1)))
            else:
                b_vars.append(cp.Variable((3, 1)))
            Q_vars.append(cp.Variable((M, M), symmetric=True))


        epi = 0.001
        epi = epi * -np.eye(M)
        # Define Constraints
        if ctr_type == 0:
            constrains += [A_vars[k].T + A_vars[k] << epi]
            # constrains += [b_vars[k].T == -A_vars[k] @ attractor]
            constrains += [b_vars[k] == -A_vars[k] @ attractor]

        elif ctr_type == 1:
            constrains += [A_vars[k].T @ P + P @ A_vars[k] << epi]
        else:
            constrains += [A_vars[k].T @ P + P @ A_vars[k] == Q_vars[k]]
            constrains += [Q_vars[k] << epi]
            constrains += [b_vars[k] == -A_vars[k] @ attractor]

    # Calculate our estimated velocities caused by each local behavior
    Xi_d_dot_c_raw = []

    for k in np.arange(K):
        h_K = np.repeat(h_k[k, :].reshape(1, len(h_k[0])), M, axis=0)
        if ctr_type == 1:
            f_k = A_vars[k] @ Xi_ref
        else:
            f_k = A_vars[k] @ Xi_ref
            f_k = f_k + b_vars[k]
        Xi_d_dot_c_raw.append(cp.multiply(h_K, f_k))

    # Sum each of the local behaviors to generate the overall behavior at
    # each point
    Xi_dot_error = np.zeros((M, N))
    for k in np.arange(K):
        Xi_dot_error = Xi_dot_error + Xi_d_dot_c_raw[k]
    Xi_dot_error = Xi_dot_error - Xi_ref_dot

    # Defining Objective Function depending on constraints
    if ctr_type == 0:
        Xi_dot_total_error = 0
        for n in np.arange(N):
            Xi_dot_total_error = Xi_dot_total_error + cp.norm(Xi_dot_error[:, n], 2)
        Objective = Xi_dot_total_error
    else:
        Objective = cp.norm(Xi_dot_error, 'fro')**2

    prob = cp.Problem(cp.Minimize(Objective), constrains)

    prob.solve(solver=cp.MOSEK, verbose=True)

    for k in np.arange(K):
        A_c[k] = A_vars[k].value
        if ctr_type != 1:
            if M == 3:
                b_c[:, k] = [b_vars[k][0].value, b_vars[k][1].value, b_vars[k][2].value]
            else:
                b_c[:, k] = [b_vars[k][0].value, b_vars[k][1].value]

    return A_c, b_c, P, gmm


if __name__ == '__main__':
    print(np.sum(np.array([[1, 2, 3],[1, 2, 3]])**2, axis=0))
