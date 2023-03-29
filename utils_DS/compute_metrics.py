import dtw
import numpy as np
from utils_DS.Simulation import Simulation
from Structs_DS import Opt_Sim

def compute_rmse(fun_handle, Xi_ref, Xi_dot_ref):
    Xi_dot_pred = fun_handle(Xi_ref)
    predict_diff = (Xi_dot_pred - Xi_dot_ref) ** 2
    trajectory_RMSE = np.sqrt(np.mean(predict_diff, axis=0))
    rmse = np.mean(trajectory_RMSE)
    print('LPV-DS got prediction RMSE on training set: {}'.format(rmse))
    return rmse


def compute_e_dot(fun_handle, Xi_ref, Xi_dot_ref):
    Xi_dot_pred = fun_handle(Xi_ref).T
    Xi_dot_ref = Xi_dot_ref.T
    e_cum = 0
    M = Xi_ref.shape[1]
    for i in np.arange(M):
        norm_term = np.linalg.norm(Xi_dot_pred[i]) * np.linalg.norm(Xi_dot_ref[i])
        if norm_term < 10 ** (-10):
            e_cum += 0
        else:
            e_cum += np.abs(1 - (Xi_dot_pred[i] @ Xi_dot_ref[i].reshape(-1, 1)) / norm_term)

    print('LPV-DS got prediction e_dot on training set: {}'.format(e_cum/M))
    return e_cum / M


def compute_dtwd(fun_handle, Xi_ref, demo_size, x0_all):
    opt_sim = Opt_Sim()
    opt_sim.dt = 0.005
    opt_sim.i_max = 10000
    opt_sim.tol = 0.001
    opt_sim.plot = 0
    num_traj = x0_all.shape[1]
    x_sim = Simulation(x0_all, fun_handle, opt_sim)
    trajs = np.array(x_sim)
    start_idx = 0
    dist = []
    for i in np.arange(num_traj):
        cur_traj = trajs[:, :, i]
        end_index = start_idx + demo_size[i]
        cur_ref_traj = Xi_ref[:, start_idx:end_index].T
        dist.append(dtw.dtw(cur_traj, cur_ref_traj).distance)
        start_idx = end_index

    mean_dist = np.mean(dist)
    cov_dist = np.std(dist)
    print('LPV-DS got prediction dwtd on training set: {} +- {}'.format(mean_dist, cov_dist))
    return dist


if __name__ == '__main__':
    a = np.ones((10, 3))
    b = np.ones((5, 3))
    b[-1] = [2, 2, 2]
    dist = dtw.dtw(a, b).distance
    print(dist)


