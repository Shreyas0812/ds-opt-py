from phys_gmm_python.fit_gmm import fig_gmm
import numpy as np
from datasets.load_dataset_DS import load_dataset_DS
from utils_DS.figure_tools.plot_reference_trajectories_DS import plot_reference_trajectories_DS
from phys_gmm_python.Structs import Est_options
from utils_DS.diy_tools.knn_search import knn_search
from gmr import GMM
from phys_gmm_python.utils.plotting.simple_classification_check import plot_result
from Structs_DS import ds_gmms, ds_plot_options
from utils.adjust_Covariances import adjust_Covariances

from lpv_opt.optimized_from_seperate_DS import optimize_lpv_ds_from_data
from lpv_opt.optimize_lpv_ds_from_data import optimize_lpv_ds_from_data

from test_data_from_ml.load_learned_data_from_ML import load_learned_data_from_ML
from test_data_from_ml.port_data_to_yaml import port_data_to_yaml
import scipy as sci
from utils.plotting.plot_ellopsoid import plot_result_3D
from lpv_opt.my_learn_function_3 import my_learn_function
from lpv_opt.lpv_ds import lpv_ds
from utils_DS.figure_tools.VisualizeEstimatedDS import VisualizeEstimatedDS
from my_debug_toolkit.visualize_ds_segment import visualize_ds_segment
from utils_DS.compute_metrics import compute_dtwd, compute_rmse, compute_e_dot
from utils_DS.compute_lyapunov_function import lyapunov_function_PQLF
from utils_DS.figure_tools.plot_lyap_fct import plot_lyap_fct
from utils_DS.compute_lyapunov_derivative import lyapunov_function_deri_PQLF
import scipy as sp

pkg_dir = r'/Users/haihui_gao/Documents/LabWorkSpace/PythonWorkSpace/ds-opt-python'
chosen_dataset = 4  # 6 # 4 (when conducting 2D test)
sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 4  # Only for real 3D data
Data, Data_sh, att, x0_all, data, dt, traj_length = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
np.save('develop_utils/boundary_plot_test/data_for_boundary.npy', Data)
# data_total = sp.io.loadmat('/Users/haihui_gao/Documents/LabWorkSpace/PythonWorkSpace/learning_ds_parameters/dataset/simple_dynamical_3.mat')
# initial point

# x0_all = data_total['x0_all']
# Data = data_total['Data']
# Data_sh = data_total['Data_sh']
# att = data_total['att']
# dt = data_total['dt']

vel_samples = 10
vel_size = 20
plot_reference_trajectories_DS(Data, att, vel_samples, vel_size)
######################
save_gibbs_result = 0
save_opt_result = 0
do_repro_plot = True

use_sh_data = False
lyap_constr = 2
######################
M = int(len(Data) / 2)
if use_sh_data:
    Xi_ref = Data_sh[:M, :]
    Xi_dot_ref = Data_sh[M:, :]
    x0_all = x0_all - att
    att = np.array([[0], [0]])
    Data = Data_sh
else:
    Xi_ref = Data[0:M, :]
    Xi_dot_ref = Data[M:, :]

##################################################
# Step 2 (GMM FITTING): Fit GMM to Trajectory Data#
##################################################

# 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
# 1: GMM-EM Model Selection via BIC
# 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = Est_options()
est_options.type = 0  # GMM Estimation Algorithm Type
# If algo 1 selected:
est_options.maxK = 10
est_options.fixed_K = []

# If algo 0 or 2 selected:
est_options.samplerIter = 25
est_options.do_plots = 1
# Size of sub-sampling of trajectories
# 1/2 for 2D datasets, >2/3 for real
nb_data = len(Data[0])
sub_sample = 1
if nb_data > 500:
    sub_sample = 2
elif nb_data > 1000:
    sub_sample = 3

est_options.sub_sample = sub_sample
# Metric Hyper-parameters
est_options.estimate_l = 1
est_options.l_sensitivity = 2
est_options.length_scale = []

# instead of wasting your time, you could save the result directly and load them in next run for debug
if save_gibbs_result:
    Priors, Mu, Sigma, table_assignment = fig_gmm(Xi_ref, Xi_dot_ref, est_options)
    idx = knn_search(Mu.T, att.reshape(len(att)), len(Mu[0]))
    Priors_old = Priors.copy()
    Mu_old = Mu.copy()
    Sigma_old = Sigma.copy()
    for i in np.arange(len(idx)):
        Priors[idx[i]] = Priors_old[i]
        Mu[:, idx[i]] = Mu_old[:, i]
        Sigma[idx[i]] = Sigma_old[i]
    # Make the closest Gaussian isotropic and place it at the attractor location
    Sigma[0] = 1 * np.max(np.diag(Sigma[0])) * np.eye(M)
    Mu[:, 0] = att.reshape(len(att))
    gmm = GMM(len(Mu[0]), Priors, Mu.T, Sigma)  # checked 10/22/2022

    # (Recommended!) Step 2.1: Dilate the Covariance matrices that are too thin
    # This is recommended to get smoother streamlines/global dynamics
    ds_gmm = ds_gmms()
    ds_gmm.Mu = Mu
    ds_gmm.Sigma = Sigma
    ds_gmm.Priors = Priors
    adjusts_C = 1
    if adjusts_C == 1:
        if M == 2:
            tot_dilation_factor = 1
            rel_dilation_fact = 0.25
        else:
            # this is for M == 3
            tot_dilation_factor = 1
            rel_dilation_fact = 0.75
        Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact)
        ds_gmm.Sigma = Sigma_

    # ds_gmm.Mu, ds_gmm.Priors, ds_gmm.Sigma, P_opt = load_learned_data_from_ML()
    # just check the result of adjust Covariances
    if M == 3:
        plot_result_3D(ds_gmm.Mu, ds_gmm.Sigma, Xi_ref)
    elif M == 2:
        gmm = GMM(len(ds_gmm.Priors), ds_gmm.Priors, ds_gmm.Mu.T, ds_gmm.Sigma)
        plot_result(Xi_ref, gmm, len(ds_gmm.Mu), ds_gmm.Mu, 2)
    # ds_gmm.Mu, ds_gmm.Priors, ds_gmm.Sigma, P_opt = load_learned_data_from_ML()
    np.save('distribution_difference_finding/Priors.npy', Priors)
    np.save('distribution_difference_finding/Mu.npy', Mu)
    np.save('distribution_difference_finding/Sigma.npy', Sigma_)
    np.save('distribution_difference_finding/table_assignment.npy', table_assignment)
    np.save('distribution_difference_finding/idx.npy', idx)
else:
    Priors = np.load('distribution_difference_finding/Priors.npy')
    Mu = np.load('distribution_difference_finding/Mu.npy')
    Sigma = np.load('distribution_difference_finding/Sigma.npy')
    table_assignment = np.load('distribution_difference_finding/table_assignment.npy')
    ds_gmm = ds_gmms()
    ds_gmm.Mu = Mu
    ds_gmm.Sigma = Sigma
    ds_gmm.Priors = Priors

#### Learn single DS for test purpose
# K = len(ds_gmm.Priors)
# position_member_clusetr_sh = []
# velo_member_cluster_sh = []
# position_member_clusetr = []
# table_assignment = table_assignment.reshape(-1)
# Xi_ref_sh = Data_sh[:M, :]
# Xi_dot_ref_sh = Data_sh[M:, :]
# # from data_sh
# for i in np.arange(1, K + 1):
#     cur_index = (table_assignment == i)
#     position_member_clusetr_sh.append(Xi_ref_sh[:, cur_index])
#     velo_member_cluster_sh.append(Xi_dot_ref_sh[:, cur_index])
#     position_member_clusetr.append(Xi_ref[:, cur_index])
#
# # single_traj = position_member_clusetr[0][:30]
# P_opt = my_learn_function(np.vstack((position_member_clusetr_sh[0], velo_member_cluster_sh[0])))
# A_k, b_k, _ = optimize_lpv_ds_from_data(Data, att, 2, ds_gmm, P_opt, table_assignment)
# num_step = int(len(position_member_clusetr[0][0]) / 3)
# visualize_ds_segment(position_member_clusetr[0], A_k[0], b_k, num_step, 0.01)


#############################################################
# Step 3 (DS ESTIMATION): ESTIMATE SYSTEM DYNAMICS MATRICES #
#############################################################
# DS OPTIMIZATION OPTIONS
# Type of constraints/optimization
# lyap_constr = 2  # % 0:'convex':     A' + A < 0 (Proposed in paper)
# 2:'non-convex': A'P + PA < -Q given P (Proposed in paper)
symm_constr = 0

if lyap_constr == 0 or lyap_constr == 1:
    P_opt = np.eye(M)
else:
    P_opt = my_learn_function(Data_sh)

# if use 1, we use lacy's method, we will put the import lacy here
if lyap_constr == 1 and use_sh_data:
    from lpv_opt.optimized_by_lacy_QP import optimize_lpv_ds_from_data
    A_k, P_opt = optimize_lpv_ds_from_data(Data_sh, table_assignment, att, ds_gmm, dt)
    np.save('A_k.npy', A_k)
    # np.save('b_k.npy', b_k)
    b_k = np.zeros_like(Mu)
else:
    if save_opt_result:
        A_k, b_k, _ = optimize_lpv_ds_from_data(Data, att, lyap_constr, ds_gmm, P_opt, symm_constr)
        # A_k, b_k, _ = optimize_lpv_ds_from_data(Data, att, lyap_constr, ds_gmm, P_opt, table_assignment)
        np.save('A_k.npy', A_k)
        np.save('b_k.npy', b_k)
    else:
        A_k = np.load('A_k.npy')
        b_k = np.load('b_k.npy')
        b_k_test = []
        A_k_test = []
        for k in np.arange(len(ds_gmm.Sigma)):
            b_k_test.append(A_k[k] @ att * (-1))
            A_k_test.append(np.linalg.eig(P_opt @ A_k[k] + A_k[k].T @ P_opt)[0])
        A_k_test = np.array(A_k_test)
        b_k_test = np.array(b_k_test)
        calculate_velo = np.zeros((M, len(Xi_ref[0])))
        for k in np.arange(len(ds_gmm.Sigma)):
            b_k_helper = np.repeat(b_k[:, k].reshape(len(b_k[:, k]), 1), len(Xi_ref[0]), axis=1)
            calculate_velo += (A_k[k] @ Xi_ref + b_k_helper) * ds_gmm.Priors[k]
        port_data_to_yaml('haruhi', ds_gmm, A_k, att, x0_all, dt)
        sci.io.savemat('A_k_b_k.mat', {'A_k': A_k, 'b_k': b_k})
        stop_bar = 1



# Test for Velocity Prediction
if do_repro_plot:
    # Xi_dot_pred = lpv_ds(Xi_ref, ds_gmm, A_k, b_k)
    ds_handle = lambda x_velo: lpv_ds(x_velo, ds_gmm, A_k, b_k)
    rmse = compute_rmse(ds_handle, Xi_ref, Xi_dot_ref)
    e_dot = compute_e_dot(ds_handle, Xi_ref, Xi_dot_ref)
    dtwd = compute_dtwd(ds_handle, Xi_ref, traj_length, x0_all)
    # predict_diff = (Xi_dot_pred - Xi_dot_ref) ** 2
    # trajectory_RMSE = np.sqrt(np.mean(predict_diff, axis=0))
    # rmse = np.mean(trajectory_RMSE)
    # print('LPV-DS got prediction RMSE on training set: {}'.format(rmse))
    ####
    if M == 2:
        print('Doing visualization for 2D dataset')
        title_1 = 'Lyapunov derivative plot'
        title_2 = 'Lyapunov function value plot'
        lyap_handle = lambda x : lyapunov_function_PQLF(x, att, P_opt)
        lyap_derivative_handle = lambda x : lyapunov_function_deri_PQLF(x, att, P_opt, ds_handle)
        plot_lyap_fct(Xi_ref, att, lyap_derivative_handle, title_1)
        plot_lyap_fct(Xi_ref, att, lyap_handle, title_2)

    ####
    # Visualize LPV-DS
    ds_opt_plot_optios = ds_plot_options()
    ds_opt_plot_optios.x0_all = x0_all
    VisualizeEstimatedDS(Xi_ref, ds_handle, ds_opt_plot_optios)
