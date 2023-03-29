import numpy as np
from Structs_DS import Opt_Sim
from utils_DS.Simulation import Simulation
import matplotlib.pyplot as plt
from utils_DS.sample_initial_points import sample_initial_points


def VisualizeEstimatedDS(Xi_ref, ds_lpv, ds_plot_options):
    dim = Xi_ref.shape[0]

    # Parse Options
    plot_repr = ds_plot_options.sim_traj  # 是否画reproduction
    x0_all = ds_plot_options.x0_all

    if dim == 3:
        plot_2D_only = 0

        init_type = ds_plot_options.init_type
        nb_pnts = ds_plot_options.nb_points
        plot_volumn = ds_plot_options.plot_vol

    if plot_repr:
        opt_sim = Opt_Sim()
        opt_sim.dt = 0.005
        opt_sim.i_max = 10000
        opt_sim.tol = 0.001
        opt_sim.plot = 0
        x_sim = Simulation(x0_all, ds_lpv, opt_sim)

    if dim == 3:
        num_of_traj = x0_all.shape[1]
        trajs = np.array(x_sim)
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter(Xi_ref[0], Xi_ref[1], Xi_ref[2], c='r', label='original demonstration', s=5)
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'black')
            else:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2],'black', label='reproduced trajectories')
        random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        trajs_rand = np.array(Simulation(random_initial_points, ds_lpv, opt_sim))
        for i in np.arange(nb_pnts):
            cur_traj = trajs_rand[:, :, i].T
            if i == nb_pnts - 1:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
            else:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')
        ax1.legend(loc="best")
        plt.show()
    elif dim == 2:
        num_of_traj = x0_all.shape[1]
        trajs = np.array(x_sim)
        fig, ax1 = plt.subplots()
        ax1.scatter(Xi_ref[0], Xi_ref[1], c='r', label='original demonstration', s=3)
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax1.plot(cur_traj[0], cur_traj[1], 'black')
            else:
                ax1.plot(cur_traj[0], cur_traj[1],'black', label='reproduced trajectories')
        # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        # trajs_rand = np.array(Simulation(random_initial_points, ds_lpv, opt_sim))
        # for i in np.arange(nb_pnts):
        #     cur_traj = trajs_rand[:, :, i].T
        #     if i == nb_pnts - 1:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
        #     else:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')
        ax1.legend(loc="best")
        plt.show()
