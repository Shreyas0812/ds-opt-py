from scipy.io import loadmat
import numpy as np

###############################################
# num_of_traj:  A List containing amount of trajectory
# num_of_case:  3 cases in total, pick whatever you like, this should be a list
# location: the place you place your data
###############################################


def load_learned_data_from_ML(location, num_of_traj, num_of_case):
    Mu_collection = []
    Sigma_collection = []
    Prior_collection = []
    for traj_num in num_of_traj:
        for case_num in num_of_case:
            traj_num = str(traj_num)
            case_num = 'case_' + str(case_num)
            pkg_dir_mu = location + '\\' + traj_num + '\\' + case_num + '\\' + r'matlab_mu.mat'
            pkg_dir_priors = location + '\\' + traj_num + '\\' + case_num + '\\' + r'matlab_priors.mat'
            pkg_dir_sigma = location + '\\' + traj_num + '\\' + case_num + '\\' + r'matlab_sigma.mat'
            data_ = loadmat(r"{}".format(pkg_dir_mu))
            Mu = np.array(data_["Mu"])
            data_ = loadmat(r"{}".format(pkg_dir_priors))
            Priors = np.array(data_["Priors"])[0]
            K = len(Priors)
            dim = len(Mu)
            data_ = loadmat(r"{}".format(pkg_dir_sigma))
            Sigma_data = np.array(data_["Sigma_"])
            # Sigma_data = np.array(data_["Sigma_"])
            Sigma_data_parts = []
            for d in np.arange(dim):
                Sigma_data_parts.append(Sigma_data[d])
            Sigma = np.zeros((K, dim, dim))
            for k in np.arange(K):
                for d in np.arange(dim):
                    Sigma[k][:, d] = Sigma_data_parts[d][:, k]
            Mu_collection.append(np.copy(Mu))
            Sigma_collection.append(np.copy(Sigma))
            Prior_collection.append(np.copy(Priors))

    return Mu_collection, Prior_collection, Sigma_collection

