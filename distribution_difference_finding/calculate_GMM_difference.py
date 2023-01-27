import numpy as np
from distribution_difference_finding.gmmot import *
from distribution_difference_finding.load_learned_data_from_ML import load_learned_data_from_ML
import itertools as it


def matlab_internal_compare(compare_type):
    location = r'E:\ds-opt-python\ds-opt-python\distribution_difference_finding\matlab_data'
    dist_collection = []
    if compare_type == 'case':
        num_of_traj = [1]
        num_of_case = [1, 2, 3]
        iteration_term = num_of_case
    else:
        num_of_traj = [1, 3, 4, 5, 6, 7]
        num_of_case = [2]
        iteration_term = np.arange(len(num_of_traj))

    Mu_collection, Prior_collection, Sigma_collection = load_learned_data_from_ML(location, num_of_traj, num_of_case)
    for combi in it.combinations(iteration_term, 2):
        G_1 = combi[0] - 1
        G_2 = combi[1] - 1
        # We should post process the Prior to make the sum it strictly equals to 1
        Prior_collection[G_1][-1] = 1 - np.sum(Prior_collection[G_1][:-1])
        Prior_collection[G_2][-1] = 1 - np.sum(Prior_collection[G_2][:-1])
        _, dist = GW2(Prior_collection[G_1], Prior_collection[G_2], Mu_collection[G_1].T, Mu_collection[G_2].T, Sigma_collection[G_1], Sigma_collection[G_2])
        print("the GW2 difference between " + compare_type + str(combi) + ' is ' + str(dist))


def cal_GMM_diff(Prior_1, Prior_2, Mu_1, Mu_2, Sigma_1, Sigma_2):
    Mu_1_ = np.copy(Mu_1.T)
    Mu_2_ = np.copy(Mu_2.T)
    ws_matrix, dist = GW2(Prior_1, Prior_2, Mu_1_, Mu_2_, Sigma_1, Sigma_2)
    print("the difference between GMM1 and GMM2 is " + str(dist))
    return ws_matrix, dist


if __name__ == '__main__':
    matlab_internal_compare('case')
    # Priors = np.load('Priors.npy')
    # Mu = np.load('Mu.npy')
    # Sigma = np.load('Sigma.npy')
    # location = r'E:\ds-opt-python\ds-opt-python\distribution_difference_finding\matlab_data'
    # Priors[-1] = 1 - np.sum(Priors[:-1])
    # Mu_collection, Prior_collection, Sigma_collection = load_learned_data_from_ML(location, [4], [4])
    # Prior_collection[0][-1] = 1 - np.sum(Prior_collection[0][:-1])
    # cal_GMM_diff(Prior_collection[0], Priors, Mu_collection[0], Mu, Sigma_collection[0], Sigma)

