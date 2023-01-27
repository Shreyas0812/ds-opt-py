import numpy as np
from utils.linalg.my_pca import my_pca
from utils.adjust_Covariances import adjust_Covariances
from datasets.load_dataset_DS import load_dataset_DS

pkg_dir = 'E:\ds-opt-python\ds-opt-python'
chosen_dataset = 7  # 6 # 4 (when conducting 2D test)
sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 1  # Only for real 3D data
Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)

print(np.array([1,2,3])/np.array([1,2,3]))
# Step 1. Test my pca method:
cluster_points = Data[:3, :50]
V_k, L_k, Mu_k = my_pca(cluster_points)
Sigma_k = V_k @ L_k @ V_k.T
print(Sigma_k)
# Step 2. Test Sigma dilate:
rel_dilation_fact = 0.15
Sigma = np.zeros((1,3,3))
Sigma[0] = Sigma_k
Sigma_k = adjust_Covariances(np.array([0.7]), Sigma, 1, rel_dilation_fact)
print(Sigma_k)
