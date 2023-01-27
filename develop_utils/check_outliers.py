import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.io import loadmat


def check_outliers(est_table, window_size):
    data_size = len(est_table)
    iter_number = int(data_size / window_size) + 1
    cur_start = 0
    cur_end = 0
    outlier_count = 0
    for i in np.arange(iter_number + 1):
        if i == iter_number:
            cur_end = data_size
        else:
            cur_end += window_size
        cur_check = est_table[cur_start:cur_end]
        count_result = sorted(Counter(cur_check).items(), key=lambda x: x[1], reverse=True)
        if len(count_result) > 1:
            cur_major = count_result[0][0]
            print(count_result)
            if len(count_result) == 2:
                if np.abs(count_result[0][1] - count_result[1][1]) <= 3 or count_result[1][1] >= 3:
                    print("maybe it is boundary, do nothing")
                    cur_start = cur_end
                    continue
            else:
                for out_item in count_result[1:]:
                    outlier_count += out_item[1]
                    cur_check[cur_check == out_item[0]] = cur_major
                est_table[cur_start:cur_end] = cur_check
        cur_start = cur_end

    print("total outliers are " + str(outlier_count))
    return est_table


def plot_cluster_result(est_table, Xi_ref):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    K = len(np.unique(est_table))
    print("cluster number is ", str(K))
    cmap = get_cmap(K)
    for i in range(1, K+1):
        cur_cluster = Xi_ref[:, est_table == i]
        ax.scatter(cur_cluster[0], cur_cluster[1], cur_cluster[2], cmap=cmap, s=1.5)
        mean = np.mean(cur_cluster, axis=1)
        ax.text(mean[0], mean[1], mean[2], str(i), fontdict=None, fontsize=10)

    plt.show()


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':
    labels = np.load(r'E:\ds-opt-python\\ds-opt-python\\ds-opt-python\develop_utils\est_labels.npy')
    # labels = loadmat(r'E:\ds-opt-python\ds-opt-python\ds-opt-python\develop_utils\est_labels.mat')['est_labels'].reshape(-1)
    Xi_ref = np.load(r'E:\ds-opt-python\ds-opt-python\ds-opt-python\develop_utils\Xi_ref.npy')
    plot_cluster_result(labels.reshape(-1), Xi_ref)
    est_table = check_outliers(labels.reshape(-1), 15)