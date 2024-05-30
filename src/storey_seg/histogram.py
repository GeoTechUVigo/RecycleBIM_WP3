# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def histogram(pcd_in, n_bins, dist):
    """
    Compute the histogram of a dataset.

    Parameters
    ----------
    pcd_in : DataFrame
        PointCloud to be segmented.
    n_bins : int
        Number of equal-width bins in the given range.
    dist : float
        Distance between floor and ceiling.

    Returns
    -------
    hgt_val_pcd : ndarray
        Height values histogram.
    indexes : ndarray
        Indices of peaks in x that satisfy all given conditions.

    """
    # Z-histogram of point cloud
    max_z = pcd_in.loc[:, 'z'].max()
    min_z = pcd_in.loc[:, 'z'].min()

    # Z-histogram of point cloud data
    pcd_zhist = np.histogram(pcd_in.loc[:, 'z'].values, bins=n_bins, range=(min_z, max_z))

    # Mean histogram
    mean_pc_hist = np.mean(pcd_zhist[0])

    # Height values histogram
    hgt_val_pcd = pcd_zhist[1]

    # Find peaks in z-histogram
    indexes, _ = find_peaks(pcd_zhist[0], 
                            height=mean_pc_hist, 
                            distance=dist)

    # View peaks on histogram
    plt.hist(pcd_in.loc[:, 'z'].values, bins=n_bins)
    plt.plot(hgt_val_pcd[1:], pcd_zhist[0], color='blue')
    plt.plot(hgt_val_pcd[1:][indexes], pcd_zhist[0][indexes], "x", color='red', markersize=10)
    plt.xlabel('Heigth (Z)')
    plt.ylabel('Frecuency')
    plt.show()


    return hgt_val_pcd, indexes

