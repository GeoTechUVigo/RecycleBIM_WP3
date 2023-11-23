# -*- coding: utf-8 -*-

import numpy as np

from src.utils import array_to_o3d

def DetectMultiPlanes(pcd_np, min_ratio=0.05, threshold=0.01, iterations=1000):
    """
    Detect multiple planes from given point clouds.

    Parameters
    ----------
    pcd_np : numpy array
        Pointcloud to use.
    min_ratio : float, optional
        The minimum left points ratio to end the Detection. The default is 0.05.
    threshold : float, optional
        RANSAC threshold in meters. The default is 0.01.
    iterations : int, optional
        Number of iteration. The default is 1000.

    Returns
    -------
    plane_list : list of numpy array
        Plane equation and plane point index.

    """

    plane_list = []
    N = len(pcd_np)
    target = pcd_np.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(target, threshold=threshold, init_n=3, iter=iterations)

        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list



def PlaneRegression(points, threshold=0.1, init_n=3, iter=1000):
    """
    Plane regression using Ransac.

    Parameters
    ----------
    points : numpy array
        N x 3 point clouds.
    threshold : float, optional
        Distance threshold. The default is 0.1.
    init_n : int, optional
        Number of initial points to be considered inliers in each iteration. 
        The default is 3.
    iter : int, optional
        Number of iteration. The default is 1000.

    Returns
    -------
    w : numpy array
        Indicates the orientation of the plane in three-dimensional space.
    index : list
        Indices of the points that belong to the plane.

    """
    
    pcd = array_to_o3d.array_to_o3d_point_cloud(points)

    w, index = pcd.segment_plane(threshold, init_n, iter)

    return w, index