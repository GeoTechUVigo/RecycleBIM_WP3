# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def clean_exteriors(cloud, epsilon, points_min, return_index=False, 
                    visualize=False):
    """
    Function to clean cloud from exterior parts. Uses DBSCAN to achive the task.

    Parameters
    ----------
    cloud : PointCloud object of Open3D
        Open3d cloud to clean exteriors from.
    epsilon : float, optional
        Size to apply DBSCAN. 
    points_min : int, optional
        Minimum points for DBSCAN.
    return_index : bool, optional
        If True return indexes of cleaned point cloud.
        The default is False.
    visualize : bool, optional
        If True shows results of cleaning.
        The default is False.

    Returns
    -------
    cloud_clean : PointCloud object of Open3D
        Open3d cloud with exteriors removed.

    """
    
    labs = np.array(cloud.cluster_dbscan(eps=epsilon, min_points=points_min))
    maxl = labs.max()
    colors = plt.get_cmap("tab20")(labs / (maxl if maxl > 0 else 1))
    colors[labs < 0] = 0
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    labsc = labs[labs>-1]
    biggest = np.bincount(labsc).argmax()
    idx_cloud_clean = np.where(labs==biggest)[0]
    cloud_clean = cloud.select_by_index(idx_cloud_clean)

    
    if visualize:
        o3d.visualization.draw_geometries([cloud])
        o3d.visualization.draw_geometries([cloud_clean])
    
    if not return_index:
        return cloud_clean
    else:
        return cloud_clean, idx_cloud_clean
