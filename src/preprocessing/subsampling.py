# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def voxel_downsample(cloud, voxel_size, order_by_index=True, visualize=False):
    """
    Function to apply voxel grid filter and keep original order of points.

    Parameters
    ----------
    cloud : PointCloud object of Open3D
        Consists of point coordinates.
    voxel_size : float
        Size of the voxel grid to filter the cloud.
    order_by_index : bool, optional
        If True will re-order points in index as it was in original cloud. 
        The default is True.
    visualize : bool, optional
        If True, shows downsampled cloud with o3d visualization. 
        The default is False.

    Returns
    -------
    downsampled : PointCloud object of Open3D
        Open3D point cloud downsampled.
    old_indices : numpy array
        Index of downsampled.

    """
   
    bbox = cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound().reshape(3,1)
    max_bound = bbox.get_max_bound().reshape(3,1)
    downsampled, array, _ = cloud.voxel_down_sample_and_trace(voxel_size,
                                                              min_bound, 
                                                              max_bound)
   
    if order_by_index:
        re_index = array
        re_index[re_index==-1] = np.max(array)+10
        order = np.amin(re_index, axis=1)
        
        # Add column of indices:
        indices = np.array(range(0, len(order)))
        brrr = np.array([order, indices]).T
        
        # Sort array:
        indices_ordered = brrr[brrr[:,0].argsort()]
        new_indices = indices_ordered[:,1]
        old_indices = indices_ordered[:,0]
        new_points = np.asarray(downsampled.points)
        new_points[:] = new_points[new_indices]
        downsampled.points = o3d.utility.Vector3dVector(new_points)
        
    
    if visualize: o3d.visualization.draw_geometries([downsampled])
    
    return downsampled, old_indices







