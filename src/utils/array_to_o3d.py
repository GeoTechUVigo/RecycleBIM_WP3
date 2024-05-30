# -*- coding: utf-8 -*-

import open3d as o3d

def array_to_o3d_point_cloud(array_pts):
    """
    Convert a numpy array to a point cloud.

    Parameters
    ----------
    array_pts : numpy array
        Consists of a data structure.

    Returns
    -------
    o3d_point_cloud : PointCloud object of Open3D
        Consists of point coordinates.

    """
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(array_pts)
    
    return o3d_point_cloud 
