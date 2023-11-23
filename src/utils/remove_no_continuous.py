# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import alphashape as alph

from src.utils import array_to_o3d
from src.utils import bounding_polygon_filter
from src.preprocessing import clean_exteriors

def remove_no_continuous(pcd_in, floor, obstacles, ceiling, idx_floor, idx_obs,
                         idx_ceil, eps, pt, alpha, voxel_size, zmax, zmin, 
                         dist, vox_idx, idx_clean, visualize, save):
    """
    Remove non continuos celling.

    Parameters
    ----------
    pcd_in : DataFrame
        Is the pointcloud.
    floor : PointCloud object of Open3D
        Floor open3d cloud.
    obstacles : PointCloud object of Open3D
        Obstacles open3d cloud.
    ceiling : PointCloud object of Open3D
        Ceiling open3d cloud.
    idx_floor : numpy array
        Floor indices.
    idx_obs : numpy array
        Onbstacles indices.
    idx_ceil : numpy array
        Ceiling indices.
    eps : float
        Size to apply DBSCAN.
    pt : int
        Minimum points for DBSCAN.
    alpha : float
        DESCRIPTION.
    voxel_size : float
        Voxel size
        Is measured in meters.
    zmax : int
        Maximum heigth of polygon filter.
    zmin : int
        Minimum heigth of polygon filter.
    dist : float
        Maximum distance a point can have to an estimated plane to be considered an inlier.
    vox_idx : numpy array
        Voxel indices.
    idx_clean : TYPE
        Indices of clean pointcloud.
    visualize : str
        True : show pictures
        False : not show pictures.
    save : str
        True : save results.
        False : not save results.
        
    Returns
    -------
    boundary_pol : shapely.geometry.polygon.Polygon
        DESCRIPTION.
    floor_plane : numpy array
        Segmentation of geometric primitives from floor pointcloud .
    ceiling_plane : numpy array
        Segmentation of geometric primitives from ceiling pointcloud.
    pcd_in : DataFrame
        Is the pointcloud.
    clean_idx : numpy array
        Are the indices of the clean point cloud.

    """
    # Remove non continuos celling
    ceiling_clean, idx_clean_ceil = clean_exteriors.clean_exteriors(ceiling, epsilon = eps, points_min = pt, return_index=(True))
    
    # Compute concave hull from cleaned celling
    concave_hull = alph.alphashape(np.asarray(ceiling_clean.points)[:,0:2], alpha)
    boundary_pol = concave_hull.buffer(voxel_size*2)
    # Extract cleaned floor by using concave hull
    floor_clean, idx_clean_floor =  bounding_polygon_filter.bounding_polygon_filter(floor, boundary_pol, zmax, zmin, eliminate='outside')
    floor_clean = floor.select_by_index(idx_clean_floor)
    # Extract cleaned obstacles by using concave hull
    obst_clean, idx_clean_obst =  bounding_polygon_filter.bounding_polygon_filter(obstacles, boundary_pol, zmax, zmin, eliminate='outside')
    obst_clean = obstacles.select_by_index(idx_clean_obst)
    
    if visualize:        
        o3d.visualization.draw_geometries([floor_clean, ceiling_clean, obst_clean], 
                                          window_name="Cleaned point cloud after boundary filtering")
    
    # Filtering based on sign of point-to-plane distance
    # Compute planes from floor and ceil
    floor_plane, _ = floor_clean.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=zmax)
    ceiling_plane, _ = ceiling_clean.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=zmax)
    
    # Compute distance from obstacle points to planes
    obs_v = np.ones((len(obst_clean.points), 4))
    obs_v[:, :3] = np.asarray(obst_clean.points)
    dist = -np.dot(obs_v, floor_plane)/np.linalg.norm(floor_plane[:3])
    idx_floor_plane = np.where(dist <= 0)[0]
    obstacles_p1 = obst_clean.select_by_index(idx_floor_plane, invert=False)
    
    # Filter points outer floor and celling planes
    obs_v = np.ones((len(obstacles_p1.points), 4))
    obs_v[:, :3] = np.asarray(obstacles_p1.points)
    dist = -np.dot(obs_v, ceiling_plane)/np.linalg.norm(ceiling_plane[:3])
    idx_ceil_plane = np.where(dist >= 0)[0]
        
    # Global indices
    glob_idx_floor_clean = vox_idx[idx_clean][idx_floor][idx_clean_floor]
    glob_idx_ceil_clean = vox_idx[idx_clean][idx_ceil][idx_clean_ceil]
    glob_idx_filt_obs = vox_idx[idx_clean][idx_obs][idx_clean_obst][idx_floor_plane][idx_ceil_plane]
    
    clean_idx = np.concatenate((glob_idx_ceil_clean, glob_idx_floor_clean, glob_idx_filt_obs))
        
    if visualize or save:
        filt_floor_o3d = array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[glob_idx_floor_clean, 'x':'z'].values)
        filt_ceiling_o3d = array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[glob_idx_ceil_clean, 'x':'z'].values)
        filt_obstacles_o3d = array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[glob_idx_filt_obs, 'x':'z'].values)
        
        if visualize:
            filt_floor_o3d.paint_uniform_color([1, 0, 0])
            filt_ceiling_o3d.paint_uniform_color([0, 1, 0])
            filt_obstacles_o3d.paint_uniform_color([0.7, 0.7, 0.7])
            
            o3d.visualization.draw_geometries([filt_floor_o3d, filt_ceiling_o3d, filt_obstacles_o3d], 
                                              window_name="Cleaned pcd after sign point-to_plane distance filtering")
            
            
    # Add classified label (floor->0, ceiling->1, obstacles->2, noise->-1)
    labels_arr = np.ones(pcd_in.shape[0], dtype=int)*-1
    labels_arr[glob_idx_floor_clean] = 0
    labels_arr[glob_idx_ceil_clean] = 1
    labels_arr[glob_idx_filt_obs] = 2
    
    pcd_in['se'] = labels_arr.tolist()
    
    
    return boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx