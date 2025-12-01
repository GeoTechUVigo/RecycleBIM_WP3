# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import pyntcloud as pc
import pandas as pd

from pyntcloud import PyntCloud

def extract_floor_ceiling(cloud, nt, search_param, ransac_th, iterat, timestamp,
                          return_index, visualize):
    """
    Extract floor, ceiling and others from pointcloud.

    Parameters
    ----------
    cloud : PointCloud object of Open3D
        Point cloud with which you work.
    nt : float
        Minimum value to be considered a candidate to floor and celling via 
        normal estimation.
    search_param : float
        KDTree search parameters for pure KNN search.
    ransac_th : float
        Maximum distance from point to model in order to be considered as inlier.
    iterat :
        
    timestamp : numpy array
        Numpy array with timestamps of each point.
    return_index : bool
        If True return indexes of points of floor, ceiling and others.
    visualize : bool, optional
        If True shows image. 

    Returns
    -------
    Point clouds of floor, ceiling and others and their respective indexes (if
    return_index is True)

    """
    
    # Candidates to floor and celling via normal estimation:
    if cloud.has_normals()==False: 
        if search_param is None: cloud.estimate_normals()
        else: cloud.estimate_normals(search_param=search_param)
        
    normals = np.asarray(cloud.normals)
    
    candidates = np.where(np.absolute(normals[:,2])>nt)[0]
    planes = cloud.select_by_index(candidates, invert=False)
    
    # Extraction with RANSAC:
    cloudPC = pc.PyntCloud.from_instance("open3d", planes)
    _ = cloudPC.add_scalar_field("plane_fit", max_dist=ransac_th, 
                                 max_iterations=iterat, n_inliers_to_stop=None) 
    is_plane = cloudPC.points['is_plane']
    
    plane1_idx = np.where(is_plane==1)[0]
    plane1 = candidates[plane1_idx] # GLOBAL indices for plane 1
     
    plane = cloudPC.xyz[is_plane==1]
    ceiling = pc.PyntCloud(pd.DataFrame(plane, index=None, 
                                        columns=['x', 'y', 'z']))
    ceiling = ceiling.to_instance("open3d", mesh=True)
    
    # Get celling timestamp:
    if timestamp is not None: 
        select = np.in1d(range(timestamp.shape[0]), plane1)
        ceiling_timestamp = timestamp[select]
        rest_timestamp = timestamp[~select]
            
    select = np.in1d(range(candidates.shape[0]), plane1_idx)
    candidates2 = candidates[~select]
        
    is_plane = cloudPC.points['is_plane']
    rest = cloudPC.xyz[is_plane==0]
    notCeiling = pc.PyntCloud(pd.DataFrame(rest, index=None, 
                                           columns=['x', 'y', 'z']))
    notCeiling.add_scalar_field("plane_fit", max_dist=ransac_th, 
                                max_iterations=iterat, n_inliers_to_stop=None) 
    is_plane = notCeiling.points['is_plane']
    
    plane2_idx = np.where(is_plane==1)[0]
    plane2 = candidates2[plane2_idx] # GLOBAL indices for plane 2
    
    floor = pc.PyntCloud(pd.DataFrame(notCeiling.xyz[is_plane==1], index=None, 
                                      columns=['x', 'y', 'z']))
    floor = floor.to_instance("open3d", mesh=True)
    
    planes_idx = np.concatenate((plane1, plane2), axis=0)
    
    # Invert indices to get obstacles indices
    obs_idx = np.arange(len(cloud.points))[~np.isin(np.arange(
        len(cloud.points)), planes_idx)]
    
    # Get floor and obst timestamp:
    if timestamp is not None: 
        select = np.in1d(range(timestamp.shape[0]), planes_idx)
        floor_timestamp = timestamp[plane2]
        obst_timestamp = timestamp[~select]
    
    obstacle_walls = pc.PyntCloud(pd.DataFrame(
        notCeiling.xyz[is_plane==0], index=None, columns=['x', 'y', 'z']))
    obstacle_walls = cloud.select_by_index(planes_idx, invert=True)
    
    # Mean heigth of celling and floor
    meanC = np.mean(np.asarray(ceiling.points)[:,2])
    meanF = np.mean(np.asarray(floor.points)[:,2])
    
    # Flip celling-floor.
    if meanC < meanF:
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(np.asarray(ceiling.points))
        if timestamp is not None: tempI = np.array(ceiling_timestamp)
        
        ceiling.points      = o3d.utility.Vector3dVector(np.asarray(floor.points))
        floor.points        = o3d.utility.Vector3dVector(np.asarray(temp.points)) 
        temp_plane = np.copy(plane1)
        plane1 = plane2
        plane2 = np.copy(temp_plane)
        del temp_plane
        if timestamp is not None: ceiling_timestamp   = np.array(floor_timestamp)
        if timestamp is not None: floor_timestamp     = np.array(tempI)
        if timestamp is not None: del temp, tempI
   
    if visualize: 
        o3d.visualization.draw_geometries([floor])
        o3d.visualization.draw_geometries([ceiling])
        o3d.visualization.draw_geometries([obstacle_walls])
    
    if timestamp is None:
        if not return_index:
            return floor, obstacle_walls, ceiling
        else:
            return floor, obstacle_walls, ceiling, plane2, obs_idx, plane1 
    else:
        if not return_index:
            return [floor, floor_timestamp, obstacle_walls, obst_timestamp, 
                    ceiling, ceiling_timestamp]
        else:
            return [floor, floor_timestamp, obstacle_walls, obst_timestamp, 
                    ceiling, ceiling_timestamp, plane2, obs_idx, plane1]