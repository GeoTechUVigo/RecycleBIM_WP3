# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from parameters import parameters_Bilbao as pm

from src.utils import array_to_o3d
from src.preprocessing import subsampling as sb
from src.preprocessing import clean_exteriors
from src.elements_seg import extract_floor_ceiling as efc
from src.utils import remove_no_continuous as rop
from src.utils import remove_no_continuous_floor as ropf
from src.utils import bounding_polygon_filter

def extract_boundary_ceiling_or_floor(pcd_in, last_storey):
    """
    In the stairwell area, there is neither a ceiling nor a floor, except on 
    the ground storey, where there is a floor, and on the top storey, where 
    there is a ceiling. This function identifies the floor or ceiling outline 
    of each storey to eliminate the stairwell area.

    Parameters
    ----------
    pcd_in : DataFrame
        Point cloud of the storey plan to identify and extract the area
        corresponding to the stairs.
    last_storey : int
        Last storey of the point cloud.

    Returns
    -------
    boundary_pol : shapely.geometry.polygon.Polygon
        Represent the outline of the floor or ceiling of the storey.

    """
    n_storey = int(pcd_in['id_storey'].unique()[0])
    
    # Voxelization
    pcd_down_o3d, vox_idx = sb.voxel_downsample(
        array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[:, 'x':'z'].values), 
        pm.voxel_size, order_by_index=(True))
    
    # Clean exteriors
    pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(
        pcd_down_o3d, epsilon = pm.eps_1, points_min = pm.pt, 
        return_index=True)
    
    # Extract floor, celling and obstacles
    [floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil
     ] = efc.extract_floor_ceiling(pcd_down_clean, pm.nt, search_param=pm.s_p, 
                                   ransac_th=pm.ransac_th, iterat=pm.iterat, 
                                   return_index=True, timestamp=None, 
                                   visualize=False)
    
    # Extract boundary
    if n_storey != last_storey: 
        [boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx
         ] = rop.remove_no_continuous(pcd_in, floor, obstacles, ceiling, 
                                      idx_floor, idx_obs, idx_ceil, pm.eps_2, 
                                      pm.pt, pm.alpha, pm.voxel_size, pm.zmax,
                                      pm.zmin, pm.dist, pm.init_n, vox_idx, idx_clean, 
                                      visualize = False, save = False)
    else:
        [boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx
         ] = ropf.remove_no_continuous(pcd_in, floor, obstacles, ceiling, 
                                       idx_floor, idx_obs, idx_ceil, pm.eps_2, 
                                       pm.pt, pm.alpha, pm.voxel_size, pm.zmax,
                                       pm.zmin, pm.dist, pm.init_n, vox_idx, idx_clean, 
                                       visualize = False, save = False)
    return boundary_pol


def remove_stairs(pcd_in, boundary_pol, s_p, nt, eps_s, pts_s, ransac_s, init_n, 
                  iterat_s, path_stairs, visualize, save):
    """
    Identify the largest cluster that corresponds to the stairs and remove 
    the area that delimits it.

    Parameters
    ----------
    pcd_in : DataFrame
        Point cloud of the storey.
    boundary_pol : shapely.geometry.polygon.Polygon
        Outline of the floor or ceiling of the storey.
    s_p : open3d.cpu.pybind.geometry.KDTreeSearchParamKNN
        Number of the neighbors that will be searched.
    nt : float
        Minimum value to be considered a candidate to floor and celling via 
        normal estimation.
    eps_s : float
        Density parameter that is used to find neighbouring points.
    pts_s : int
        Minimum number of points to form a cluster.
    ransac_s : float
        Max distance a point can be from the plane model, and still be 
        considered an inlier.
    init_n : int
        Number of initial points to be considered inliers in each iteration.
    iterat_s : int
        Number of iterations.
    path_stairs : str
        File location where the stairs information is stored.
    visualize : bool
        Indicates whether you want to view the results.
    save : bool
        Indicates whether you want to save the results.

    Returns
    -------
    storey_without_stairs : DataFrame
        Point cloud of the storey plan without the area corresponding to the 
        stairs.

    """
    
    p_c = array_to_o3d.array_to_o3d_point_cloud(np.array(pcd_in[['x','y','z']]))
    cl, cl_i = bounding_polygon_filter.bounding_polygon_filter(
        p_c, boundary_pol, pm.zmax, pm.zmin, eliminate='inside')  
    
    pcd_in[['x','y','z']].iloc[cl_i]
    
    if cl.has_normals()==False: 
       cl.estimate_normals(search_param=pm.s_p)
        
    horizontal_index = np.where(np.absolute(np.asarray(cl.normals)[:,2])>nt)[0]  
    
    p_new = array_to_o3d.array_to_o3d_point_cloud(
        np.array(pcd_in[['x','y','z']].iloc[cl_i].iloc[horizontal_index]))
    
    pcd_down_o3d, vox_idx = sb.voxel_downsample(p_new, pm.voxel_size, 
                                                order_by_index=(True))
    p_new = pcd_down_o3d
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug
                                             ) as ctxtman:
            labels = np.array(p_new.cluster_dbscan(eps=eps_s, min_points=pts_s,
                                                   print_progress=True))
    
    # DBSCAN
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.colormaps.get_cmap("tab20")(labels / (
        max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    p_new.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # Select or largest cluster
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug
                                             ) as ctxtman:
        labels = np.array(p_new.cluster_dbscan(eps=eps_s, min_points=pts_s, 
                                               print_progress=True))
    
    # Total number of clusters (excluding noise)
    max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters (excluding noise)")
    
    # Filter out noise (label -1) and count the number of points in each cluster
    cluster_sizes = np.bincount(labels[labels >= 0])
    
    # Finding the index of the largest cluster
    largest_cluster_idx = np.argmax(cluster_sizes)
    # print(f"Largest cluster: {largest_cluster_idx}. Points: {cluster_sizes[largest_cluster_idx]}")
    
    # Filter the points that belong to the largest cluster
    largest_cluster_mask = (labels == largest_cluster_idx)
    
    # Create a new point cloud with only the points from the largest cluster
    largest_cluster_pcd = p_new.select_by_index(np.where(largest_cluster_mask)[0])
    
    # Apply RANSAC to detect a plane in the point cloud
    plane_model, inliers = largest_cluster_pcd.segment_plane(
        distance_threshold=ransac_s, ransac_n=init_n, num_iterations=iterat_s)
    
    # Create a new point cloud with the points belonging to the plane
    inlier_cloud = largest_cluster_pcd.select_by_index(inliers)
    
    # Color the detected plane to visualize it
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) 
    
    # Min and max coordinates
    x_min = min(np.array(largest_cluster_pcd.points)[:,0])
    x_max = max(np.array(largest_cluster_pcd.points)[:,0])
    y_min = min(np.array(largest_cluster_pcd.points)[:,1])
    y_max = max(np.array(largest_cluster_pcd.points)[:,1])
    
    # Filter the points that are inside the square
    filtered_points = pcd_in[
        (pcd_in['x'] >= x_min) & (pcd_in['x'] <= x_max) &
        (pcd_in['y'] >= y_min) & (pcd_in['y'] <= y_max)]
      
    # Filter the points that are inside the square
    st_without_stairs = pcd_in[
        ~((pcd_in['x'] >= x_min) & (pcd_in['x'] <= x_max) &
          (pcd_in['y'] >= y_min) & (pcd_in['y'] <= y_max))]
    
    storey_without_stairs = st_without_stairs.drop(columns=['se']).reset_index(drop=True)
    
    if visualize:
        o3d.visualization.draw_geometries([cl])
        o3d.visualization.draw_geometries([
            array_to_o3d.array_to_o3d_point_cloud(np.array(pcd_in[
                ['x','y','z']].iloc[cl_i].iloc[horizontal_index]))])
        o3d.visualization.draw_geometries([p_new])   
        o3d.visualization.draw_geometries([largest_cluster_pcd])
        o3d.visualization.draw_geometries([inlier_cloud, largest_cluster_pcd])
    
    if save:
        filtered_points.to_csv(path_stairs + '\\stairs.txt', 
                               sep=' ', index=False)
       
        
    return storey_without_stairs, filtered_points
        
        
        
        
        
        
        