# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.utils import array_to_o3d
from scipy.ndimage import label
import statistics as stat

def seg_traj(path_building, path_trajectory, nt, ransac_th_2, init_n, iterat,
             h_2, g_res, threshold_size):
    """
    Segment the initial point cloud by storeys using the trajectory.
    

    Parameters
    ----------
    path_building : str
        Path of the building point cloud.
    path_trajectory : str
        Point cloud path of the trajectory.
    nt : float
        Threshold that allows filtering normals based on their Z component.
    ransac_th_2 : float
        Max distance a point can be from the plane model, and still be 
        considered an inlier.
    init_n : int
        Number of initial points to be considered inliers in each iteration.
    iterat : int
        Number of iterations.
    h_2 : float
        Estimation of the height at which the trajectory was scanned.
    g_res : int
        Grid resolution.
    threshold_size : int
        Number of empty pixels that will be considered a long size (empty area).

    Returns
    -------
    intervals : list
        Indicates the positions where the point cloud must be cut to perform 
        the segmentation by storeys.

    """
    
    cloud_B = pd.read_csv(path_building, sep=' ', usecols=[0,1,2])
    traj = pd.read_csv(path_trajectory, sep=' ', usecols=[0,1,2], 
                       names=['x','y','z'])
    points = np.array(traj)
      
    # Horizontal planes are selected
    cloud = array_to_o3d.array_to_o3d_point_cloud(points)
    if cloud.has_normals()==False: 
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(40))
        
    horizontal_index = np.where(np.absolute(
        np.asarray(cloud.normals)[:,2])>nt)[0]  
    
    # Point cloud of trajectory
    pcd = array_to_o3d.array_to_o3d_point_cloud(
        np.array(traj.iloc[horizontal_index]))
    
    pcd = traj.iloc[horizontal_index]
    N = len(pcd)
    count = 0
    
    point_clouds = []
    df = []
    equation_df_walls = []
    df_walls = []
    target = pcd
    while count < (1 - 0.05) * N:    
        cloud_target = o3d.geometry.PointCloud()
        cloud_target.points = o3d.utility.Vector3dVector(target.values)
        # Segment planes in the point cloud
        plane_model, inliers = cloud_target.segment_plane(
            distance_threshold=ransac_th_2, ransac_n=init_n, 
            num_iterations=iterat)
    
        index_plane_list = target.index[inliers]
        target = target.drop(index_plane_list)
        
        count += len(inliers)    
        equation_df_walls.append(plane_model)
        df_walls.append(index_plane_list)            
            
    for i in df_walls:
        df.append(traj.loc[i])
        point_clouds.append(array_to_o3d.array_to_o3d_point_cloud(
            np.array(traj.loc[i])))
        print(len(i))    
            
    # Analyze the empty spaces where the storey will be changed
    values = []
    for i in df_walls:
        z_mean = traj.loc[i]['z'].mean()
        values.append(z_mean + h_2)
        values.append(z_mean - h_2)
    
    # Sort the values
    values.sort()
    
    # The intervals are created in z
    interval = []
    for ind_i, i in enumerate(values):
        if ind_i == 0:
            interval.append((float('-inf'), i))
            
        elif ind_i == (len(values)-1):
            interval.append((i, float('inf')))
            
        elif ind_i % 2 != 0:
            interval.append((values[ind_i], values[ind_i + 1]))    
    
    # Create a list to store the filtered dataframes
    filtered_dataframes = []
    
    # Iterate over the ranges and apply the filter
    for z_min, z_max in interval:
        df_filtered = cloud_B[(cloud_B['z'] >= z_min) & (
            cloud_B['z'] <= z_max)]
        filtered_dataframes.append(df_filtered)
    
    # Show results
    for i, df_filtered in enumerate(filtered_dataframes):
        print(f"Values of 'z' between {interval[i][0]} and {interval[i][1]}:")
        print(df_filtered)
        print("\n")
    
    
    #%%
    cut = []
    for data_2 in range(1, len(filtered_dataframes) - 1):
        # Assuming you have the x, y, z coordinates in arrays or columns of a dataframe
        x = filtered_dataframes[data_2]['x'].values
        y = filtered_dataframes[data_2]['y'].values
        z = filtered_dataframes[data_2]['z'].values
        
        # Create a boolean mask that does not select the edges
        mask = (y >= np.min(y) + 1) & (y <= np.max(y) - 1)
        
        # Apply the mask to the array and select only the values ​​within the range
        y = y[mask]
        z = z[mask]
            
        # Step 1: Define the grid resolution
        res_y = g_res  # number of divisions on the y-axis
        res_z = g_res  # number of divisions on the z-axis
        
        # Step 2: Create a 2D histogram (y vs z)
        hist, yedges, zedges = np.histogram2d(y, z, bins=[res_y, res_z])
        
        # Step 3: Identify empty cells
        empty_spaces = (hist == 0)
        
        # Step 4: Label the adjacent empty areas
        struct = np.ones((3, 3), dtype=bool) 
        labeled_empty_spaces, num_areas = label(empty_spaces, structure=struct)
        
        # Step 5: Calculate the size of each empty area
        area_size = np.bincount(labeled_empty_spaces.ravel())[1:]  
        
        # Filter out large areas
        large_areas = np.where(area_size >= threshold_size)[0] + 1 
        
        # Create a mask for large areas
        mask_large_areas = np.isin(labeled_empty_spaces, large_areas)
        
        # Step 6: Visualize the large areas on the grid
        plt.figure()
        plt.imshow(mask_large_areas.T, origin='lower', cmap='Greens', 
                   interpolation='nearest')
        plt.colorbar(label='Large spaces (1: large gap, 0: not gap)')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('Detection of large gaps in the point cloud')
        plt.show()
        
        # Step 7: Map the large spaces back to the point cloud
        # Find the edges of the grid in y and z
        y_min, y_max = yedges[0], yedges[-1]
        z_min, z_max = zedges[0], zedges[-1]
        
        # Convert grid positions to original (y, z) coordinates
        y_coords_large = []
        z_coords_large = []
        
        for i in range(res_y):
            for j in range(res_z):
                if mask_large_areas[i, j]:
                    # Map cell (i, j) to its y and z coordinates
                    y_center = (yedges[i] + yedges[i+1]) / 2
                    z_center = (zedges[j] + zedges[j+1]) / 2
                    y_coords_large.append(y_center)
                    z_coords_large.append(z_center)
        
        # Convert lists to arrays
        y_coords_large = np.array(y_coords_large)
        z_coords_large = np.array(z_coords_large)
        
        # Step 8: Plot the points of the large empty areas on the original cloud
        plt.figure()
        plt.scatter(y, z, s=1, label='Original points')
        plt.scatter(y_coords_large, z_coords_large, c='red', s=20, 
                    label='Large empty spaces')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.legend()
        plt.title('Large empty areas mapped to point cloud')
        plt.show()
        
        cut.append(stat.mode(z_coords_large))
    
    # Create a dictionary to store the sliced ​​DataFrames
    intervals = {}
    
    # Define the first interval
    first_upper_bound = cut[0]
    
    # Filter the first interval
    mask_first = (cloud_B['z'] < first_upper_bound)
    intervals['interval_0'] = cloud_B[mask_first]
    
    # Iterate over the intermediate intervals and the last one
    for i in range(0, len(cut)-1):
        lower_bound = cut[i]
        upper_bound = cut[i + 1]
    
        # Filter rows where 'z' is in the range (lower_bound, upper_bound]
        mask = (cloud_B['z'] > lower_bound) & (cloud_B['z'] <= upper_bound)
        intervals[f"interval_{i + 1}"] = cloud_B[mask]
    
    # Define the last interval
    last_lower_bound = cut[-1]
    last_upper_bound = float('inf')  
    
    # Filter the last interval
    mask_last = (cloud_B['z'] > last_lower_bound)
    intervals['interval_last'] = cloud_B[mask_last]
    
   
    return intervals










