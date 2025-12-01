# -*- coding: utf-8 -*-

import geopandas as gpd
from shapely.geometry import Point

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import cv2

def lin_reg(points_wall):
    """
    Performs linear regression to fit a line to a set of 2D points representing
    a wall.

    Parameters
    ----------
    points_wall : ndarray
        A 2D numpy array of shape (n, 2) containing the points, where each row
        corresponds to a point with the first column being the x-coordinate 
        and the second column being the y-coordinate.

    Returns
    -------
    m : float
        The slope (m) of the best-fit line obtained from the linear regression.

    """
    x = points_wall[:,0]
    y = points_wall[:,1]
    model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    m = model.coef_[0][0]
    return m

def rot_pcd(m1, m2, points):
    """
    Rotates a set of 2D points in a point cloud based on the angle between
    two lines with slopes m1 and m2.


    Parameters
    ----------
    m1 : float
        The slope of the first line.
    m2 : float
        The slope of the second line.
    points : ndarray
        A 2D numpy array of shape (n, 2) representing the points in the point 
        cloud, where each row corresponds to a point with (x, y) coordinates.

    Returns
    -------
    angle_radians :  float
        The rotation angle in radians between the two lines.
    angle_degrees : float
        The rotation angle in degrees between the two lines.
    rotated_points : ndarray
        A 2D numpy array of shape (n, 2) containing the points after applying 
        the rotation.
    cw : bool
        A boolean indicating the rotation direction: 
        `True` if the rotation is clockwise, `False` if counter-clockwise.

    """
    # Calculate the angle between the fitted line and a horizontal line
    angle_radians = abs(math.atan2((m2 - m1), 1 + m1 * m2))
    angle_degrees = math.degrees(angle_radians)        
    if m2 > m1:
        # Counter-clock wise
        rotation_matrix = np.array([[np.cos(-angle_radians), -np.sin(-angle_radians)],
                                  [np.sin(-angle_radians), np.cos(-angle_radians)]])
        cw = False
    elif m1 > m2:
        # Clockwise
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                  [np.sin(angle_radians), np.cos(angle_radians)]])
        cw = True

    else: 
        angle_radians = 2 * np.pi
        angle_degrees = math.degrees(angle_radians)   
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                  [np.sin(angle_radians), np.cos(angle_radians)]])
        cw = True 
    # Applies rotation to each point in the point cloud
    rotated_points = np.dot(points, rotation_matrix)
       
    return angle_radians, angle_degrees, rotated_points, cw

def rot_some_points(points_wall, r_points, iter_i, min_a, max_a):
    """
    Rotate the points to view them in 2D and identify the end points of the 
    windows

    Parameters
    ----------
    points_wall : ndarray
        Points corresponding to the wall.
    r_points : ndarray
        Rotated points.
    iter_i : int
        Number of iterations.
    min_a : float
        Minimum size that the window can have.
    max_a : float
        Maximum size that the window can have.

    Returns
    -------
    rotated_points_z : ndarray
        Wall points rotated in 3D.
    ext_window : ndarray
        Extreme points of windows.

    """
    # Rotate only necessary points
    rotated_points_z = np.column_stack((r_points, points_wall[:,2]))
    
    # Now extreme points of the windows are calculated.
    # Defines the dimensions of the plane
    x_min, x_max = np.min(rotated_points_z[:,0]), np.max(rotated_points_z[:,0])
    z_min, z_max = np.min(rotated_points_z[:,2]), np.max(rotated_points_z[:,2])
    resolution = 255

    # Create a matrix to represent the plane
    plane = np.zeros((resolution, resolution))

    # Convert point coordinates to array indices
    indices_x = np.floor((rotated_points_z[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    indices_z = np.floor((rotated_points_z[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)).astype(int)

    # Mark the points on the plane matrix
    plane[indices_z, indices_x] = 1

    # Mark the absence of points on the plane matrix
    no_points_plane = 1 - plane

    # Apply erosion to reduce cluster size
    eroded_plane = binary_erosion(no_points_plane, iterations=iter_i)

    # Apply dilation to restore the original size of the clusters
    dilated_plane = binary_dilation(eroded_plane, iterations=iter_i)

    # Find the contours in the dilated plane
    contours, _ = cv2.findContours((dilated_plane * 255).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Plotting
    # plt.figure(figsize=(10, 10))
    
    # # Original no_points_plane
    # plt.subplot(1, 3, 1)
    # plt.title('Original no_points_plane')
    # plt.imshow(no_points_plane, cmap='gray', origin='lower')
    
    # # Eroded Plane
    # plt.subplot(1, 3, 2)
    # plt.title('Eroded Plane')
    # plt.imshow(eroded_plane, cmap='gray', origin='lower')
    
    # # Dilated Plane with Contours
    # plt.subplot(1, 3, 3)
    # plt.title('Dilated Plane with Contours')
    # plt.imshow(dilated_plane, cmap='gray', origin='lower')
    
    # # Draw contours on the dilated plane
    # for contour in contours:
    #     contour = contour.squeeze()  
    #     plt.plot(contour[:, 0], contour[:, 1], color='red')
    
    # plt.show()
    
    # Find the bounding box for the largest outline (the area without points)    
    one_or_more = len(contours)
    
    if one_or_more > 1:
        ext_window = []
        scale_x = (x_max - x_min)  
        scale_z = (z_max - z_min)  

        for c in contours:
            # Find the bounding box for windows
            x, z, w, h = cv2.boundingRect(c)
            real_area = (w / resolution * scale_x) * (h / resolution * scale_z)
            if real_area > min_a and real_area < max_a:            
                # # Draws the bounding box on the original visualization
                # plt.imshow(no_points_plane, cmap='Blues',
                #            extent=[x_min, x_max, z_min, z_max], origin='lower')
                # plt.title('Area without points with boundary box')
                # plt.xlabel('X')
                # plt.ylabel('Z')
                # # Draw the bounding box
                # plt.gca().add_patch(plt.Rectangle((x_min + x / resolution * (x_max - x_min), z_min + z / resolution * (z_max - z_min)),
                #                                   w / resolution * (x_max - x_min), h / resolution * (z_max - z_min),
                #                                   linewidth=2, edgecolor='red', facecolor='none'))
                # plt.show()
    
                # Coordinates of the bottom left point of the bounding box
                bottom_left = (x_min + x / resolution * (x_max - x_min),
                               z_min + z / resolution * (z_max - z_min))
    
                # Coordinates of the bottom rigth point of the bounding box
                bottom_right = (bottom_left[0] + w / resolution * (x_max - x_min), 
                                bottom_left[1])
    
                # Coordinates of the top left point of the bounding box
                top_left = (bottom_left[0],
                            bottom_left[1] + h / resolution * (z_max - z_min))
    
                # Coordinates of the top rigth point of the bounding box
                top_right = (bottom_right[0], top_left[1])
                
                ext_window_t = []
                ext_window_t.append(np.array(top_left))
                ext_window_t.append(np.array(top_right))
                ext_window_t.append(np.array(bottom_left))
                ext_window_t.append(np.array(bottom_right))
                
                ext_window.append(ext_window_t)
        
    elif one_or_more == 1:
        ext_window = []
        scale_x = (x_max - x_min) 
        scale_z = (z_max - z_min)  
        x, z, w, h = cv2.boundingRect(contours[0])
        
        real_area = (w / resolution * scale_x) * (h / resolution * scale_z)
        if real_area > min_a and real_area < max_a:            
            # # Draws the bounding box on the original visualization
            # plt.imshow(no_points_plane, cmap='Blues', 
            #            extent=[x_min, x_max, z_min, z_max], origin='lower')
            # plt.title('Area without points with boundary box')
            # plt.xlabel('X')
            # plt.ylabel('Z')
            # # Draw the bounding box
            # plt.gca().add_patch(plt.Rectangle((x_min + x / resolution * (x_max - x_min), z_min + z / resolution * (z_max - z_min)),
            #                                   w / resolution * (x_max - x_min), h / resolution * (z_max - z_min),
            #                                   linewidth=2, edgecolor='red', facecolor='none'))
            # plt.show()
            
            # Coordinates of the bottom left point of the bounding box
            bottom_left = (x_min + x / resolution * (x_max - x_min), z_min + z / resolution * (z_max - z_min))
    
            # Coordinates of the bottom rigth point of the bounding box
            bottom_right = (bottom_left[0] + w / resolution * (x_max - x_min), bottom_left[1])
    
            # Coordinates of the top left point of the bounding box
            top_left = (bottom_left[0], bottom_left[1] + h / resolution * (z_max - z_min))
    
            # Coordinates of the top rigth point of the bounding box
            top_right = (bottom_right[0], top_left[1])
    
            ext_window_t = []
            ext_window_t.append(np.array(top_left))
            ext_window_t.append(np.array(top_right))
            ext_window_t.append(np.array(bottom_left))
            ext_window_t.append(np.array(bottom_right))
            
            ext_window.append(ext_window_t)

    elif one_or_more == 0:
        ext_window = []
     
    return rotated_points_z, ext_window

# Find nearest point
def find_nearest_point(point, points):
    """
    Finds the nearest point to a given point from a set of points in a 2D space.

    Parameters
    ----------
    point : ndarray
        A 1D numpy array representing the coordinates (x, y) of the point from which 
        the nearest neighbor is to be found.
    points : ndarray
        A 2D numpy array of shape (n, 2) representing the set of points (x, y) 
        where the nearest point to the given point will be searched.

    Returns
    -------
    nearest_point : ndarray
        A 1D numpy array representing the coordinates of the nearest point to 
        the given point.
    nearest_index : int
        The index of the nearest point within the `points` array.

    """
    distances = np.linalg.norm(points - point, axis=1)
    
    # Find the index of the nearest point
    nearest_index = np.argmin(distances)
    
    # Returns the nearest point and its index
    return points[nearest_index], nearest_index
    
def inverse_rotate_selected_points(selected_points, angle_radians, 
                                   clockwise=True):
    """
    Applies the inverse of a 2D rotation to a set of selected points.

    Parameters
    ----------
    selected_points : ndarray
        A 2D numpy array of shape (n, 2) representing the selected points 
        (x, y) that need to be inverse rotated.
    angle_radians : float
        The angle by which the points were originally rotated, expressed in 
        radians. This is used to reverse the rotation.
    clockwise : bool, optional
        A boolean indicating the direction of the original rotation. If `True`,
        the points were rotated clockwise; 
        if `False`, the points were rotated counter-clockwise. 
        The default is `True`.

    Returns
    -------
    original_points : ndarray
        A 2D numpy array of shape (n, 2) representing the points after the 
        inverse rotation has been applied, restoring them to their original positions.

    """
    if not clockwise:
        angle_radians = -angle_radians

    # Calculate the inverse rotation matrix
    inverse_rotation_matrix = np.array([[np.cos(angle_radians), 
                                         np.sin(angle_radians)],
                                         [-np.sin(angle_radians), 
                                          np.cos(angle_radians)]])
    # Applies reverse rotation to selected points only
    original_points = np.dot(selected_points, inverse_rotation_matrix)

    return original_points



def filter_points_boundary(pcd_in, boundary_pol, th_boundary):
    """
    Filters a set of points based on their proximity to a specified boundary 
    polygon.

    Parameters
    ----------
    pcd_in : DataFrame
        A pandas DataFrame containing the points to be filtered. The DataFrame 
        should include at least two columns representing the x and y
        coordinates of the points (e.g., 'x' and 'y').
    boundary_pol : shapely.geometry.Polygon
        A Shapely polygon object representing the boundary within which the
        points will be filtered. The points within a specified distance
        threshold from the boundary will be retained.
    th_boundary : float
        A numeric value representing the distance threshold. Only points that 
        are within this distance from the boundary will be included in the 
        output.

    Returns
    -------
    filtered_points : geopandas.GeoDataFrame
        A GeoDataFrame containing the points that are within the specified 
        distance threshold from the boundary.

    """
    
    geometry = [Point(xy) for xy in zip(pcd_in['x'], pcd_in['y'])]
    
    gdf_points = gpd.GeoDataFrame(pcd_in, geometry=geometry)
    
    gdf_points['distance'] = gdf_points.geometry.distance(boundary_pol.boundary)
    
    filtered_points = gdf_points[gdf_points['distance'] < th_boundary]
    
    return filtered_points


def windows(filtered_points, dist_th_w, init_n, iterat, iter_i, min_a, max_a):
    """
    Segments a point cloud into windows based on a plane fitting algorithm and 
    applies rotation and filtering to extract points of interest.

    Parameters
    ----------
    filtered_points : pandas.DataFrame
        A DataFrame containing the 3D points ('x', 'y', 'z') to be segmented. 
        It represents the filtered point cloud.
    dist_th_w : float
        The distance threshold for the RANSAC plane segmentation. This defines 
        the maximum distance a point can be from the plane to be considered as part of it.
    init_n : int
        The number of points used to estimate the plane in the RANSAC algorithm.
    iterat : int
        The number of iterations for the RANSAC algorithm.
    iter_i : int
        The number of iterations used for point rotation filtering.
    min_a : float
        The minimum angle for filtering points after rotation.
    max_a : float
        The maximum angle for filtering points after rotation.

    Returns
    -------
    all_points : list
        A list of 3D points after segmentation, rotation, and filtering, 
        where each element corresponds to a different segmented window of points.


    """
    all_points = []
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array(filtered_points[['x','y','z']]))
    
    if cloud.has_normals()==False: 
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(40))
            
    N = len(filtered_points[['x','y','z']])
    count = 0
            
    equation_df_walls = []
    df_walls = []
    target = filtered_points[['x','y','z']]
    while count < (1 - 0.05) * N:        
        cloud_target = o3d.geometry.PointCloud()
        cloud_target.points = o3d.utility.Vector3dVector(target.values)

        plane_model, inliers = cloud_target.segment_plane(
            distance_threshold=dist_th_w, ransac_n=init_n, num_iterations=iterat) 
        # o3d.visualization.draw_geometries([cloud_target.select_by_index(inliers)])
         
        index_plane_list = target.index[inliers]
        target = target.drop(index_plane_list)
        
        count += len(inliers) 
    
        equation_df_walls.append(plane_model)
        df_walls.append(index_plane_list)         
        
    
        m = lin_reg(filtered_points[['x','y','z']].loc[index_plane_list].values) 
            
        # Calculate the angle
        angle_r, angle_d, r_points, cw = rot_pcd(
            np.round(m, 2), 0, filtered_points[['x','y']].loc[index_plane_list
                                                              ].values)
    
        rotated_points_z, ext_window = rot_some_points(
            filtered_points[['x','y','z']].loc[index_plane_list].values, 
            r_points, iter_i, min_a, max_a)
    
        if len(ext_window) != 0:   
            if len(ext_window) == 1:
                p_t_r = []
                for p_i in ext_window[0]:
                    nearest_point, index = find_nearest_point(
                        p_i, rotated_points_z[:, [0, 2]])
                    p_t_r.append(rotated_points_z[index, :])
                
                pt_f = np.column_stack((np.array(ext_window[0])[:,0], 
                                        np.array(p_t_r)[:,1], 
                                        np.array(ext_window[0])[:,1]))
                
                original_selected_points = inverse_rotate_selected_points(
                    np.round(np.array(p_t_r)[:,0:2], 1), angle_r, clockwise=cw)
                
                points_f = np.round(np.column_stack((original_selected_points, 
                                                     pt_f[:,2])), 2)
            
            elif len(ext_window) > 1:
                points_f = []
                for ext_w in ext_window:
                    p_t_r = []
                    for p_i in ext_w:
                        nearest_point, index = find_nearest_point(
                            p_i, rotated_points_z[:, [0, 2]])
                        p_t_r.append(rotated_points_z[index, :])
                    pt_f = np.column_stack((np.array(ext_w)[:,0], 
                                            np.array(p_t_r)[:,1], 
                                            np.array(ext_w)[:,1]))
                    
                    original_selected_points = inverse_rotate_selected_points(
                        np.round(np.array(p_t_r)[:,0:2], 1), angle_r, clockwise=cw)
                    
                    points_f.append(np.round(np.column_stack(
                        (original_selected_points, pt_f[:,2])), 2))
            all_points.append(points_f)
               
    return all_points
