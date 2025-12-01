# -*- coding: utf-8 -*-

import numpy as np


def detect_plane(points, th):
    """
    Detects the plane (XZ or YZ) that a set of points lies on, based on their 
    geometric spread.


    Parameters
    ----------
    points : numpy.ndarray
        A 2D array of shape (n, 3) representing a set of 3D points where each point is specified by 
        its (X, Y, Z) coordinates.
    th : float
        A threshold value to determine whether the variation in the X or Y coordinates is negligible. 

    Returns
    -------
    str
        Returns 'XZ' if the points lie on the XZ plane (i.e., the variation in Y is smaller than the threshold),
        'YZ' if the points lie on the YZ plane.
    

    """
    dif_y = np.ptp(points[:, 1])  
    dif_x = np.ptp(points[:, 0])  
    
    if dif_y < th:
        return 'XZ'  
    elif dif_x < th:
        return 'YZ'  
    else:
        raise ValueError("The wall is clearly not in the XZ or YZ plane.")

def sort_YZ(array):
    """
    Sorts a set of 3D points based on their Y and Z coordinates, organizing them into a specific order.


    Parameters
    ----------
    array : numpy.ndarray
        A 2D array of shape (n, 3) where each row represents a 3D point (X, Y, Z). The sorting is 
        done based on the Y and Z coordinates of these points.

    Returns
    -------
    o_d : list
        A list of four points in a specific order based on the minimum and maximum Y and Z values.

    """
    
    max_y = np.max(array[:, 1])
    max_z = np.max(array[:, 2])
    min_y = np.min(array[:, 1])
    min_z = np.min(array[:, 2])
    
    row_with_min_y_and_max_z = array[(array[:, 1] == min_y) & (
        array[:, 2] == max_z)][0]
    row_with_max_y_and_max_z = array[(array[:, 1] == max_y) & (
        array[:, 2] == max_z)][0]
    row_with_max_y_and_min_z = array[(array[:, 1] == max_y) & (
        array[:, 2] == min_z)][0]
    row_with_min_y_and_min_z = array[(array[:, 1] == min_y) & (
        array[:, 2] == min_z)][0]
    
    o_d = [row_with_min_y_and_max_z, row_with_max_y_and_max_z, 
           row_with_max_y_and_min_z, row_with_min_y_and_min_z]
    
    return o_d

def sort_XZ(array):
    """
    Sorts a set of 3D points based on their X and Z coordinates, organizing them into a specific order.


    Parameters
    ----------
    array : numpy.ndarray
        A 2D array of shape (n, 3) where each row represents a 3D point (X, Y, Z). The sorting is 
        done based on the X and Z coordinates of these points.

    Returns
    -------
    o_d : list
        A list of four points in a specific order based on the minimum and maximum X and Z values.

    """
    
    max_x = np.max(array[:, 0])
    max_z = np.max(array[:, 2])
    min_x = np.min(array[:, 0])
    min_z = np.min(array[:, 2])
    
    row_with_min_x_and_max_z = array[(array[:, 0] == min_x) & (
        array[:, 2] == max_z)][0]
    row_with_max_x_and_max_z = array[(array[:, 0] == max_x) & (
        array[:, 2] == max_z)][0]
    row_with_max_x_and_min_z = array[(array[:, 0] == max_x) & (
        array[:, 2] == min_z)][0]
    row_with_min_x_and_min_z = array[(array[:, 0] == min_x) & (
        array[:, 2] == min_z)][0]
    
    o_d = [row_with_min_x_and_max_z, row_with_max_x_and_max_z, 
           row_with_max_x_and_min_z, row_with_min_x_and_min_z]
    
    return o_d


def save_doors_BIM(doors_final, doors_non_intersecting, th):
    """
    Processes and sorts doors based on their plane (XZ or YZ), and organizes 
    them in a specific order.


    Parameters
    ----------
    doors_final : list of lists of tuples
        A list containing doors that have been finalized for processing. Each door is represented as a 
        list of points (X, Y, Z) defining its geometry.
    doors_non_intersecting : list of lists of tuples
        A list containing doors that do not intersect with other geometry. Each door is represented 
        as a list of points (X, Y, Z).
    th : float
        A threshold value used in the `detect_plane` function to determine if a door lies in the XZ or YZ plane.

    Returns
    -------
    order_doors : list
        A list of sorted doors, where each door is represented as a list of four points, sorted based on 
        their YZ or XZ coordinates.

    """
    doors = doors_final + doors_non_intersecting

    order_doors = []
    for d in doors:
        # print(detect_plane(d, th))
        if detect_plane(d, th) == 'YZ':
            o_d = sort_YZ(d)
            order_doors.append(o_d)
            
        elif detect_plane(d, th) == 'XZ':
            o_d = sort_XZ(d)
            order_doors.append(o_d)
            
    return order_doors
            
def is_door_on_wall(door_pts, wall_pts):
    """
    Determines if a door is placed on a wall based on its coordinates

    Parameters
    ----------
    door_pts : numpy.ndarray
        A 2D array where each row represents a point (X, Y, Z) defining the geometry of the door. 
        Only the X and Y coordinates are considered for this check.
    wall_pts : numpy.ndarray
        A 2D array where each row represents a point (X, Y, Z) defining the geometry of the wall. 
        The function uses the minimum and maximum X and Y coordinates of the wall points to check 
        if the door lies within these bounds.

    Returns
    -------
    bool
        `True` if all door points are within the X and Y ranges of the wall, otherwise `False`.
  

    """
    # Get the X and Y ranges of the wall
    x_min, x_max = np.min(wall_pts[:, 0]), np.max(wall_pts[:, 0])
    y_min, y_max = np.min(wall_pts[:, 1]), np.max(wall_pts[:, 1])
    
    # Check if the door points are within the X and Y ranges of the wall
    for pt in door_pts:
        x, y = pt[0], pt[1]
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False
    return True            
            
#%%     
  
def windows(p_windows, th_g, th_w, s_w, s_h):
    """
    Processes the window geometries and calculates their positions and dimensions for BIM modeling.


    Parameters
    ----------
    p_windows : list of numpy.ndarray
        A list of numpy arrays, each containing the coordinates (X, Y, Z) of a window. The window can be
        either in the XZ or YZ plane. The function adjusts these coordinates based on the plane detection.
    th_g : float
        A threshold value used for determining the plane type ('XZ' or 'YZ'). The `detect_plane` function compares
        the window's coordinates with this threshold to classify the window's plane.
    th_w : float
        Another threshold value used to further refine the window placement classification. This is also used 
        by `detect_plane` to determine the window's plane (XZ or YZ).
    s_w : float
        The half-width of the window. This value is used to calculate the window's spatial extent along the width 
        axis, helping define the window's boundaries.
    s_h : float
        The height of the window. This value is used to define the vertical extent of the window and the height 
        at which the window will be placed.

    Returns
    -------
    n_w : list of numpy.ndarray
        A list of numpy arrays, each containing the coordinates of the windows, including both the start and 
        end points for each window based on their geometry and position. Each window's coordinates are adjusted 
        based on the width and height provided.


    """
    for w in p_windows:
        if detect_plane(w, th_g) == 'XZ':
            
            mean_first_col = np.mean(w[:, 1])
            
            w[:, 1] = mean_first_col
            
        elif detect_plane(w, th_g) == 'YZ':
            mean_first_col = np.mean(w[:, 0])
            
            w[:, 0] = mean_first_col
        
    n_w = []        
    for w in p_windows:
        center = np.mean(w, axis=0)
        if detect_plane(w, th_w) == 'YZ':
            y_centro = center[1]
            x_centro = center[0]  
            z_centro = center[2]
            # Define the ends of the door
            puerta = np.array([[x_centro, y_centro - s_w],  
                                [x_centro, y_centro + s_w]])
            
        else:
            x_centro = center[0]
            y_centro = center[1]
            z_centro = center[2]
            # Define the ends of the door
            puerta = np.array([[x_centro - s_w, y_centro],  
                                [x_centro + s_w, y_centro]])
            
        w_s = np.array([[x, y, z_centro - s_h] for x, y in puerta])
        w_t = np.array([[x, y, z_centro + s_h] for x, y in puerta])
        
        n_w.append(np.round(np.vstack((w_s, w_t)), 2))    
    
    return n_w
        


def final_points_windows(all_points, class_pcd_df, h_f_w, th_g, th_w, s_w, s_h):
    """
    Detects the points that correspond to windows based on their elevation and
    specified thresholds.

    Parameters
    ----------
    all_points : list
        A list containing points, which can be either arrays or nested lists 
        representing 3D coordinates.
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data, with an 'id_element' column 
        that classifies the elements.
    h_f_w : float
        A threshold used to filter the height of the points to ensure they 
        lie within the window height range.
    th_g : float
        A threshold for grouping points based on geometric proximity.
    th_w : float
        A threshold for the window's width to filter the window-like structures.
    s_w : float
        The size of the window in the horizontal direction.
    s_h : float
        The size of the window in the vertical direction.

    Returns
    -------
    p_windows : list
        A list of points that correspond to windows, based on the filtering and
        conditions provided.

    """
    flattened_points = []

    for item in all_points:
        if isinstance(item, np.ndarray):
            flattened_points.append(item)  
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, np.ndarray):
                    flattened_points.append(sub_item)  

    all_points = flattened_points

    min_v = np.min(class_pcd_df[class_pcd_df['id_element']==2][['z']]) + h_f_w
    max_v = np.max(class_pcd_df[class_pcd_df['id_element']==2][['z']]) - h_f_w


    n_w = windows(all_points, th_g, th_w, s_w, s_h)

    p_windows = []
    for a in n_w:
        result = np.all((a[:,2] >= min_v) & (a[:,2] <= max_v))
        if result:
            p_windows.append(a)

    return p_windows

    
def save_windows_BIM(p_windows, th_w):
    """
    Processes a list of windows and classifies them into different planes (XZ or YZ).


    Parameters
    ----------
    p_windows : list of numpy.ndarray
       A list of numpy arrays, each representing the coordinates of a window in 3D space. 
       Each window can be in either the XZ or YZ plane. The function classifies and processes 
       the windows based on their plane.
   th_w : float
       A threshold value used to help detect whether the window lies in the XZ or YZ plane. 
       This threshold is used by the `detect_plane` function to classify the window's orientation.

   Returns
   -------
   order_windows : list of list of numpy.ndarray
       A list containing the windows sorted by their plane and orientation. Each window is sorted 
       based on its coordinates in the detected plane (XZ or YZ).

    """
    order_windows = []
    for w in p_windows:
        if detect_plane(w, th_w) == 'YZ':
            # print(detect_plane(w, th_w))
            o_w = sort_YZ(w)
            order_windows.append(o_w)
            
        elif detect_plane(w, th_w) == 'XZ':
            # print(detect_plane(w, th_w))
            o_w = sort_XZ(w)
            order_windows.append(o_w)

            
    return order_windows            
            
            
            

            
            
            
            
            
        