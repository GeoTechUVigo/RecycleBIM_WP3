# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def points_boundary_floor(list_points_floor, boundary_pol, dist_bound):
    """
    Identify and order the points from a set of floor points that are within a 
    specified distance from the boundary of a polygon. The points are sorted 
    based on their proximity to the boundary.

    Parameters
    ----------
    list_points_floor : list of np.ndarray
        A list of 2D points representing the coordinates of the floor.
    boundary_pol : shapely.geometry.Polygon
        A Polygon object representing the boundary of the area. 
    dist_bound : float
        The maximum distance within which the floor points are considered close
        to the boundary. 

    Returns
    -------
    ordered_close_points : np.ndarray
        An array of shape (m, 2) containing the ordered points that are within 
        the specified distance to the boundary, sorted based on their proximity 
        to the boundary.

    """
    points_floor_all = np.concatenate(list_points_floor)
    boundary_coords = np.array(boundary_pol.boundary.coords)
        
    # We create the KD tree of the boundary points
    boundary_tree = KDTree(boundary_coords)
    
    # We look for the points in points_floor_all that are close to boundary_coords
    distances, indices = boundary_tree.query(points_floor_all[:, :2], distance_upper_bound=dist_bound)
    
    # We filter the points that are within the limit
    close_points = points_floor_all[distances < dist_bound]
    
    boundary_tree = KDTree(boundary_coords)
    
    # Find the closest point in boundary_coords for each point in close_points
    distances, indices = boundary_tree.query(close_points[:, :2])
    
    # Sort close_points based on boundary_coords indices
    ordered_close_points = close_points[np.argsort(indices)]
    
    # Representing points and lines
    # plt.figure(figsize=(10, 8))
    
    # # Plot the points
    # plt.scatter(ordered_close_points[:, 0], ordered_close_points[:, 1], color='red')
    
    # # Join the selected points
    # for i in range(len(ordered_close_points) - 1):
    #     plt.plot(ordered_close_points[i:i+2, 0], ordered_close_points[i:i+2, 1], color='blue')
    
    # # Join the last point with the first
    # plt.plot([ordered_close_points[0, 0], ordered_close_points[-1, 0]], 
    #          [ordered_close_points[0, 1], ordered_close_points[-1, 1]], color='blue')
    
    # # Chart Configuration
    # plt.title('Order points floor')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid()
    # plt.axis('equal')  
    # plt.show()

    return ordered_close_points


def points_boundary_ceiling(list_points_ceiling, boundary_pol, dist_bound):
    """
    Identify and order the points from a set of ceiling points that are within
    a specified distance from the boundary of a polygon. 


    Parameters
    ----------
    list_points_ceiling : list of np.ndarray
        A list of 2D points representing the coordinates of the ceiling. 
    boundary_pol : shapely.geometry.Polygon
        A Polygon object representing the boundary of the area. 
    dist_bound : float
        The maximum distance within which the ceiling points are considered 
        close to the boundary. 

    Returns
    -------
    ordered_close_points : np.ndarray
        An array of shape (m, 2) containing the ordered points that are within 
        the specified distance to the boundary, sorted based on their proximity
        to the boundary. These points represent the ordered sequence of 
        points along the boundary.
        
    """
    points_ceiling_all = np.concatenate(list_points_ceiling)
    boundary_coords = np.array(boundary_pol.boundary.coords)
    
    # We create the KD tree of the boundary points
    boundary_tree = KDTree(boundary_coords)
    
    # We look for the points in points_floor_all that are close to boundary_coords
    distances, indices = boundary_tree.query(points_ceiling_all[:, :2], distance_upper_bound=dist_bound)
    
    # We filter the points that are within the limit
    close_points = points_ceiling_all[distances < dist_bound]
    
    boundary_tree = KDTree(boundary_coords)
    
    # Find the closest point in boundary_coords for each point in close_points
    distances, indices = boundary_tree.query(close_points[:, :2])
    
    # Sort close_points based on boundary_coords indices
    ordered_close_points = close_points[np.argsort(indices)]

    # # Representing points and lines
    # plt.figure(figsize=(10, 8))
    
    # # Plot the points
    # plt.scatter(ordered_close_points[:, 0], ordered_close_points[:, 1], color='red')
    
    # # Join the selected points
    # for i in range(len(ordered_close_points) - 1):
    #     plt.plot(ordered_close_points[i:i+2, 0], ordered_close_points[i:i+2, 1], color='blue')
    
    # # Join the last point with the first
    # plt.plot([ordered_close_points[0, 0], ordered_close_points[-1, 0]], 
    #          [ordered_close_points[0, 1], ordered_close_points[-1, 1]], color='blue')
    
    # # Chart Configuration
    # plt.title('Order points ceiling')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid()
    # plt.axis('equal') 
    # plt.show()

    return ordered_close_points
    