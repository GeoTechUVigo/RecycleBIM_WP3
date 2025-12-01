# -*- coding: utf-8 -*-

import numpy as np
from src.utils import functions_geom
from src.utils import functions_walls


def find_point_intersection(coef1, coef2):
    """
    Finds the intersection point of two lines given their coefficients in the 
    form of the line equation Ax + By + C = 0.


    Parameters
    ----------
    coef1 : tuple
        A tuple representing the coefficients of the first line.
    coef2 : tuple
        A tuple representing the coefficients of the second line .

    Returns
    -------
    tuple or None
        If the lines intersect, returns a tuple (x, y) with the rounded 
        coordinates of the intersection point.
        If the lines are parallel (determinant is 0), returns None.

    """
    a1, b1, c1 = coef1
    a2, b2, c2 = coef2

    # Calculate the determinant
    det = a1 * b2 - a2 * b1

    # If the determinant is 0, the lines are parallel and there is no intersection
    if det == 0:
        return None
    else:
        # Calculate the x and y coordinates of the intersection point
        x = - (c1 * b2 - c2 * b1) / det
        y = - (a1 * c2 - a2 * c1) / det
        return round(x, 2), round(y, 2)     


def intersect(class_pcd_df, lines_position_t, df_walls_t, b_f, min_p_int, 
              max_dist_int, min_angle_d, max_angle_d, perfect_angle, 
              flat_angle, ex_v):
    """
    Identifies intersection points between walls and classifies them into rooms
    based on adjacency.


    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        DataFrame containing the point cloud data for each room.
    lines_position_t : list
        A list where each element corresponds to a room, containing the list of
        lines that define the boundaries of that room.
    df_walls_t : list
        A list where each element corresponds to a room and contains indices 
        of points that form the walls of that room.
    b_f : float
        The buffer radius used for filtering points within the buffer zone 
        during intersection detection.
    min_p_int : int
        The minimum number of points required to form an intersection between 
        walls before considering them valid.
    max_dist_int : float
        The maximum allowable distance for a point to be considered inside the 
        buffer zone during intersection detection.
    min_angle_d : float
        The minimum angle difference (in radians) between the intersection 
        lines, used to filter out non-relevant intersections.
    max_angle_d : float
        The maximum angle difference (in radians) between the intersection
        lines, used to filter out non-relevant intersections.
    perfect_angle : float
        The target angle (in radians) to which the intersection lines 
        should be close to, for accuracy purposes.
    flat_angle : float
        The angle (in radians) below which an intersection is considered too 
        flat and therefore excluded.
    ex_v : list
        A list of room IDs to be excluded from the intersection analysis.

    Returns
    -------
    points_intersection : list of lists
        A list of lists where each sublist contains the intersection points 
        found within each room.
    room_data_list : list of dicts
        A list of dictionaries where each dictionary contains the adjacency
        relationships and intersection points for each room.
    walls_adjacency : list of lists
        A list of lists where each sublist contains the walls that are adjacent
        to each other for each room.

    """
    points_intersection = []
    room_data_list = []
    walls_adjacency = []
    for r, room in enumerate([valor for valor in range(1, len(
            class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v]):
        room_data = {}
        room_data["id_room"] = room
        print(f"----------------- Room {room} -----------------")
        rec = lines_position_t[r] 
        rpc = df_walls_t[r]    #  points_room[r]
        walls_adjacency_room = []
        # Save walls adjacency, angles and intersection points.                
        list_adj = []
        points_int = []
        buffer_radius = b_f
        while len(points_int) < min_p_int and buffer_radius <= max_dist_int:
            # Verify perpendicularity between all pairs of planes
            i = 0
            while i < len(rec):
                j = i + 1
                while j < len(rec):
                    intersection_point = find_point_intersection(rec[i], rec[j])
                    if intersection_point is not None:
                        # Points inside the buffer
                        points_within_buffer_i = []
                        points_within_buffer_j = []
                        for points_array in np.array(class_pcd_df.iloc[rpc[i]][['x', 'y', 'z']]):
                            if functions_geom.distance((points_array[0], 
                                                        points_array[1]), 
                                                       intersection_point
                                                       ) <= buffer_radius:
                                points_within_buffer_i.append(points_array)
                                
                        for points_array in np.array(
                                class_pcd_df.iloc[rpc[j]][['x', 'y', 'z']]):
                            if functions_geom.distance((points_array[0], 
                                                        points_array[1]), 
                                                       intersection_point
                                                       ) <= buffer_radius:
                                points_within_buffer_j.append(points_array)

                        if len(points_within_buffer_i) > 0 and len(
                                points_within_buffer_i) > 0:
                            if f'Wall {i} is adjacent to Wall {j}' not in list_adj:
                                list_adj.append(f"Wall {i} is adjacent to Wall {j}")
                                walls_adjacency_room.append([i, j])
                                points_int.append(intersection_point)                            
                    j += 1
                i += 1
            
            if len(points_int) < min_p_int:
                # Increase buffer radius by one units
                buffer_radius += 1
            
        points_intersection.append(list(points_int))
                
        room_data["adjacency"] = list(list_adj)
        room_data["intersection_points"] = list(points_int)
        room_data_list.append(room_data)
        walls_adjacency.append(walls_adjacency_room)
        
    return points_intersection, room_data_list, walls_adjacency



#%% With JSON    
def only_intersection_points(points_intersection, z_0):
    """
    Adds a z-coordinate (height) to the intersection points in each room. 

    Parameters
    ----------
   points_intersection : list of lists of tuples
        A list where each element represents a room, and each room contains a 
        list of intersection points.
    z_0 : float
        The z-coordinate (height) to be added to each intersection point, 
        representing the average height of the point cloud.

    Returns
    -------
    modified_points_intersection : list of lists of tuples
        A list where each element represents a room, and each room contains 
        the modified intersection points.
    

    """     
       
    modified_points_intersection = []        
    for room in points_intersection:
        room_m = []
        for point in room:
            # Add the average height at which the point cloud is
            room_m.append(point + (np.round(z_0, 2),))
        
        modified_points_intersection.append(room_m)
    
    return modified_points_intersection
    

def floor_ceiling_plane_json(class_pcd_df, modified_points_intersection, 
                             min_ratio, threshold, iterat):
    """
    Detects the floor and ceiling planes in a point cloud and projects 
    intersection points onto those planes.

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        The point cloud data.
    modified_points_intersection : list of list of tuples
        A list of lists of 3D intersection points (x, y, z) that need to be
        projected onto the floor and ceiling planes.
    min_ratio : float
        The minimum ratio for plane detection. It is used in the plane
        detection algorithm to define the minimum 
        acceptable variance of the points for fitting a plane.
    threshold : float
        The threshold value for plane fitting. It helps control the sensitivity
        of plane detection, influencing how tightly the points fit to the 
        detected planes.
    iterat : int
        The number of iterations for the plane detection process.

    Returns
    -------
    list_points_ceiling : list of list of tuples
        A list where each element is a list of projected points (x, y, z) 
        on the ceiling plane for each room.
    list_points_floor : list of list of tuples
        A list where each element is a list of projected points (x, y, z) 
        on the floor plane for each room.

    """
    
    points_floor = class_pcd_df[class_pcd_df['id_element'] == 0]
    points_ceil = class_pcd_df[class_pcd_df['id_element'] == 1]
    
    floor_plane = functions_walls.DetectMultiPlanes(
        np.array(points_floor[['x','y','z']]), min_ratio, threshold, 
        iterations=iterat)
    ceil_plane = functions_walls.DetectMultiPlanes(
        np.array(points_ceil[['x','y','z']]), min_ratio, threshold, 
        iterations=iterat)

    list_points_floor = []
    list_points_ceiling = []
    
    for r in modified_points_intersection:
        list_points_floor_r = []
        list_points_ceiling_r = []
        for point in r:
            # Coefficients of the general floor plane equation
            coefficients_floor = floor_plane[0][0]        
            # Coefficients of the general ceiling plane equation
            coefficients_ceiling = ceil_plane[0][0]
            # Normal of floor plane
            normal_vector_floor = coefficients_floor[:3]        
            # Normal of ceiling plane
            normal_vector_ceiling = coefficients_ceiling[:3]
            # Calculate the distance of the point from the floor plane
            distance_to_floor_plane = np.dot(normal_vector_floor, point) + coefficients_floor[3]        
            # Calculate the distance of the point from the ceiling plane
            distance_to_ceiling_plane = np.dot(normal_vector_ceiling, point) + coefficients_ceiling[3]
            # Calculate the projection of the point on the floor plane
            projection_point_floor = point - distance_to_floor_plane * normal_vector_floor
            list_points_floor_r.append(np.round(projection_point_floor, 2))
            # Calculate the projection of the point on the ceiling plane
            projection_point_ceiling = point - distance_to_ceiling_plane * normal_vector_ceiling
            list_points_ceiling_r.append(np.round(projection_point_ceiling, 2))
        
        list_points_floor.append(list_points_floor_r)
        list_points_ceiling.append(list_points_ceiling_r)
          
    return list_points_ceiling, list_points_floor
    