# -*- coding: utf-8 -*-

import numpy as np

def clockwise_sort(points):
    """
    Sort a set of 2D points in clockwise order around their centroid.


    Parameters
    ----------
    points : np.ndarray
        A 2D array of shape (n, 2), where n is the number of points, and each 
        row represents a point with (x, y) coordinates.

    Returns
    -------
    sorted_points : np.ndarray
        A 2D array of the same shape as the input `points`, but with the points
        sorted in clockwise order around their centroid. Each row corresponds 
        to a sorted point with (x, y) coordinates.

    """
    center = np.mean(points, axis=0)

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    return sorted_points

def points_room_order(points_intersection, path_walls):
    """
    Order points of intersection for each room in clockwise direction.


    Parameters
    ----------
    points_intersection : list of list of tuple
        A list where each element corresponds to a room and contains a list 
        of intersection points (x, y) for that room. The intersection points 
        represent the vertices of walls in the room.
    path_walls : str
        This parameter is included in the function signature but is unused in 
        the implementation. It might be intended for future functionality or 
        for additional processing logic outside the provided code.

    Returns
    -------
    points_f : list of list of list
        A list where each element corresponds to a room and contains a sorted 
        list of intersection points in clockwise order. The points are 
        represented as (x, y) coordinates.

    """
    points_f = []
    for r in points_intersection:
        pts = []
        if len(r) != 0:
            for i in r:
                pts.append([i[0], i[1]])
        
            sorted_points = clockwise_sort(np.array(pts))
            points_f.append(sorted_points[::-1].tolist())
    
        else:
            points_f.append(r)
    
    return points_f

        

def evaluate_point_on_line(point, line, tolerance=5e-1):
    """
    Check if a given point lies on a line within a specified tolerance.

    Parameters
    ----------
    point : tuple of float
        A tuple representing the point's coordinates (x, y) that needs to be 
        evaluated.
    line : tuple of float
        A tuple representing the coefficients of the line equation (A, B, C) in 
        the general form `Ax + By + C = 0`.
    tolerance : float, optional
        The maximum allowed difference between the evaluation of the line equation 
        and zero for the point to be considered on the line. The default value is `5e-1`.

    Returns
    -------
    bool
        `True` if the point lies on the line within the specified tolerance, 
        `False` otherwise.

    """
    x, y = point
    A, B, C = line
    return abs(A * x + B * y + C) < tolerance

def ordered_walls(class_pcd_df, ex_v, lines_position_t, points_f, df_walls_t, 
                  path_walls, list_points_ceiling, list_points_floor):
    """
    Organizes and orders walls, ceiling points, and floor points based on their 
    positions and intersections with lines in the room.

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        DataFrame containing point cloud data with room information.
    ex_v : list of int
        List of room IDs that should be excluded from the processing.
    lines_position_t : list of list of tuples
        A list of room lines, where each line is represented by a tuple 
        (A, B, C) for the line equation `Ax + By + C = 0`.
    points_f : list of list of tuples
        A list of points for each room, represented as (x, y) coordinates of 
        the intersection points.
    df_walls_t : list of list
        List of indices that correspond to walls in each room, representing 
        the walls' relationships 
        with the room's points.
    path_walls : list of paths
        List containing paths that define the walls for each room (not used directly in the function body 
        but could be used for additional analysis).
    list_points_ceiling : list of lists of tuples
        A list of ceiling points for each room.
    list_points_floor : list of lists of tuples
        A list of floor points for each room.
    
    Returns
    -------
    info_results : list of dicts
        A list containing dictionaries, each corresponding to a room with the following structure:
        - "IdRoom": The room ID (integer).
        - "OrderIdWalls": List of wall indices sorted based on their positions.
        - "Points_xy": A dictionary containing the points for each wall in the room.
        - "Points_ceiling": A dictionary containing ceiling points for each wall in the room.
        - "Points_floor": A dictionary containing floor points for each wall in the room.

    """
    
    v = [valor for valor in range(1, len(class_pcd_df.groupby('id_room')
                                         ) + 1) if valor not in ex_v]
    
    info_results = []
    room_results = []
    for id_room, (line_set, points_set, walls_set, points_set_c, points_set_f
                  ) in enumerate(zip(lines_position_t, points_f, df_walls_t, 
                                     list_points_ceiling, list_points_floor)): 
        room_results = {"IdRoom": v[id_room], 
                        "OrderIdWalls": [], 
                        "Points_xy": [], 
                        "Points_ceiling": [], 
                        "Points_floor": []}
        # Verificación de pares de puntos para cada recta
        res = {}
        res_c = {}
        res_f = {}
        
        order = []
        # Se crean los pares de cada par de puntos
        num_points = len(points_set)
        pares = [(i, (i + 1) % num_points) for i in range(num_points)]
        
        
        for i, (p1, p2) in enumerate(pares):
            res[f'Wall {i+1}'] = []
            res_c[f'Wall {i+1}'] = []
            res_f[f'Wall {i+1}'] = []
            
            for j, recta in enumerate(line_set):
                if evaluate_point_on_line(points_set[p1], recta
                                          ) and evaluate_point_on_line(
                                              points_set[p2], recta):
                    order.append(j)
                    res[f'Wall {i+1}'].append([points_set[p1], points_set[p2]])
                    res_c[f'Wall {i+1}'].append([points_set_c[p1], points_set_c[p2]])
                    res_f[f'Wall {i+1}'].append([points_set_f[p1], points_set_f[p2]])
        
        room_results["OrderIdWalls"].append(order)
        room_results["Points_xy"].append(res) 
        room_results["Points_ceiling"].append(res_c)
        room_results["Points_floor"].append(res_f)
        
        
        info_results.append(room_results)    
    

    return info_results    

























