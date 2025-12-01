# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial import ConvexHull
import math
from src.utils import functions_geom
from src.utils import functions_walls
import matplotlib.pyplot as plt
from src.info_elements import points_boundary_floor_ceiling
from statistics import mean
from src.info_elements import infoBIM_doors_windows


def clockwise_sort_xy(points):
    """
    Sorts a list of 2D points in clockwise order around their centroid.

    Parameters
    ----------
    points : list of tuple or array-like
        A list or array of 2D points (x, y) to be sorted. 

    Returns
    -------
    sorted_points : ndarray
        An array of the input points sorted in clockwise order around the centroid.

    """
    points = np.array(points)
    # Calculate the midpoint of the points
    center = np.mean(points, axis=0)

    # Calculate the polar angles of the points with respect to the midpoint
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Order the points according to their angles
    sorted_indices = np.argsort(angles)[::-1]
    sorted_points = points[sorted_indices]

    return sorted_points

# External walls
def is_exterior(class_pcd_df, boundary_pol, lines_position_t, df_walls_t, 
                equation_df_walls_t, ex_v, dist_max, distancia_umbral=0.15):
    """
    Identifies which walls of a building are exterior walls based on their 
    proximity to a boundary polygon.

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data.
    boundary_pol : shapely.geometry.Polygon
        A Shapely Polygon object representing the boundary of the building or area.
    lines_position_t : list of lists
        A list of lists, where each sublist represents a wall.
    df_walls_t : list of lists
        A list of walls, each represented by a list of point indices referring 
        to the walls in the point cloud data.
    equation_df_walls_t : list
        A list containing the equations or parameters of the walls, 
        which might not be directly used in this function.
    ex_v : list of int
        A list of room IDs that are considered to be exterior rooms and 
        should be excluded from interior wall checks.
    dist_max : float
        The maximum allowed distance to classify a point as being near the boundary.
    distancia_umbral : float, optional
        A threshold distance for considering a point close to the boundary
        of the polygon. Default is 0.15.

    Returns
    -------
    information : list of dicts
        A list of dictionaries containing the room ID, the IDs of all walls, 
        the IDs of exterior walls, and the IDs of interior walls.


    """
    information = []
    walls_exteriors = []
    v = [valor for valor in range(1, len(class_pcd_df.groupby('id_room')) + 1
                                  ) if valor not in ex_v]
    # Check if each point in the point cloud is on the edge of the polygon
    for ind_i, i in enumerate(df_walls_t):  
        data = {}
        wall_exterior = []
        wall_interior = []
        walls = []
        for ind_j, j in enumerate(i):
            walls.append(ind_j)
            points_in_border = []
            for point in class_pcd_df.iloc[j][['x','y']].values:
                point_shapely = Point(point)  
                polygon_border = boundary_pol.boundary
                dist = polygon_border.distance(point_shapely)                
                if dist < distancia_umbral:
                    points_in_border.append(point)
            
            # If it is greater than one tenth of the point cloud on the wall
            if len(points_in_border) > (1/3)*len(df_walls_t[ind_i][ind_j]):
                walls_exteriors.append({'IdRoom': v[ind_i], 
                                        'IdWallExterior': ind_j})
                wall_exterior.append(ind_j)
            else:
                wall_interior.append(ind_j)
        
        data['IdRoom'] = v[ind_i]        
        data['IdWalls'] = walls
        data['IdWallsExterior'] = wall_exterior
        data['IdWallsInterior'] = wall_interior
        information.append(data)
        
    return information

    
#%% Floor and Ceiling
def floor_ceiling_BIM(list_points_ceiling, list_points_floor, boundary_pol, 
                      dist_bound):
    """
    Classifies and extracts the points that are close to the boundary for both 
    the floor and ceiling in a BIM.

    Parameters
    ----------
    list_points_ceiling : list of tuples
        A list of 3D points representing the positions of points on the ceiling.
    list_points_floor : list of tuples
        A list of 3D points representing the positions of points on the floor.
    boundary_pol : shapely.geometry.Polygon
        A Shapely Polygon object representing the boundary of the building or area.
    dist_bound : float
        The distance threshold within which points are considered to be near the boundary.


    Returns
    -------
    list
        Intormation about the points of the ceiling and the floor.

    """
    points_floor = points_boundary_floor_ceiling.points_boundary_floor(
        list_points_floor, boundary_pol, dist_bound)
    points_ceiling = points_boundary_floor_ceiling.points_boundary_ceiling(
        list_points_ceiling, boundary_pol, dist_bound)
    
    flattened_points_floor = [
        coord for point in points_floor for coord in point]
    flattened_points_ceiling = [
        coord for point in points_ceiling for coord in point]
    
    return [points_floor, points_ceiling, flattened_points_floor, 
            flattened_points_ceiling]
    

#%% Exterior walls

def find_point_intersection(coeficientes1, coeficientes2):
    """
    Finds the intersection point of two lines given their coefficients.

    Parameters
    ----------
    coeficientes1 : tuple of float
        A tuple containing the coefficients (a1, b1, c1) of the first line's equation.
    coeficientes2 : tuple of float
        A tuple containing the coefficients (a2, b2, c2) of the second line's equation.

    Returns
    -------
    tuple of float or None
        The coordinates (x, y) of the intersection point rounded to 2 decimal places, 
        or `None` if the lines are parallel (no intersection).

    """
    a1, b1, c1 = coeficientes1
    a2, b2, c2 = coeficientes2

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

def info_walls(class_pcd_df, ex_v, lines_position_t, df_walls_t, b_f, 
               min_p_int, max_dist_int):
    """
    This function computes and returns the adjacency and intersection information 
    for walls in a 3D point cloud dataset, while considering certain geometric properties.

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        DataFrame containing the 3D coordinates of the points in the point cloud. 
    ex_v : list of int
        List of room IDs to exclude from the analysis 
    lines_position_t : list of lists
        A list where each element corresponds to a room and contains the walls 
        of that room, each represented by a line segment defined by a pair 
        of coordinates.
    df_walls_t : list of lists
        A list where each element corresponds to a room and contains indices
        referencing walls in that room.
    b_f : float
        The initial buffer radius. This value determines how far the intersection points
        can be from the wall segments.
    min_p_int : int
        The minimum number of intersection points required for each room. 
        The function iterates over increasing buffer radio until this number 
        of intersections is found.
    max_dist_int : float
        The maximum distance for which an intersection 
        point can be considered valid. If no valid intersection points are 
        found within this distance range, the process stops.

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
        
        wall_intersections = {i: set() for i in range(len(rec))}
        
        list_adj = []
        points_int = []
        buffer_radius = b_f
        
        while len(points_int) < min_p_int and buffer_radius <= max_dist_int:
            i = 0
            while i < len(rec):
                j = i + 1
                while j < len(rec):
                    intersection_point = find_point_intersection(rec[i], rec[j])
                    
                    if intersection_point is not None:
                        # Points inside the buffer
                        points_within_buffer_i = []
                        points_within_buffer_j = []
                        
                        for points_array in np.array(class_pcd_df.iloc[
                                rpc[i]][['x', 'y', 'z']]):
                            if functions_geom.distance((points_array[0], 
                                                        points_array[1]), 
                                                       intersection_point
                                                       ) <= buffer_radius:
                                points_within_buffer_i.append(points_array)
                                wall_intersections[i].add(intersection_point)
                        
                        for points_array in np.array(class_pcd_df.iloc[
                                rpc[j]][['x', 'y', 'z']]):
                            if functions_geom.distance((points_array[0], 
                                                        points_array[1]), 
                                                       intersection_point
                                                       ) <= buffer_radius:
                                points_within_buffer_j.append(points_array)
                                wall_intersections[j].add(intersection_point)
    
                        if len(points_within_buffer_i) > 0 and len(
                                points_within_buffer_j) > 0:
                            if f'Wall {i} is adjacent to Wall {j}' not in list_adj:
                                list_adj.append(f"Wall {i} is adjacent to Wall {j}")
                                walls_adjacency_room.append([i, j])
                                points_int.append(intersection_point)                            
                    j += 1
                i += 1
                
            if len(points_int) < min_p_int:
                # Increase the buffer radius by one unit
                buffer_radius += 1
                
        points_intersection.append(list(points_int))
                    
        room_data["adjacency"] = list(list_adj)
        room_data["intersection_points"] = list(points_int)
        
        # Convert intersection sets to lists before adding them to room_data
        room_data["wall_intersections"] = {k: list(v) for k, v in wall_intersections.items()}
        
        room_data_list.append(room_data)
        walls_adjacency.append(walls_adjacency_room)
        
    return room_data_list

# Function to project a point onto a plane
def project_point_on_plane_advanced(point, coefficients, normal_vector):
    """
    Projects a given point onto a plane defined by the plane's coefficients and 
    normal vector

    Parameters
    ----------
    point : numpy.ndarray
        A 1D array or list representing the coordinates of the point to be projected.
    coefficients : tuple or list of floats
        A list or tuple of four values representing the coefficients [A, B, C, D] of 
        the plane equation Ax + By + Cz + D = 0.
    normal_vector : numpy.ndarray
        A 1D array or list representing the normal vector to the plane, which 
        is a vector perpendicular to the plane.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the projected point's coordinates 
        [x_proj, y_proj, z_proj], rounded to two decimal places.


    """
    A, B, C, D = coefficients
    # Distance from point to plane
    distance_to_plane = np.dot(normal_vector, point) + D
    # Projection of the point on the plane
    projection_point = point - distance_to_plane * np.array(normal_vector)
    return np.round(projection_point, 2)


def info_walls_exterior(room_data_list, class_pcd_df, z_0, min_ratio, 
                        threshold, iterat):
    """
    Processes the intersection points of walls in each room, projects them onto 
    the floor and ceiling planes, and adds the centroid for each room

    Parameters
    ----------
    room_data_list : list of dicts
        A list of room data, each containing intersection points for walls.
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data with room IDs, element IDs, and
        3D coordinates.
    z_0 : float
        The base height (z-coordinate) for projecting intersection points onto
        the floor and ceiling.
    min_ratio : float
        The minimum ratio used for plane detection during floor and ceiling 
        plane estimation.
    threshold : float
        The threshold for plane detection accuracy in floor and ceiling 
        plane estimation.
    iterat : int
        The number of iterations for the plane detection algorithm.

    Returns
    -------
    walls_with_intersections : list of dicts
        A list of rooms with their respective wall intersection points
        projected onto the floor and ceiling planes, and the centroids of 
        each room.

    """
    # Print the intersection points per wall for each room
    for room_data in room_data_list:
        print(f"Room ID: {room_data['id_room']}")
        for wall_id, intersections in room_data["wall_intersections"].items():
            print(f"Wall {wall_id} Intersection Points: {intersections}")
    
    # Iterate over each room in room_data_list
    for room in room_data_list:
        # Add z to intersection points
        room['intersection_points'] = [(x, y, round(z_0,2)) for (x, y) in room[
            'intersection_points']]
        
        # Add z to wall intersection points
        for wall_id, intersections in room['wall_intersections'].items():
            room['wall_intersections'][wall_id] = [(x, y, round(z_0, 2)) for (
                x, y) in intersections]
    
    # Print the result to verify
    for room in room_data_list:
        print(f"Room ID: {room['id_room']}")
        print("Intersection Points:", room['intersection_points'])
        print("Wall Intersections:", room['wall_intersections'])
       
    points_floor = class_pcd_df[class_pcd_df['id_element'] == 0]
    points_ceil = class_pcd_df[class_pcd_df['id_element'] == 1]
    floor_plane = functions_walls.DetectMultiPlanes(np.array(
        points_floor[['x','y','z']]), min_ratio, threshold, iterations=iterat)
    ceil_plane = functions_walls.DetectMultiPlanes(np.array(
        points_ceil[['x','y','z']]), min_ratio, threshold, iterations=iterat)
    
    # Floor plane coefficients
    coefficients_floor = floor_plane[0][0]  
    normal_vector_floor = coefficients_floor[:3]  
    
    # Ceiling plane coefficients
    coefficients_ceiling = ceil_plane[0][0]  
    normal_vector_ceiling = coefficients_ceiling[:3] 
    
    
    # Projects the intersection points onto the floor and ceiling
    for room_data in room_data_list:
        wall_intersection_ceiling = {}
        wall_intersection_floor = {}
        
        for wall_id, points in room_data['wall_intersections'].items():
            # Project points onto the ceiling and floor
            projected_ceiling_points = []
            projected_floor_points = []
            
            for point in points:
                # Add the Z_0 (initial) coordinate
                point_3d = np.array([point[0], point[1], z_0])
                
                # Project onto the floor
                projection_point_floor = project_point_on_plane_advanced(
                    point_3d, coefficients_floor, normal_vector_floor)
                projected_floor_points.append(projection_point_floor)
                
                # Project onto the ceiling
                projection_point_ceiling = project_point_on_plane_advanced(
                    point_3d, coefficients_ceiling, normal_vector_ceiling)
                projected_ceiling_points.append(projection_point_ceiling)
            
            # Store projections on each wall
            wall_intersection_ceiling[wall_id] = projected_ceiling_points
            wall_intersection_floor[wall_id] = projected_floor_points
        
        # Add the projected data to the room dictionary
        room_data['wall_intersection_ceiling'] = wall_intersection_ceiling
        room_data['wall_intersection_floor'] = wall_intersection_floor
       
    
    # For each wall extract the points of the ceiling and floor
    # Ready to store results by room
    walls_with_intersections = []
    
    # Iterate over each room in room_data_list
    for room in room_data_list:
        id_room = room['id_room']
        # Dictionary for storing walls with their combined points (ceiling + floor)
        room_walls_intersections = {'id_room': id_room, 'walls': []}
        
        # Iterate over each wall 
        for wall_id in room['wall_intersection_ceiling'].keys():
            # Get the ceiling and floor points
            ceiling_points = room['wall_intersection_ceiling'][wall_id]
            floor_points = room['wall_intersection_floor'][wall_id]
            
            # Combine ceiling and floor points into a single list
            points_walls = ceiling_points + floor_points
            
            # Save points in the proper format
            wall_info = {
                'wall_id': wall_id,
                'points_ceiling': ceiling_points,
                'points_floor': floor_points,
                'points_walls': points_walls
            }
            
            # Add wall information to the room wall list
            room_walls_intersections['walls'].append(wall_info)
        
        # Add room details to the main list
        walls_with_intersections.append(room_walls_intersections)
    
    
    # Group by room_id and calculate centroid (average of x, y, z)
    centroides = class_pcd_df.groupby('id_room')[
        ['x', 'y', 'z']].mean().reset_index()
    
    for room in walls_with_intersections:
        id_room = room['id_room']
        # Find the centroid of the current room_id
        centroide = centroides[centroides['id_room'] == id_room][
            ['x', 'y', 'z']].values[0]
        # Adding the centroid to the room dictionary
        room['centroid'] = np.round(centroide, 2)
        
    return walls_with_intersections
    
    

# Detect on which plane the wall is with a threshold 
def detect_plane(puntos, umbral=0.1):
    """
    Detects whether a set of points lies primarily in the XZ or YZ plane based
    on the range of their coordinates.

    Parameters
    ----------
    puntos : numpy.ndarray
        A 2D array of points where each row represents a point with at least
        x and y coordinates.
    umbral : float, optional
        A threshold for the maximum allowable difference between the coordinates. 
        If the difference in one coordinate is smaller than this threshold, 
        it is considered negligible. 
        The default is 0.1.

    Returns
    -------
    str
        Returns 'XZ' if the points lie predominantly in the XZ plane, 
        or 'YZ' if the points lie predominantly in the YZ plane.

    """
    dif_y = np.ptp(puntos[:, 1])  
    dif_x = np.ptp(puntos[:, 0])  
    
    if dif_y < umbral:
        return 'XZ'  
    elif dif_x < umbral:
        return 'YZ'  
    else:
        raise ValueError("The wall is clearly not in the XZ or YZ plane.")


def only_exterior_walls(walls_exteriors, walls_with_intersections): 
    """
    Filters and orders the points of exterior walls based on the angle relative
    to the room's centroid.

    Parameters
    ----------
    walls_exteriors : list of dicts
        A list of dictionaries with information about exterior walls for each room.
    walls_with_intersections : list of dicts
        A list of dictionaries containing information about walls and their 
        intersection points.

    Returns
    -------
    lista_puntos_orden : list
        A list of ordered points for each exterior wall.
    union_results : list
        A list of dictionaries with the room ID, selected exterior walls,
        and their centroids.

    """
    # We iterate over the list of rooms
    ind_exterior = []
    for room in walls_exteriors:
        print(f"IdRoom: {room['IdRoom']}, IdWallsExterior: {room['IdWallsExterior']}")
        ind_exterior.append(room['IdWallsExterior'])
    
    # We initialize a list to store the results
    union_results = []
    
    # We iterate over walls_with_intersections and the corresponding indices in ind_exterior
    for i, room in enumerate(walls_with_intersections):
        room_id = room['id_room']
        walls = room['walls']
        centroide = room['centroid']
        
        exterior_indices = ind_exterior[i]  
        
        # Ready to store selected walls
        selected_walls = []
        
        # We iterate on the walls of the room
        for wall in walls:
            if wall['wall_id'] in exterior_indices:
                
                # We add the result of this wall with the combined points
                selected_walls.append({
                    'wall_id': wall['wall_id'],
                    'points': wall['points_walls']
                })
        
        # We save the result for this room
        union_results.append({
            'id_room': room_id,
            'walls': selected_walls,
            'centroid': centroide
        })
       
    
    # We initialize a list to store all the sublists of 'points'
    list_points_order = []
    
    # We iterate over the result of 'union_results'
    for room in union_results:
        center = room['centroid']
        for wall in room['walls']:
            print(wall['points'])
            w = np.array(wall['points'])
            if len(w) != 0:
                plane_side = detect_plane(w, umbral=0.45) 
    
                # Calculate the angles in the corresponding plane
                if ((plane_side == 'XZ') & (center[1] > np.mean(w[:,1]))):
                    angles = np.arctan2(w[:, 2] - center[2], w[:, 0] - center[0])   
                    sorted_indices = np.argsort(angles)
                    ordered_side = w[sorted_indices]
                    
                elif ((plane_side == 'XZ') & (center[1] < np.mean(w[:,1]))):
                    angles = np.arctan2(w[:, 2] - center[2], w[:, 0] - center[0])
                    sorted_indices = np.argsort(angles)
                    ordered_side = w[sorted_indices][::-1]
                    
                elif ((plane_side == 'YZ') & (center[0] > np.mean(w[:,0]))):
                    angles = np.arctan2(w[:, 2] - center[2], w[:, 1] - center[1])
                    sorted_indices = np.argsort(angles)
                    ordered_side = w[sorted_indices][::-1]
                    
                elif ((plane_side == 'YZ') & (center[0] < np.mean(w[:,0]))):
                    angles = np.arctan2(w[:, 2] - center[2], w[:, 1] - center[1]) 
                    sorted_indices = np.argsort(angles)
                    ordered_side = w[sorted_indices]
    
                print(f"\nPoints on wall w in plane {plane_side} clockwise:")
                print(ordered_side)
     
                list_points_order.append(ordered_side)            

    return list_points_order, union_results



# We have connection_rooms_walls_unique   
def interior_walls(connection_rooms_walls_unique, walls_with_intersections):
    """
    Selects and returns the walls from two rooms based on their connections, 
    including the ceiling and floor points for each selected wall.

    Parameters
    ----------
    connection_rooms_walls_unique : list of tuples
        A list where each tuple contains four elements: the IDs of two rooms 
        and the IDs of the walls in each room that are connected.
    walls_with_intersections : list of dicts
        A list of dictionaries where each dictionary represents a room.

    Returns
    -------
    sel : list of lists of dicts
        A list of lists, where each inner list contains dictionaries 
        representing the selected walls from both rooms involved in each 
        connection.

    """
    sel = []
    # Iterate over connection_rooms_walls_unique data
    for connection in connection_rooms_walls_unique:
        selected_walls = []
        id_room_1, id_wall_1, id_room_2, id_wall_2 = connection
        
        # Select the corresponding wall in room 1
        for room in walls_with_intersections:
            if room['id_room'] == id_room_1:
                for wall in room['walls']:
                    if wall['wall_id'] == id_wall_1:
                        wall_room_1 = {
                            'id_room': id_room_1,
                            'wall_id': id_wall_1,
                            'points_ceiling': wall['points_ceiling'],
                            'points_floor': wall['points_floor']
                        }
                        selected_walls.append(wall_room_1)
                        
        # Select the corresponding wall in room 2
        for room in walls_with_intersections:
            if room['id_room'] == id_room_2:
                for wall in room['walls']:
                    if wall['wall_id'] == id_wall_2:
                        wall_room_2 = {
                            'id_room': id_room_2,
                            'wall_id': id_wall_2,
                            'points_ceiling': wall['points_ceiling'],
                            'points_floor': wall['points_floor']
                        }
                        selected_walls.append(wall_room_2)
        sel.append(selected_walls)
    return sel

#%%
def save_floor_ceiling(path_elements, points_floor, points_ceiling, n_storey, 
                       flattened_points_floor, flattened_points_ceiling):
    """
    Saves information about the floor and ceiling of a building storey to a
    text file

    Parameters
    ----------
    path_elements : str
        The path where the text file will be saved.
    points_floor : list of tuples
        A list of tuples representing the floor points' coordinates.
    points_ceiling : list of tuples
        A list of tuples representing the ceiling points' coordinates.
    n_storey : int
        The storey number to be used in the output file.
    flattened_points_floor : list of floats
        A flattened list of floor points, where each coordinate value 
        (X, Y, Z) is in sequence.
    flattened_points_ceiling : list of floats
        A flattened list of ceiling points, where each coordinate value 
        (X, Y, Z) is in sequence.

    Returns
    -------
    file_name : str
        The path and name of the saved text file.

    """
    # Create text file
    file_name = path_elements + "\\infoBIM.txt"

    # Generate the points part dynamically
    num_pts = len(points_floor)
    pts_columns = ','.join([f'pt{i+1}_x,pt{i+1}_y,pt{i+1}_z' for i in range(num_pts)])

    # Create the header
    header = f"storey_id,element_type,element_id,is_external,height,opening_in_wall,no_pts,{pts_columns}\n"

    # Open the file in write mode and add the header and information
    with open(file_name, 'w') as file:
        # Write the header
        file.write(header)  
        # Create the row with the data
        row = [n_storey, "floor", 0, 1, '' , '', len(points_floor)] + flattened_points_floor
        # Convert each data to string and join with commas
        row_str = ",".join(map(str, row)) + "\n"
        file.write(row_str)


    # Open the file in write mode and add the header and information
    with open(file_name, 'a') as file:
        # Create the row with the data
        new_row = [n_storey, "ceiling", 1, 1, '', '', len(points_ceiling)] + flattened_points_ceiling
        # Convert each data to string and join with commas
        new_row_str = ",".join(map(str, new_row)) + "\n"
        file.write(new_row_str)

    return file_name


def save_exterior_walls(file_name, lista_puntos_orden, n_storey, height):
    """
    Saves the exterior wall information to a file and visualizes the exterior 
    walls on a plot.


    Parameters
    ----------
    file_name : str
        The name of the file to which the exterior wall information will be
        appended.
    lista_puntos_orden : list of lists of tuples
        A list containing lists of ordered tuples, where each tuple represents
        a point (X, Y) that defines a polygon for each exterior wall.
    n_storey : int
        The storey number for which the wall data is being saved.
    height : float
        The height of the exterior walls.

    Returns
    -------
    file_name : str
        The name of the file that contains the saved exterior wall data.

    """
    # Open the file in write mode and add the header and information
    with open(file_name, 'a') as file:
        for i, sublista in enumerate(lista_puntos_orden):
            # Create the row with the data
            flattened_points_walls = [coord for point in sublista for coord in point]

            new_row = [n_storey, 'wall', i, 1, height, '', len(sublista)] + flattened_points_walls
            # Convert each data to string and join with commas
            new_row_str = ",".join(map(str, new_row)) + "\n"
            file.write(new_row_str)

    # Represent exterior walls
    plt.figure(figsize=(12, 12))
    for polygon in lista_puntos_orden:
        x = polygon[:, 0]
        y = polygon[:, 1]
        plt.fill(x, y, alpha=0.5, edgecolor='r')  

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Exterior walls')
    plt.grid(True)

    # Mostrar la gráfica
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.show()
    
    return file_name


def save_interior_walls(merged_rectangles, file_name, n_storey, height):
    """
    Saves the interior wall information to a file.


    Parameters
    ----------
    merged_rectangles : list of lists of tuples
        A list containing merged rectangles, where each rectangle is 
        represented by a list of ordered points (X, Y) that define the vertices
        of the polygon.
    file_name : str
        The name of the file where the interior wall data will be saved.
    n_storey : int
        The storey number for which the wall data is being saved.
    height : float
        The height of the interior walls.

    Returns
    -------
    file_name : str
        The name of the file where the interior wall data is stored.
    all_int_walls : list of lists of tuples
        A list of all interior walls after sorting their points in a clockwise
        direction.

    """
    all_int_walls = []
    for m_p in merged_rectangles:
        m_p_order = clockwise_sort_xy(m_p)
        all_int_walls.append(m_p_order)
        # Read the file to find the last value of ind_s
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1]  
                    last_ind_s = int(last_line.split(",")[2])  
                else:
                    last_ind_s = -1  
        except FileNotFoundError:
            # If the file does not exist, start from 0
            last_ind_s = -1
        
        # The new ind_s will be the last value + 1
        c = last_ind_s + 1
        
        # You can now continue writing using the new ind_s 
        with open(file_name, 'a') as file:
            # Create the row with the data
            flattened_points_walls_interior = [coord for point in m_p_order for coord in point]
            
            new_row = [n_storey, 'wall', c, 0, height, '', len(m_p_order)] + flattened_points_walls_interior
            # Convert each data to string and join with commas
            new_row_str = ",".join(map(str, new_row)) + "\n"
            file.write(new_row_str)
            
    return file_name, all_int_walls



def save_doors(file_name, order_doors, n_storey):
    """
    Saves door information to a file and maps each door to the wall it belongs to.


    Parameters
    ----------
    file_name : str
        The name of the file where the door information will be saved.
    order_doors : list of lists of tuples
        A list of doors, where each door is represented as a list of points 
        (X, Y, Z) that define the geometry of the door.
    n_storey : int
        The storey number for which the door data is being saved.

    Returns
    -------
    file_name : str
        The name of the file where the door data is stored.

    """
    # Put in text file
    # Now I want to see which wall in the textfile each door belongs to.
    # Ready to stock the walls that meet the condition
    walls = []

    # Read the file and process each line
    with open(file_name, 'r') as file:
        for line in file:
            # Split the line into comma-separated items
            parts = line.strip().split(',')
            
            # Check if column 3 is equal to 0
            if parts[3] == '0':
                # Extract the number from the wall (column 2)
                wall_num = int(parts[2])
                
                # Extract the points (starting from column 6, in groups of three for X, Y, Z)
                points = np.array([
                    [float(parts[7]), float(parts[8]), float(parts[9])],
                    [float(parts[10]), float(parts[11]), float(parts[12])],
                    [float(parts[13]), float(parts[14]), float(parts[15])],
                    [float(parts[16]), float(parts[17]), float(parts[18])]
                ])
                
                # Add wall to list as a dictionary
                walls.append({"wall_num": wall_num, "points": points})

    # Assigning doors to walls
    door_to_wall_mapping = {}

    for i, door in enumerate(order_doors):
        for wall in walls:
            if infoBIM_doors_windows.is_door_on_wall(door, wall["points"]):
                door_to_wall_mapping[i] = wall["wall_num"]
                break 

    with open(file_name, 'a') as f:
        for i, (door_pts, wall_num) in enumerate(door_to_wall_mapping.items()):
            # Get the door points
            flattened_points_doors = [coord for point in order_doors[door_pts] for coord in point]

            # Create the data line in the desired format
            new_row = [n_storey, 'opening_door', i, 0, '', wall_num, 
                       len(flattened_points_doors)//3] + flattened_points_doors
            # Convert each data to string and join with commas
            new_row_str = ",".join(map(str, new_row)) + "\n"
            f.write(new_row_str)
            
    return file_name


def save_windows(file_name, order_windows, n_storey):
    """
    Saves window information to a file and maps each window to the wall it belongs to.


    Parameters
    ----------
    file_name : str
        The name of the file where the window information will be saved.
    order_windows : list of lists of tuples
        A list of windows, where each window is represented as a list of points
        (X, Y, Z) that define the geometry of the window.
    n_storey : int
        The storey number for which the window data is being saved.

    Returns
    -------
    file_name : str
        The name of the file where the window data is stored.

    """
    walls_e = []
    # Read the file and process each line
    with open(file_name, 'r') as file:
        for line in file:
            # Split the line into comma-separated items
            parts = line.strip().split(',')
        
            # Check if column 3 is equal to 0
            if parts[3] == '1':
                # Extract the number from the wall (column 2)
                wall_num = int(parts[2])
                if len(parts) == 19:
                    # Extract the points (starting from column 6, in groups of three for X, Y, Z)
                    points = np.array([
                        [float(parts[7]), float(parts[8]), float(parts[9])],
                        [float(parts[10]), float(parts[11]), float(parts[12])],
                        [float(parts[13]), float(parts[14]), float(parts[15])],
                        [float(parts[16]), float(parts[17]), float(parts[18])]
                    ])
                    
                    # Add wall to list as a dictionary
                    walls_e.append({"wall_num": wall_num, "points": points})

    # Assigning doors to walls
    window_to_wall_mapping = {}
    for i, window in enumerate(order_windows):
        for wall in walls_e:
            
            # Seleccionar los dos puntos más altos (últimos dos después de ordenar)
            sorted_array =  wall["points"][ wall["points"][:, 2].argsort()]
            highest_points = sorted_array[-2:]
            
            # Crear una línea usando LineString
            line = LineString([highest_points[0][0:2], highest_points[1][0:2]])
            
            # Crear buffers y verificar intersecciones
            buffer_radius = 0.05  # Radio del buffer
            intersections = []
            for p, point_coords in enumerate(window):
                point = Point(point_coords)  # Crear un punto
                buffer = point.buffer(buffer_radius)  # Crear un buffer alrededor del punto
                
                # Verificar si el buffer intersecta con la línea
                if buffer.intersects(line):
                    intersections.append((p, point_coords))
                    
                    window_to_wall_mapping[i] = wall["wall_num"]
                    break 

    with open(file_name, 'a') as f:
        for i, (window_pts, wall_num) in enumerate(window_to_wall_mapping.items()):
            # Get the door points
            flattened_points_windows = [coord for point in order_windows[window_pts] for coord in point]

            # Create the data line in the desired format
            new_row = [n_storey, 'opening_window', i, 1, '', wall_num, 
                       len(flattened_points_windows)//3] + flattened_points_windows
            # Convert each data to string and join with commas
            new_row_str = ",".join(map(str, new_row)) + "\n"
            f.write(new_row_str)

    return file_name


