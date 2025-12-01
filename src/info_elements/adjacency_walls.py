# -*- coding: utf-8 -*-

import numpy as np 

def are_parallel(r1, r2):
    """
    Determines whether two lines represented by the ratios of their 
    coefficients are parallel.


    Parameters
    ----------
    r1 : tuple
        A tuple containing the coefficients for the first line equation.
    r2 :tuple
        A tuple containing the coefficients for the second line equation.

    Returns
    -------
    bool
        True if the two lines are parallel, False otherwise.

    """
    A1, B1, _ = r1
    A2, B2, _ = r2
    return np.isclose(A1 * B2, A2 * B1)

def dist_lines_parallel(r1, r2):
    """
    Calculates the perpendicular distance between two parallel lines.

    Parameters
    ----------
    r1 : tuple
        A tuple containing the coefficients for the first line equation.
    r2 :tuple
        A tuple containing the coefficients for the second line equation.

    Returns
    -------
    float
        The perpendicular distance between the two parallel lines.

    """
    A, B, C1 = r1
    _, _, C2 = r2
    return abs(C2 - C1) / np.sqrt(A**2 + B**2)


def parallel_d_room(class_pcd_df, lines_position_t, df_walls_t, 
                    equation_df_walls_t, ex_v, dist_th, min_dist_w):
    """
    Identifies and connects rooms based on their parallel wall lines and 
    proximity.

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing the point cloud data for the rooms.
    lines_position_t : list of tuples
        A list where each element represents the positions of lines 
        of walls in each room.
    df_walls_t : list of lists
        A list containing indices that reference the walls in each room's point
        cloud data.
    equation_df_walls_t : list of lists
        A list containing the equations representing the planes of the walls
        for each room.
    ex_v : list
        A list of room IDs to exclude from the analysis.
    dist_th : float
        The threshold distance below which two parallel walls are considered 
        connected.
    min_dist_w : float
        The minimum distance between points in the point cloud for a connection
        to be considered valid.

    Returns
    -------
    connection_rooms : list of lists
        A list of pairs of room IDs that are considered connected due to their 
        parallel walls being sufficiently close.
    connection_rooms_walls : list of lists
        A list of walls from the two connected rooms, indicating which walls 
        form the connection.
    

    """
    connection_rooms = []
    connection_rooms_walls = []
    
    add_room = []
    add_wall = []
    
    v = [valor for valor in range(1, len(
        class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v]
    n = len([valor for valor in range(1, len(
        class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v])
    
    for i in range(n):
        for j in range(i + 1, n):
            for id_r1, r1 in enumerate(lines_position_t[i]):
                for id_r2, r2 in enumerate(lines_position_t[j]):
                    if are_parallel(r1, r2):
                        distance = dist_lines_parallel(r1, r2)
                        if distance < dist_th:
                            add_room.append((v[i], v[j], distance))
                            add_wall.append((id_r1, id_r2))                           
                            
                            xy_values = class_pcd_df.iloc[
                                df_walls_t[i][id_r1]][['x', 'y']].copy()

                            x_range = np.max(xy_values['x']) - np.min(
                                xy_values['x'])
                            y_range = np.max(xy_values['y']) - np.min(
                                xy_values['y'])

                            if x_range > y_range:
                                direction = 'x'
                            else:
                                direction = 'y'

                            xy_values['segment'] = np.floor(
                                xy_values[direction] / 0.5) * 0.5

                            # Group the points by the 0.5 meter segments
                            grouped_segments = xy_values.groupby('segment')

                            # Create a list to store the center of each segment
                            segment_centers = []

                            for segment, group in grouped_segments:
                                center_x = group['x'].mean()
                                center_y = group['y'].mean()
                                
                                segment_centers.append([center_x, center_y])

                            if len(segment_centers) < 6:
                                segment_centers = []
                                segment_centers.append([class_pcd_df.iloc[
                                    df_walls_t[i][id_r1]][['x','y','z']][
                                        'x'].mean(), class_pcd_df.iloc[
                                            df_walls_t[i][id_r1]][
                                                ['x','y','z']]['y'].mean()])

                            else:
                                segment_centers = segment_centers[2:-2]

                            for p in segment_centers:
                                p.append(class_pcd_df.iloc[df_walls_t[i][
                                    id_r1]][['x','y','z']]['z'].max()) 
                                
                            
                            for p in segment_centers:
                                eq_plane = equation_df_walls_t[j][id_r2]                    
                                normal_vector = eq_plane[:3]
                                
                                # Calculate the distance of the point from the equation plane
                                distance_to_plane = np.dot(normal_vector, p
                                                           ) + eq_plane[3]                               
                                projection_point = p - distance_to_plane * normal_vector
                                
                                z_mean = class_pcd_df.iloc[df_walls_t[j][
                                    id_r2]]['z'].mean()
                                pc_sup_z = class_pcd_df.iloc[df_walls_t[j][
                                    id_r2]][class_pcd_df.iloc[df_walls_t[j][
                                        id_r2]]['z'] > z_mean]
                                pointcloud = np.array(pc_sup_z[['x','y','z']])
                                distancias = np.linalg.norm(
                                    pointcloud - projection_point, axis=1)
                                
                                if len(pointcloud[distancias <= min_dist_w]) > 0:
                                    connection_rooms.append([v[i], v[j]]) 
                                    connection_rooms_walls.append([v[i], id_r1,
                                                                   v[j], id_r2]
                                                                  ) 
                                    [list(item) for item in set(tuple(
                                        row) for row in connection_rooms_walls)]
                                    
                                                       
                            
                            xy_values = class_pcd_df.iloc[df_walls_t[j][
                                id_r2]][['x', 'y']].copy()

                            x_range = np.max(xy_values['x']) - np.min(
                                xy_values['x'])
                            y_range = np.max(xy_values['y']) - np.min(
                                xy_values['y'])

                            # Determine the main direction to segment
                            if x_range > y_range:
                                direction = 'x'
                            else:
                                direction = 'y'

                            xy_values['segment'] = np.floor(
                                xy_values[direction] / 0.5) * 0.5

                            # Group the points by the 0.5 meter segments
                            grouped_segments = xy_values.groupby('segment')

                            # Create a list to store the center of each segment
                            segment_centers = []

                            # Calcular el centroide de cada segmento
                            for segment, group in grouped_segments:
                                # Calcular el promedio de x e y en cada grupo
                                center_x = group['x'].mean()
                                center_y = group['y'].mean()
                                
                                # Save the center point (centroid)
                                segment_centers.append([center_x, center_y])

                            if len(segment_centers) < 6:
                                segment_centers = []
                                segment_centers.append([class_pcd_df.iloc[
                                    df_walls_t[j][id_r2]][['x','y','z']][
                                        'x'].mean(), class_pcd_df.iloc[
                                            df_walls_t[j][id_r2]][
                                                ['x','y','z']]['y'].mean()])

                            else:
                                segment_centers = segment_centers[2:-2]

                            for p in segment_centers:
                                p.append(class_pcd_df.iloc[df_walls_t[j][
                                    id_r2]][['x','y','z']]['z'].max()) #mean
                                
                            
                            for p in segment_centers:
                                # The point is projected with the plane of the other equation
                                eq_plane = equation_df_walls_t[i][id_r1]                    
                                normal_vector = eq_plane[:3]
                                
                                # Calculate the distance of the point from the equation plane
                                distance_to_plane = np.dot(normal_vector, p) + eq_plane[3]                               
                                projection_point = p - distance_to_plane * normal_vector
                                
                                z_mean = class_pcd_df.iloc[df_walls_t[i][
                                    id_r1]]['z'].mean()
                                pc_sup_z = class_pcd_df.iloc[df_walls_t[i][
                                    id_r1]][class_pcd_df.iloc[df_walls_t[i][
                                        id_r1]]['z'] > z_mean]
                                pointcloud = np.array(pc_sup_z[['x','y','z']])
                                distancias = np.linalg.norm(
                                    pointcloud - projection_point, axis=1)
                                
                                if len(pointcloud[distancias <= min_dist_w]) > 0:
                                    connection_rooms.append([v[i], v[j]]) 
                                    connection_rooms_walls.append([v[i], id_r1,
                                                                   v[j], id_r2]) 
                                    [list(item) for item in set(tuple(
                                        row) for row in connection_rooms_walls)]  
                            
                            
    return connection_rooms, connection_rooms_walls                            



def check_adjacency_exterior_walls(connection_rooms_walls_unique, 
                                   connection_rooms_unique, walls_exteriors):
    """
    Checks and removes connections between rooms that share exterior walls.

    Parameters
    ----------
    connection_rooms_walls_unique : list of lists
        A list of unique connections between rooms represented by their walls.
    connection_rooms_unique : list of lists
        A list of unique connections between rooms. 
        connected.
    walls_exteriors : list of dicts
        A list of dictionaries containing information about exterior walls.

    Returns
    -------
    connection_rooms_unique : list of lists
        The updated list of unique room connections.
    connection_rooms_walls_unique : list of lists
        The updated list of unique wall connections.
   

    """
    indices_to_remove_walls = []
    indices_to_remove_rooms = []
    for ind_c, c in enumerate(connection_rooms_walls_unique):
        r1, w1, r2, w2 = c
        room_1 = next((
            item for item in walls_exteriors if item['IdRoom'] == r1), None)
        room_2 = next((
            item for item in walls_exteriors if item['IdRoom'] == r2), None)
    
        if (w1 in room_1['IdWallsExterior']) and (
                w2 in room_2['IdWallsExterior']):         
            indices_to_remove_walls.append(ind_c)
            indices_to_remove_rooms.append(ind_c)
        
    for index in reversed(indices_to_remove_walls):
        connection_rooms_walls_unique.pop(index)
    
    for index in reversed(indices_to_remove_rooms):
        connection_rooms_unique.pop(index)
    
    return connection_rooms_unique, connection_rooms_walls_unique


    
    
    

    



























                 