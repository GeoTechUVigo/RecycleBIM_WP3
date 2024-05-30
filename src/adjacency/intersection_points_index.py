# -*- coding: utf-8 -*-

import math
import numpy as np
from src.utils import functions_geom
from src.utils import functions_walls

def angle_lines(coefficients1, coefficients2):
    """
    Calculate angle between two lines

    """
    # Convert coefficients to normalized direction vectors
    a1, b1, _ = coefficients1
    a2, b2, _ = coefficients2
    magn1 = math.sqrt(a1**2 + b1**2)
    magn2 = math.sqrt(a2**2 + b2**2)
    
    # Check if magnitudes are not zero to avoid division by zero
    if magn1 != 0 and magn2 != 0:
        # Calculate direction vectors
        direccion1 = (a1 / magn1, b1 / magn1)
        direccion2 = (a2 / magn2, b2 / magn2)

        # Calculate the dot product between the direction vectors
        producto_punto = direccion1[0] * direccion2[0] + direccion1[1] * direccion2[1]

        # Ensure the dot product is within the valid range [-1, 1]
        if -1 <= producto_punto <= 1:
            # Calculate the angle between the vectors
            angulo_rad = math.acos(producto_punto)
            angulo_deg = math.degrees(angulo_rad)
        else:
            # Handle the case when the dot product is outside the valid range
            angulo_deg = float('nan')  # Or any other value you want to assign
    else:
        # Handle the case when one or both magnitudes are zero
        angulo_deg = float('nan')  # Or any other value you want to assign

    return round(angulo_deg, 2)

def find_point_intersection(coeficientes1, coeficientes2):
    """
    Calculate point intersection between two lines

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

#%%
def intersect(class_pcd_df, lines_position_t, df_walls_t, b_f, min_p_int, max_dist_int, min_angle_d, max_angle_d, perfect_angle, flat_angle, ex_v):
    """
    Find the intersection points in the different rooms

    """
    points_intersection = []
    room_data_list = []
    for r, room in enumerate([valor for valor in range(1, len(class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v]):
        room_data = {}
        room_data["id_room"] = room
        print(f"----------------- Room {room} -----------------")
        rec = lines_position_t[r] 
        rpc = df_walls_t[r]    
        
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
                            if functions_geom.distance((points_array[0], points_array[1]), intersection_point) <= buffer_radius:
                                points_within_buffer_i.append(points_array)
                                
                        for points_array in np.array(class_pcd_df.iloc[rpc[j]][['x', 'y', 'z']]):
                            if functions_geom.distance((points_array[0], points_array[1]), intersection_point) <= buffer_radius:
                                points_within_buffer_j.append(points_array)

                        if len(points_within_buffer_i) > 0 and len(points_within_buffer_i) > 0:
                            if f'Wall {i} is adjacent to Wall {j}' not in list_adj:
                                list_adj.append(f"Wall {i} is adjacent to Wall {j}")
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
        
    return points_intersection, room_data_list


 
def only_intersection_points(points_intersection, z_0):            
    modified_points_intersection = []        
    for room in points_intersection:
        room_m = []
        for point in room:
            # Add the average height at which the point cloud is
            room_m.append(point + (np.round(z_0, 2),))
        
        modified_points_intersection.append(room_m)
    
    return modified_points_intersection
    

def floor_ceiling_plane_json(class_pcd_df, modified_points_intersection, min_ratio, threshold, iterat):
    """
    Calculate the orthogonal projection to the plane of the floor and ceiling.

    """
    points_floor = class_pcd_df[class_pcd_df['id_element'] == 0]
    points_ceil = class_pcd_df[class_pcd_df['id_element'] == 1]
    floor_plane = functions_walls.DetectMultiPlanes(np.array(points_floor[['x','y','z']]), 
                                                    min_ratio, threshold, iterations=iterat)
    ceil_plane = functions_walls.DetectMultiPlanes(np.array(points_ceil[['x','y','z']]), 
                                                   min_ratio, threshold, iterations=iterat)

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
    
