# -*- coding: utf-8 -*-
import math
import numpy as np

def angle_lines(coefficients1, coefficients2):
    """
    Calculates the angle between two lines defined by their coefficients in a 
    2D space.


    Parameters
    ----------
    coefficients1 : tuple of float
        A tuple containing the coefficients (A, B, C) of the first line.
    coefficients2 : tuple of float
        A tuple containing the coefficients (A, B, C) of the second line.

    Returns
    -------
    float
        The angle between the two lines in degrees, rounded to two decimal places.

    """
    # Convert coefficients to normalized direction vectors
    a1, b1, _ = coefficients1
    a2, b2, _ = coefficients2
    magn1 = math.sqrt(a1**2 + b1**2)
    magn2 = math.sqrt(a2**2 + b2**2)
    
    # Check if magnitudes are not zero to avoid division by zero
    if magn1 != 0 and magn2 != 0:
        # Calculate direction vectors
        direction1 = (a1 / magn1, b1 / magn1)
        direction2 = (a2 / magn2, b2 / magn2)

        # Calculate the dot product between the direction vectors
        product_point = direction1[0] * direction2[0] + direction1[1] * direction2[1]

        # Ensure the dot product is within the valid range [-1, 1]
        if -1 <= product_point <= 1:
            # Calculate the angle between the vectors
            angle_rad = math.acos(product_point)
            angle_deg = math.degrees(angle_rad)
        else:
            angle_deg = float('nan') 
    else:
        angle_deg = float('nan')  

    return round(angle_deg, 2)

def find_point_intersection(coefficients1, coefficients2):
    """
    Finds the intersection point of two lines represented by their
    coefficients in a 2D space.

    Parameters
    ----------
    coeficientes1 : tuple of float
        A tuple containing the coefficients (A1, B1, C1) of the first line.
    coeficientes2 : tuple of float
        A tuple containing the coefficients (A2, B2, C2) of the second line.

    Returns
    -------
    tuple of float or None
        If the lines intersect, returns the coordinates (x, y) of the 
        intersection point, rounded to two decimal places.
        If the lines are parallel (determinant is zero), returns `None`
        indicating no intersection.


    """
    a1, b1, c1 = coefficients1
    a2, b2, c2 = coefficients2

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
    
def eq_vd_p(vd, p):
    """
    Calculates the general form of the equation of a line (Ax + By + C = 0)
    given a direction vector and a point on the line.

    Parameters
    ----------
    vd : tuple of float
        A tuple representing the direction vector (v1, v2) of the line.
    p : tuple of float
        A tuple representing the coordinates (x, y) of a point on the line.

    Returns
    -------
    list of float
        A list containing the coefficients [A, B, C] of the general line equation.

    """
    v1, v2 = vd
    x, y = p
    
    # Calculate the slope
    m = v2 / v1
    
    # Calculate the independent term (b) using the equation of the line y = mx + b
    b = y - m * x
    
    # Calculate the coefficients of the general equation of the line Ax + By + C = 0
    A = -m
    B = 1
    C = -b    
    
    return [A, B, C]

    
def adjust_points(class_pcd_df, points_intersection, room_data_list, 
                  lines_position_t, df_walls_t, z_0, angle_points_global, 
                  ang_1, ang_2):
    """
    Adjust the positions of lines in the 2D plane based on intersection points
    and adjacency data from a 3D model

    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        Dataframe containing the 3D point cloud data for the model, where each 
        row represents a point in the 3D space.
    points_intersection : list of list of tuples
        A nested list containing intersection points of lines in 2D space. 
    room_data_list : list of dicts
        A list of dictionaries, each containing data for a room, including 
        adjacency information, which is used to determine the relationships 
        between different walls and rooms.
    lines_position_t : list of list of lists
        A nested list representing the positions of the walls/lines in the 
        2D plane. Each room has a list of lines, and each line is defined by 
        its coefficients in the general form of a line equation.
    df_walls_t : list of lists
        A nested list of indices pointing to specific rows in `class_pcd_df`
        for each wall in each room.
    z_0 : float
        A reference z-coordinate (used for alignment or consistency with 3D 
        space, though not directly used in the function itself).
    angle_points_global : float
        The global angle used for angle comparison when adjusting the wall 
        positions.
    ang_1 : float
        The lower angle threshold to filter wall adjustments based on angular 
        conditions.
    ang_2 : float
        The upper angle threshold to filter wall adjustments based on angular 
        conditions.

    Returns
    -------
    lines_position_t : list of list of lists
        The updated positions of the lines in the 2D plane after adjustments 
        to ensure that certain walls are perpendicular.

    """
    max_y = float('-inf')  
    max_point = None 
    max_set_index = None 
    max_point_index = None  
    
    for set_index, points_set in enumerate(points_intersection):
        for point_index, point in enumerate(points_set):
            if point[1] > max_y: 
                max_y = point[1]  
                max_point = point  
                max_set_index = set_index  
                max_point_index = point_index  
    
    target_id_room = max_set_index
    index_in_adjacency = max_point_index  
    
    desired_entry = room_data_list[max_set_index]['adjacency'][index_in_adjacency]
    
    # The main equations are:
    v_1 = int(desired_entry.split(' ')[1])
    v_2 = int(desired_entry.split(' ')[6])
    
    # We have that the lines that belong to those walls are:
    r1 = lines_position_t[target_id_room][v_1]
    r2 = lines_position_t[target_id_room][v_2]
    
    # Make them form 90º
    A1, B1, C1 = r1
    A2, B2, C2 = r2

    # Calculate the directional vectors of the lines
    v1 = (B1, -A1)
    v2 = (B2, -A2)

    # Calculate the scalar product between the directional vectors
    producto_escalar = v1[0] * v2[0] + v1[1] * v2[1]

    # Check if the lines are already perpendicular
    if producto_escalar != 0:
        # Adjust one of the directional vectors to be perpendicular to the other
        v1 = (-v2[1], v2[0])

        # Check again if the directional vectors are perpendicular
        producto_escalar = v1[0] * v2[0] + v1[1] * v2[1]

        if producto_escalar == 0:
            print("The lines are now perpendicular.")
        else:
            print("Could not properly adjust the vectors so that the lines are perpendicular.")
    else:
        print("The lines are now perpendicular.")
     
    # Point through which the first original line passes
    p_b1 = np.median(class_pcd_df.iloc[df_walls_t[target_id_room][v_1]], axis=0)[0:2]

    # Point through which the second original line passes
    p_b2 = np.median(class_pcd_df.iloc[df_walls_t[target_id_room][v_2]], axis=0)[0:2]

    r1 = eq_vd_p(v1, p_b1)
    r2 = eq_vd_p(v2, p_b2)
    
    for ind_i, i in enumerate(lines_position_t):
        for ind_j, j in enumerate(i):
            if r1 != j and r2 != j:
                a_1 = angle_lines(r1, j)
                a_2 = angle_lines(r2, j)
                if a_1 < ang_1 or a_1 > ang_2: 
                    pt = np.median(class_pcd_df.iloc[df_walls_t[ind_i][ind_j]], axis=0)[0:2]   
                    vd = (-r1[1], r1[0])
                    lines_position_t[ind_i][ind_j] = eq_vd_p(vd, pt)
                    
                elif a_2 < ang_1 or a_2 > ang_2: 
                    pt = np.median(class_pcd_df.iloc[df_walls_t[ind_i][ind_j]], axis=0)[0:2]                   
                    vd = (-r2[1], r2[0])
                    lines_position_t[ind_i][ind_j] = eq_vd_p(vd, pt)
                    
    return lines_position_t
