# -*- coding: utf-8 -*-
import math
import copy
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance


def distance_point_to_line(x0, y0, A, B, C):
    return abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)

def distance_lines(line1, line2):
    """
    Calculates the perpendicular distance between two parallel lines in 2D space.


    Parameters
    ----------
    line1 : tuple
        A tuple containing the coefficients of the first line in the form.
    line2 : tuple
        A tuple containing the coefficients of the second line in the form.

    Returns
    -------
    float or None
        Returns the perpendicular distance between the two parallel lines if 
        they are parallel 

    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    if A1 == A2 and B1 == B2:  
        return abs(C2 - C1) / math.sqrt(A1**2 + B1**2)
    else:
        return None 


def mod_lines(class_pcd_df, lines_position_t, df_walls_t, equation_df_walls_t, 
              walls_df, dist_p):
    """
    Modifies the walls based on their proximity to each other and cleans up 
    redundant entries.


    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data.
    lines_position_t : list of lists
        A list of rooms, where each room contains a list of lines represented 
        by their coefficients in the general form Ax + By + C = 0.
    df_walls_t : list of lists
        A list of walls for each room, where each wall is represented by a 
        list of indices into `class_pcd_df`. 
    equation_df_walls_t : list of lists
        A list of wall equations for each room, where each wall is represented
        by its equation coefficients. 
    walls_df : pandas.DataFrame
        A DataFrame representing the walls in a given building. The function 
        removes rows corresponding to walls that are removed
        from the `lines_position_t` and `df_walls_t` lists.
    dist_p : float
        The threshold distance below which two walls are considered redundant and one is removed.

    Returns
    -------
    list
        A list containing the updated `lines_position_t`, `df_walls_t`, 
        `equation_df_walls_t`, `walls_df`, and `class_pcd_df`. These data 
        structures are modified in place during the function's operation.

    """
    lines_position_t_copy = copy.deepcopy(lines_position_t) 
    for n_room, lines in enumerate(lines_position_t_copy):
        # Create a list of distances between parallel lines
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                distance = distance_lines(lines[i], lines[j])
                if distance is not None:
                    if distance < dist_p:
                        if len(df_walls_t[n_room][i]) > len(
                                df_walls_t[n_room][j]):
                            equation_df_walls_t[n_room].pop(j)
                            lines_position_t[n_room].pop(j)
                            
                            # Fix assignment in class_pcd_df
                            class_pcd_df.loc[df_walls_t[n_room][j], 
                                             'id_entity'] = 99
                            
                            # Reassign walls_df after drop
                            walls_df = walls_df.drop(walls_df[(
                                walls_df['id_entity'] == j) & (
                                    walls_df['id_room'] == (n_room + 1))].index
                                    )
                            
                            df_walls_t[n_room].pop(j)
    
                        elif len(df_walls_t[n_room][i]) < len(
                                df_walls_t[n_room][j]):
                            equation_df_walls_t[n_room].pop(i)
                            lines_position_t[n_room].pop(i)
                            
                            # Fix assignment in class_pcd_df
                            class_pcd_df.loc[df_walls_t[n_room][i], 
                                             'id_entity'] = 99
                            
                            # Reassign walls_df after drop
                            walls_df = walls_df.drop(walls_df[(
                                walls_df['id_entity'] == i) & (
                                    walls_df['id_room'] == (n_room + 1))].index
                                    )
                            
                            df_walls_t[n_room].pop(i)
                            
    return [lines_position_t, df_walls_t, equation_df_walls_t, walls_df, 
            class_pcd_df]



def minimum_4_walls(lines_position_t, class_pcd_df, df_walls_t, 
                    equation_df_walls_t, walls_df, um=0.005, um_d=0.1):
    """
    Adds new walls between rooms based on their proximity and parallelism.


    Parameters
    ----------
    lines_position_t : list of lists
        A list of rooms, where each room contains a list of lines represented
        by their coefficients in the general line equation.
    class_pcd_df : pandas.DataFrame
        A DataFrame containing point cloud data. 
    df_walls_t : list of lists
        A list of lists, where each inner list contains indices of points 
        representing walls for each room.
    equation_df_walls_t : list of lists
        A list of lists containing the equations of walls for each room.
    walls_df : pandas.DataFrame
        A DataFrame containing the walls data.
    um : float, optional, default=0.005
        The distance threshold (in meters) used to filter points that are 
        considered "close" to the wall contours.
    um_d : float, optional, default=0.1
        The maximum distance allowed for a point to be considered as 
        part of a new wall to be added.
    
    Returns
    -------
    tuple
        A tuple containing the following updated data structures:
        - lines_position_t (list): The updated list of line positions for each room.
        - df_walls_t (list): The updated list of wall data for each room.
        - equation_df_walls_t (list): The updated list of wall equations for each room.
        - walls_df (pandas.DataFrame): The updated DataFrame containing all wall data.
        - class_pcd_df (pandas.DataFrame): The updated point cloud DataFrame with new wall points.

    """
    lines_position_t_check = copy.deepcopy(lines_position_t)
    for n_lines, lines in enumerate(lines_position_t_check): 
        if len(lines) == 3:
            line_no_parallel = []
            
            # See which one has no parallel:
            # Create a dictionary to count the occurrences of each (A, B)
            coefficient_counting = {}
            
            # Count how many times the coefficients (A, B) appear on the lines
            for recta in lines:
                A, B, _ = recta
                if (A, B) in coefficient_counting:
                    coefficient_counting[(A, B)] += 1
                else:
                    coefficient_counting[(A, B)] = 1
            
            # Find the line that has no parallel
            for recta in lines:
                A, B, C = recta
                if coefficient_counting[(A, B)] == 1:
                    line_no_parallel.append(recta)
            # Filter points for room 1
            room_1 = class_pcd_df[class_pcd_df['id_room'] == (n_lines + 1)]
            
            # Get the XY coordinates of room 1
            points_r1 = room_1[['x', 'y']].values
            
            # Calculate the contour (Convex Hull) of the points in room 1
            hull = ConvexHull(points_r1)
            
            # Get the outline points of room 1
            contour_r1 = points_r1[hull.vertices]
            
            # Filter points from other rooms
            other_rooms = class_pcd_df[class_pcd_df['id_room'] != (n_lines + 1)]
            
            # Get the XY coordinates of the other rooms
            points_others = other_rooms[['x', 'y']].values
            
            # Calculate the distance from points in other rooms to the outline 
            # of room 1
            distances = distance.cdist(points_others, contour_r1)
            
            # Filter points that are within the distance threshold
            close_index = np.any(distances <= um, axis=1)
            
            # Get the glued points from the other rooms
            close_points = other_rooms[close_index]
            
            number_of_contour_points = []
            possible_lines_to_add = []
            position_room = []
            position_line = [] 
            
            for c in close_points['id_room'].unique():
                for n_l, l in enumerate(lines_position_t_check[c - 1]): 
                    if (distance_lines(line_no_parallel[0], l) != None) and (
                            distance_lines(line_no_parallel[0], l) > 1):

                        # Assuming you already have your DataFrame class_pcd_df
                        # Extract the outline of room 1
                        room_contour = class_pcd_df[
                            class_pcd_df['id_room'] == (n_lines + 1)][
                                ['x', 'y']].to_numpy()   
                        A, B, C = lines_position_t_check[c - 1][n_l]

                        # Check if any point on the contour is close to the line
                        pts_close = []
                        for punto in room_contour:
                            x, y = punto
                            dist = distance_point_to_line(x, y, A, B, C)
                            if dist < um_d:
                                pts_close.append((x, y, dist))
                        
                        # Results
                        if pts_close:
                            number_of_contour_points.append(len(pts_close))

                            # Now the missing "wall" will be created
                            # Coefficients of the lines
                            A = line_no_parallel[0][0]
                            B = line_no_parallel[0][1]
                            C1 = lines_position_t_check[c - 1][n_l][2]
                            C2 = line_no_parallel[0][0]
                            
                            # Move the first line (0.1 m) closer to the second line
                            distancia_a_mover = -um_d 
                            nuevo_C1 = C1 + (distancia_a_mover * (C2 - C1) / abs(C2 - C1))
                            
                            # Result
                            possible_lines_to_add.append([A, B, nuevo_C1])                        
                            position_room.append(c-1)
                            position_line.append(n_l)
                                                        
            if len(number_of_contour_points) == 1:
                lines_position_t[n_lines].append(possible_lines_to_add[0])
                equation_df_walls_t[n_lines].append(equation_df_walls_t[
                    position_room[0]][position_line[0]])
                
                df_walls_t[n_lines].append(df_walls_t[position_room[0]][
                    position_line[0]])
                
                # Filter points that belong to 'id_room' = 2 and 'id_entity' = 0
                filtered_points = walls_df[(walls_df['id_room'] == (
                    position_room[0] + 1)) & (
                        walls_df['id_entity'] == position_line[0])]
                new_points = filtered_points.copy()
                new_points['id_room'] = (n_lines+1)
                new_points['id_entity'] = (walls_df['id_entity'].max() + 1)
                # Concatenate the new points to the original DataFrame
                walls_df = pd.concat([walls_df, new_points], ignore_index=True)
                      
                # Filter the points you want to add
                filtered_points = class_pcd_df[(class_pcd_df['id_room'] == (
                    position_room[0] + 1)) & (
                        class_pcd_df['id_entity'] == position_line[0])]
                new_points = filtered_points.copy()
                new_points['id_room'] = (n_lines+1)
                new_points['id_entity'] = (class_pcd_df[
                    class_pcd_df['id_room'] == (
                        n_lines + 1)]['id_entity'].max() + 1)
                # Concatenate the new points to the original DataFrame
                class_pcd_df = pd.concat([class_pcd_df, new_points], 
                                         ignore_index=True)
                
            elif len(number_of_contour_points) > 1:
                max_index = number_of_contour_points.index(
                    max(number_of_contour_points))
                lines_position_t[n_lines].append(
                    possible_lines_to_add[max_index]) 
                equation_df_walls_t[n_lines].append(
                    equation_df_walls_t[position_room[max_index]][
                        position_line[max_index]])
                
                df_walls_t[n_lines].append(df_walls_t[
                    position_room[max_index]][position_line[max_index]])
                
                # Filter the points you want to add
                filtered_points = walls_df[(walls_df['id_room'] == (
                    position_room[max_index] + 1)) & (
                        walls_df['id_entity'] == position_line[max_index])]
                new_points = filtered_points.copy()
                new_points['id_room'] = (n_lines+1)
                new_points['id_entity'] = (walls_df['id_entity'].max() + 1)
                # Concatenate the new points to the original DataFrame
                walls_df = pd.concat([walls_df, new_points], ignore_index=True)
                
                # Filter points that belong to 'id_room' = 2 and 'id_entity' = 0
                filtered_points = class_pcd_df[(class_pcd_df['id_room'] == (
                    position_room[max_index] + 1)) & (
                        class_pcd_df['id_entity'] == position_line[max_index])]
                new_points = filtered_points.copy()
                new_points['id_room'] = (n_lines+1)
                new_points['id_entity'] = (class_pcd_df[
                    class_pcd_df['id_room'] == (
                        n_lines + 1)]['id_entity'].max() + 1)
                # Concatenate the new points to the original DataFrame
                class_pcd_df = pd.concat([class_pcd_df, new_points], 
                                         ignore_index=True)
                   
    return lines_position_t, df_walls_t, equation_df_walls_t, walls_df, class_pcd_df







