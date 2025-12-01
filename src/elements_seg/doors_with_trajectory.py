# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


def detect_doors_traj(class_pcd_df, df, dist_traj, b_traj): 
    """
    Detects potential doors or passageways between rooms in a 3D point cloud, 
    based on trajectory analysis.


    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing the classified 3D points from a point cloud. 
    df : pandas.DataFrame
        A DataFrame representing a trajectory or path with 3D points, with 
        columns for the x, y, and z coordinates. 
    dist_traj : float
        Distance of the trajectory points.
    b_traj : float
        Create buffer for each point of the trajectory
        
    Returns
    -------
    pairs : list of list
        A list of pairs of room IDs, where each pair represents two rooms that
        are connected by a potential doorway or passageway.
    points_door : list of list
        A list of points representing the locations of doors, with each 
        element in the list being a pair of points in 2D (x, y)  coordinates 
        that potentially represent door locations.
    df_1m : pandas.DataFrame
        A DataFrame containing the points selected at 1-meter intervals along 
        the trajectory, along with an additional 'interval' column.
    list_all : list of list
        A list of lists, where each inner list contains 3D points 
        (x, y, id_room) from `class_pcd_df` that are within a buffer around 
        a selected point in the trajectory.
    list_all_t : list of list
        A list of lists where each inner list contains the points from 
        `class_pcd_df` that intersect with the buffer around the 
        corresponding trajectory points. These points are potential doorways 
        or passageway markers.


    """
    # Step 1: Calculate the differences between consecutive points
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dz'] = df['z'].diff()
    
    # Replace NaN values ​​in the first point with 0
    df.fillna(0, inplace=True)
    
    # Step 2: Calculate the Euclidean distance between consecutive points
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    
    # Step 3: Calculate the cumulative distance
    df['cumulative_distance'] = df['distance'].cumsum()
    
    # Step 4: Select the points that are at 0.5 meter intervals
    points_1m = [0]  # Initially, the first point is always included
    
    for i in range(1, len(df)):
        if df['cumulative_distance'].iloc[i] - df['cumulative_distance'].iloc[
                points_1m[-1]] >= dist_traj:
            points_1m.append(i)
    
    # Crear un nuevo DataFrame solo con los puntos seleccionados
    df_1m = df.iloc[points_1m]
       
    # Step 5: Add a column indicating the interval number
    df_1m['interval'] = np.arange(len(df_1m))
        
    # Show the result
    # print(df_1m[['x', 'y', 'z', 'interval']])
    
    # Create buffer for each point of the trajectory  
    point_buffer = []
    list_all_t = []
    for point in np.array(df_1m[['x','y']]):
        # print(Point(point))
        buffer = Point(point).buffer(b_traj)
        list_int = []
    
        for p in np.array(class_pcd_df[class_pcd_df['id_element']==1
                                       ][['x','y','id_room']]):
            p_geom = Point(p[0:2])
            if buffer.contains(p_geom):
               list_int.append(p)
               point_buffer.append(point)
    
        list_all_t.append(list_int)
       
    # Delete empty lists
    list_all = [lst for lst in list_all_t if len(lst) > 0]
    
    union_rooms = []
    for ind_i, i in enumerate(list_all):
        if len(np.unique(np.array(i)[:,2])) == 1:
            union_rooms.append(i)
                
    # We convert the union_rooms arrays into a flat list of values
    union_rooms_flat = [x[0] for x in union_rooms]
     
    pairs = []
    points_door = []
    # We go through the list and compare the consecutive numbers
    for i in range(len(union_rooms_flat) - 1):
        if union_rooms_flat[i][2] != union_rooms_flat[i + 1][2]:
            # If the numbers are different, we add them as a pair
            pairs.append([int(union_rooms_flat[i][2]), 
                          int(union_rooms_flat[i + 1][2])]) 
            points_door.append([union_rooms_flat[i][0:2], 
                                union_rooms_flat[i + 1][0:2]])


    return pairs, points_door, df_1m, list_all, list_all_t


#%%
def calculate_rectangle_orientation(points):
    """
    Calculates the orientation of a rectangle based on a set of 2D points.

    Parameters
    ----------
    points : list of tuple or list of list
        A list containing the 2D points (x, y) representing the corners or 
        edges of a rectangle. 

    Returns
    -------
    orientation : str
        The orientation of the rectangle.

    """    
    # Extract x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Calculate the width and height of the rectangle
    width = max(x_coords) - min(x_coords)
    high = max(y_coords) - min(y_coords)
    
    # Determine the orientation
    if width > high:
        orientation = "X"
    else:
        orientation = "Y"
    
    return orientation

def detect_doors_traj_2(df_1m, lines_final, all_int_walls, w_th, w_width, w_high): 
    """
    Detects door locations in a set of trajectory lines by checking 
    intersections with walls and filtering out redundant lines.

    Parameters
    ----------
    df_1m : pandas.DataFrame
        A DataFrame containing the 3D points sampled at 1-meter intervals 
        along the trajectory.
    lines_final : list of list
        A list of trajectory lines, where each line is represented as a 
        sequence of 2D or 3D coordinates.
    all_int_walls : list of np.ndarray
        A list of internal wall geometries, where each wall is represented as 
        a set of coordinates (x, y, z) forming a polygon.
    w_th : float
        A threshold distance to determine whether lines are too close to each 
        other or intersecting.
    w_width : float
        The width of the door to be defined at each intersection.
    w_high : float
        The height of the door to be defined at each intersection.

    Returns
    -------
    doors_final : list of np.ndarray
        A list of detected doors, each represented by a 2D array of coordinates
        (start and end points) with the correct height.
    non_intersecting_lines : list of list
        A list of lines that do not intersect with any walls or have been 
        filtered out based on proximity or intersection with others.


    """
    # Filter information. Convert lines to LineString
    lines_final_shapely = [LineString(line) for line in lines_final]    
    
    # Create a list to store the filtered lines
    filtered_lines = []
    
    # Iterate over the lines and compare distances or intersections
    for i, line1 in enumerate(lines_final_shapely):
        keep_line = True
        for j, line2 in enumerate(filtered_lines):
            # Calculate the minimum distance between the two lines
            distance = line1.distance(line2)
            # Check if the lines intersect or are too close
            if distance < w_th or line1.intersects(line2):
                keep_line = False
                break
        if keep_line:
            filtered_lines.append(line1)
    
    # Convert filtered lines back to arrays
    filtered_lines_final = [np.array(line.coords) for line in filtered_lines]
    
    doors_final = []
    lines_final = []
    non_intersecting_lines = []
    
    count = 0
    for line in filtered_lines_final:
        linea = LineString(line)
        door_added = False  
        for r in all_int_walls:
            if len(r) > 2 and len(r[0]) == 3:
                rectangulo = Polygon(r[:, 0:2])
                interseccion = linea.intersects(rectangulo)
                if interseccion and not door_added:
                    points_intersection = linea.intersection(rectangulo)
                    if not points_intersection.is_empty:
                        coordinates = list(
                            points_intersection.interpolate(
                                0.5, normalized=True).coords[0])
                        coordinates.append(np.min(r[:,2]))

                        count += 1
                        c = np.round(coordinates, 2)
                
                        orientation = calculate_rectangle_orientation(r[:,0:2])
                        # Determine if the wall is oriented in X or Y
                        if orientation == 'Y':
                            y_center = c[1]
                            x_center = c[0]  
                            # Define the ends of the door
                            door = np.array([[x_center, y_center - w_width],  
                                             [x_center, y_center + w_width]]) 
                        else:  
                            x_center = c[0]
                            y_center = c[1]  
                            # Define the ends of the door
                            door = np.array([[x_center - w_width, y_center], 
                                             [x_center + w_width, y_center]])
                            
                        p_s = np.array([[x, y, np.min(r[:,2])
                                         ] for x, y in door])
                        p_t = np.array([[x, y, np.min(r[:,2]) + w_high
                                         ] for x, y in door])

                    doors_final.append(np.round(np.vstack((p_s, p_t)), 2))
                    lines_final.append(line)
                    door_added = True  
                    break  
                
        if not door_added:
            non_intersecting_lines.append(line)       
                     
    # # Plot 
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for rect in all_int_walls:
    #     x_rect = [p[0] for p in rect] + [rect[0][0]]
    #     y_rect = [p[1] for p in rect] + [rect[0][1]]
    #     ax.plot(x_rect, y_rect, 'b-', linewidth=2)
    #     ax.fill(x_rect, y_rect, 'b', alpha=0.1)
    #     ax.plot(x_rect, y_rect, 'ro')
        
    # # Draw the points
    # plt.scatter(df_1m["x"], df_1m["y"], color="green", label="df_1m points")
    
    # # Draw the lines
    # for line in lines_final:
    #     plt.plot(line[:, 0], line[:, 1], color='red', lw=3)
    
    # ax.set_title("Non-overlapping walls")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.grid(True)
    # plt.show()
    
    return doors_final, non_intersecting_lines

#%%
def p_traj(df_1m, list_all_t, merged_rectangles, pairs, points_door):
    """
    Filters and organizes door pairs from the given trajectory data based on 
    symmetry and room information.

    Parameters
    ----------
    df_1m : pandas.DataFrame
        A DataFrame containing the 3D points sampled at 1-meter intervals
        along the trajectory, with the columns 'x', 'y', and 'z'.
    list_all_t : list of list
        A list where each element is a list of points at specific trajectory 
        intervals, with each point containing 3D coordinates.
    merged_rectangles : list of np.ndarray
        A list of merged rectangles representing walls or other geometrical
        features.
    pairs : list of list
        A list of door pairs, where each pair consists of two door endpoints
        represented by 2D coordinates.
    points_door : list of list
        A list of door points, where each point is a pair of 2D coordinates 
        representing the start and end of a door.

    Returns
    -------
    filtered_points_door : list of list
        A list of filtered door points, where symmetric pairs have been 
        grouped, and non-symmetric ones are retained as is.


    """
    df_1m['room'] = None
    v_room = []
    for id_list_b, list_b in enumerate(list_all_t):
        print(id_list_b)
        if len(list_b) == 0:
            print('None')
            v_room.append(None)
        elif len(list_b) != 0:
            third_column = np.array([arr[2] for arr in list_b])
            
            third_column_int = third_column.astype(int)
            
            counts = np.bincount(third_column_int)
            
            most_frequent_value = np.argmax(counts)
            
            v_room.append(most_frequent_value)
    
    df_1m['room'] = v_room
     
    # fig, ax = plt.subplots(figsize=(10, 10))  
    
    # for rect in merged_rectangles:
    #     x_rect = [p[0] for p in rect] + [rect[0][0]]  
    #     y_rect = [p[1] for p in rect] + [rect[0][1]]  
    #     ax.plot(x_rect, y_rect, 'b-', linewidth=2) 
    #     ax.fill(x_rect, y_rect, 'b', alpha=0.1)  
    #     ax.plot(x_rect, y_rect, 'ro')  
    
    # ax.scatter(df_1m['x'], df_1m['y'], c='green')  
    
    # for pair in points_door:
    #     x_line = [pair[0][0], pair[1][0]]  
    #     y_line = [pair[0][1], pair[1][1]]  
    #     ax.plot(x_line, y_line, 'r-', linewidth=2) 
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.grid(True)
    # ax.set_title('Walls and Candidates for Doors')
    
    # plt.show()
    
    symmetric_positions = []
    non_symmetric_positions = []
    
    checked_pairs = set()
    
    # Bucle para buscar pares simétricos
    for i in range(len(pairs)):
        found_symmetric = False 
        for j in range(i + 1, len(pairs)):
            if pairs[i] == pairs[j][::-1]:  
                symmetric_positions.append((i, j))
                checked_pairs.add(i)
                checked_pairs.add(j)
                found_symmetric = True
    
        if not found_symmetric and i not in checked_pairs:
            non_symmetric_positions.append(i)
    
    symmetric_positions, non_symmetric_positions
    
    first_column = [i for i, _ in symmetric_positions]
    
    if len(non_symmetric_positions) != 0:
        first_column.extend(non_symmetric_positions)
    
    filtered_points_door = [points_door[i] for i in first_column]
        
    # fig, ax = plt.subplots(figsize=(10, 10))  
    
    # for rect in merged_rectangles:
    #     x_rect = [p[0] for p in rect] + [rect[0][0]] 
    #     y_rect = [p[1] for p in rect] + [rect[0][1]] 
    #     ax.plot(x_rect, y_rect, 'b-', linewidth=2)  
    #     ax.fill(x_rect, y_rect, 'b', alpha=0.1)  
    #     ax.plot(x_rect, y_rect, 'ro') 
    
    # ax.scatter(df_1m['x'], df_1m['y'], c='green')  
    
    # for pair in filtered_points_door:
    #     x_line = [pair[0][0], pair[1][0]] 
    #     y_line = [pair[0][1], pair[1][1]]  
    #     ax.plot(x_line, y_line, 'r-', linewidth=2) 
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.grid(True)
    
    # ax.set_title('Walls and Candidates for Doors')
    
    # plt.show()

    return filtered_points_door


def orientation_lines(non_intersecting_lines, floor, w_width, w_high):
    """
    Determines the orientation of door lines and generates door geometry for 
    non-intersecting lines.

    Parameters
    ----------
    non_intersecting_lines : list of np.ndarray
        A list of non-intersecting door lines, where each line is represented
        by two 2D points (start and end coordinates).
    floor : object
        An object representing the floor, which contains information about
        the Z-coordinates of the floor level.
    w_width : float
        The width of the door, used to define the door opening along the Y or 
        X axis based on the orientation.
    w_high : float
        The height of the door, which is used to calculate the Z-coordinate of
        the door at the top of the door opening.

    Returns
    -------
    doors_non_intersecting : list of np.ndarray
        A list of door geometries, where each door is represented by an array 
        of points that define the start and end of the door at both the floor 
        and ceiling levels.
        

    """
    doors_non_intersecting = [] 
    point_mean_floor = np.round(np.mean(np.array(floor.points)[:,2]), 2)
    for line in non_intersecting_lines:
        midpoint = (line[0] + line[1]) / 2

        x1, y1 = line[0]
        x2, y2 = line[1]
        
        delta_x = abs(x2 - x1)
        delta_y = abs(y2 - y1)
        
        if delta_y > delta_x:
            orientation = 'Y'  
            door = np.array([[midpoint[0] - w_width, midpoint[1]], 
                             [midpoint[0] + w_width, midpoint[1]]])
            
        else:
            orientation = 'X'  
            door = np.array([[midpoint[0], midpoint[1] - w_width],  
                             [midpoint[0], midpoint[1] + w_width]]) 
        
        p_s = np.array([[x, y, point_mean_floor] for x, y in door])
        p_t = np.array([[x, y, point_mean_floor + w_high] for x, y in door])
        
        doors_non_intersecting.append(np.round(np.vstack((p_s, p_t)), 2))
        
        
    return doors_non_intersecting


