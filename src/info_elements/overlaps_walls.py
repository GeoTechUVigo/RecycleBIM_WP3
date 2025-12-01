# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def overlap(sel):
    """
    Calculates the bounding rectangles of the floors of rooms, based on the 
    floor points, and visualizes them.


    Parameters
    ----------
    sel : list of dicts
        A list of selected room data, where each room is represented by a 
        dictionary containing at least a "points_floor" key. 

    Returns
    -------
    rectangles_points : list of lists
        A list containing the coordinates of the bounding rectangles. 
    
    mean_z : float
        The average Z-coordinate of all the floor points across all the 
        selected rooms. 

    """
    rectangles_points = []
    for data in sel:
        points = []
        for room in data:
            for point in room['points_floor']:
                points.append(point)

        if len(points) != 0:
            # Extract the X and Y coordinates (ignore Z for the rectangle)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            z_coords = [p[2] for p in points]

            # Find the minima and maxima in X and Y
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            z_mean = np.round(np.mean(z_coords), 2)

            # Define the vertices of the rectangle
            rectangle_F = [
                (x_min, y_min, z_mean),
                (x_min, y_max, z_mean),
                (x_max, y_max, z_mean),
                (x_max, y_min, z_mean)
            ]

            # Print the vertices of the rectangle
            rectangles_points.append(rectangle_F)

    # # Represent all rectangles
    # plt.figure(figsize=(10, 10))

    # # Iterate over the rectangles to graph
    # for rect in rectangles_points:
    #     # Extract the X and Y coordinates of the vertices of the rectangle
    #     x_rect = [p[0] for p in rect] + [rect[0][0]]
    #     y_rect = [p[1] for p in rect] + [rect[0][1]]

    #     # Graph the rectangle
    #     plt.plot(x_rect, y_rect, color='blue')
    #     plt.scatter(x_rect, y_rect, color='red')

    # # Setting up the axes and title
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Visualizing Rectangles of Walls")
    # plt.grid(True)
    # plt.show()

    # Extraemos las coordenadas Z de todos los puntos
    z_coords = [point[2] 
                for rectangle in rectangles_points for point in rectangle]

    # Calculamos la media de las coordenadas Z
    mean_z = np.round(np.mean(z_coords), 2)

    return rectangles_points, mean_z
#%%
# Function to create a polygon from a rectangle
def create_polygon(rect):
    return Polygon(rect)

# Function to check if two rectangle buffers intersect
def check_buffer_overlap(rect1, rect2, buffer_distance=0.05):
    """
    Checks if two rectangles (after applying a buffer zone) overlap or intersect.

    Parameters
    ----------
    rect1 : list of tuples
        The first rectangle defined by four points, each in the format (x, y, z). 
    rect2 : list of tuples
        The second rectangle defined by four points, each in the format (x, y, z). 
    buffer_distance : float, optional
        The distance of the buffer zone to be applied around the polygons. 
        The default value is 0.05. A larger value will increase the size of the
        buffer zone.

    Returns
    -------
    bool
        Returns `True` if the buffered polygons (rectangles) overlap or intersect, 
        and `False` otherwise.

    """
    poly1 = create_polygon(rect1)
    poly2 = create_polygon(rect2)
    
    buffered_poly1 = poly1.buffer(buffer_distance)
    buffered_poly2 = poly2.buffer(buffer_distance)
    
    return buffered_poly1.intersects(buffered_poly2)

# Function to obtain the coordinates of a rectangle
def get_rectangle_coords(rect):
    """
    Extracts the X and Y coordinates from a given rectangle and closes the
    polygon by appending the first point at the end to form a complete loop.

    Parameters
    ----------
    rect : list of tuples
        A list of four points defining the rectangle, each point in the format 
        (x, y, z). 

    Returns
    -------
    x_values : list of floats
        A list of the X coordinates of the rectangle's vertices, with the first
        point repeated at the end to close the loop.
    y_values : list of floats
        A list of the Y coordinates of the rectangle's vertices, with the first
        point repeated at the end to close the loop.

    """
    x_values = [point[0] for point in rect]
    y_values = [point[1] for point in rect]
    x_values.append(x_values[0])  
    y_values.append(y_values[0])  
    return x_values, y_values

# Function to merge two overlapping rectangles
def merge_rectangles(rect1, rect2):
    """
    Merges two rectangles by calculating the union of their areas and returns
    the coordinates of the bounding box of the merged area. 

    Parameters
    ----------
    rect1 : list of tuples
        The first rectangle, defined by four points (x, y) that represent the 
        corners of the rectangle. 
    rect2 : list of tuples
        The second rectangle, defined by four points (x, y) in the same format 
        as rect1.

    Returns
    -------
    merged_rect : list of tuples
        A list of four points representing the bounding box of the merged area, in the form 
        [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]. 

    """
    poly1 = create_polygon(rect1)
    poly2 = create_polygon(rect2)
    
    # Create the union of the polygons (this returns the total area covered by both)
    merged_poly = poly1.union(poly2)
    
    # Get the coordinates of the new rectangle
    minx, miny, maxx, maxy = merged_poly.bounds
    merged_rect = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    
    return merged_rect

# Function to get the center of the rectangle
def get_center(rect):
    """
    Calculates the center point of a rectangle based on the average of its 
    corner coordinates. 
    

    Parameters
    ----------
    rect : list of tuples
        The rectangle, defined by four points (x, y) that represent the corners
        of the rectangle. 

    Returns
    -------
    center_x : float
        The X-coordinate of the center of the rectangle.
    center_y : float
        The Y-coordinate of the center of the rectangle.

    """
    x_values, y_values = get_rectangle_coords(rect)
    center_x = sum(x_values) / len(x_values)
    center_y = sum(y_values) / len(y_values)
    return center_x, center_y

# Find rectangles that overlap or are close
def find_and_merge_nearby_rectangles(rectangles):
    """
    Finds and merges nearby (overlapping or touching) rectangles from a list of rectangles. 

    Parameters
    ----------
    rectangles : list of list of tuples
        A list of rectangles, where each rectangle is defined by a list of four tuples. Each tuple represents
        the (x, y, z) coordinates of a corner point. The Z-coordinate is ignored when checking for overlap and merging.

    Returns
    -------
    merged_rectangles : list of list of tuples
        A list of rectangles after merging the nearby overlapping rectangles. Each rectangle is represented by
        a list of four tuples, which define the corners of the merged rectangle.

    """
    merged_rectangles = rectangles.copy()
    nearby_rectangles = []
    
    while True:
        nearby_rectangles.clear()
        for i in range(len(merged_rectangles)):
            for j in range(i + 1, len(merged_rectangles)):
                if check_buffer_overlap(merged_rectangles[i], merged_rectangles[j]):
                    nearby_rectangles.append((i, j))
        
        # If there are no more overlapping rectangles, finish.
        if not nearby_rectangles:
            break
        
        # Merge overlapping rectangles
        merged_rectangles_new = []
        used_indices = set()
        
        for i, j in nearby_rectangles:
            if i not in used_indices and j not in used_indices:
                merged_rect = merge_rectangles(merged_rectangles[i], merged_rectangles[j])
                merged_rectangles_new.append(merged_rect)
                used_indices.add(i)
                used_indices.add(j)
        
        # Add unused rectangles
        for k in range(len(merged_rectangles)):
            if k not in used_indices:
                merged_rectangles_new.append(merged_rectangles[k])
        
        merged_rectangles = merged_rectangles_new
    
    return merged_rectangles

def walls_to_IFC(rectangles, mean_z):
    """
    Visualizes and processes a set of rectangles representing walls, merging 
    nearby rectangles and assigning a Z-coordinate (mean_z) to the 2D 
    projections. 

    Parameters
    ----------
    rectangles : list of list of tuples
        A list of rectangles, where each rectangle is defined by four tuples 
        representing (x, y) coordinates of its corners. 
    mean_z : float
        The average Z-coordinate (height) to be assigned to all the rectangles 
        after processing them.

    Returns
    -------
    all_rect : list of list of tuples
        A list of all processed rectangles, including merged ones, with the 
        Z-coordinate set to the provided `mean_z`.


    """
    # # Create the figure and the axes
    # fig, ax = plt.subplots()
    
    # # Draw each rectangle
    # for rect in rectangles:
    #     x_values = [point[0] for point in rect]
    #     y_values = [point[1] for point in rect]
    #     x_values.append(x_values[0])  
    #     y_values.append(y_values[0])  
    #     ax.plot(x_values, y_values, 'b-', lw=1)  
    
    # # Tags and Title
    # ax.set_xlabel('Coordinate X')
    # ax.set_ylabel('Coordinate Y')
    # ax.set_title('Rectangles in 2D')
    
    # # Show chart
    # plt.grid(True)
    # plt.show()
    
    # Initialize lists to sort rectangles
    x_oriented = []
    y_oriented = []
    
    # Classify rectangles
    for rect in rectangles:
        x_values = [point[0] for point in rect]
        y_values = [point[1] for point in rect]
        
        diff_x = max(x_values) - min(x_values)
        diff_y = max(y_values) - min(y_values)
        
        if diff_x > diff_y:
            x_oriented.append(rect)  
        else:
            y_oriented.append(rect)  
    
    # # Create the figure and the axes
    # fig, ax = plt.subplots()
    
    # # Draw rectangles oriented along the X axis in red
    # for rect in x_oriented:
    #     x_values = [point[0] for point in rect]
    #     y_values = [point[1] for point in rect]
    #     x_values.append(x_values[0]) 
    #     y_values.append(y_values[0]) 
    #     ax.plot(x_values, y_values, 'r-', lw=2)  
    
    # # Draw rectangles oriented on the Y axis in blue
    # for rect in y_oriented:
    #     x_values = [point[0] for point in rect]
    #     y_values = [point[1] for point in rect]
    #     x_values.append(x_values[0])  
    #     y_values.append(y_values[0])  
    #     ax.plot(x_values, y_values, 'b-', lw=2)  
    
    # # Etiquetas y título
    # ax.set_xlabel('Coordinate X')
    # ax.set_ylabel('Coordinate Y')
    # ax.set_title('Rectangles Oriented in X (red) and Y (blue)')
    
    # # Show chart
    # plt.grid(True)
    # plt.show()

    # Find and merge nearby rectangles
    merged_rectangles_x = find_and_merge_nearby_rectangles(x_oriented)
    
    # # Displaying the merged and original rectangles
    # fig, ax = plt.subplots()
    
    # # Draw all original rectangles in gray
    # for rect in y_oriented:
    #     x_values, y_values = get_rectangle_coords(rect)
    #     ax.plot(x_values, y_values, 'k-', lw=2)  
    
    # # Draw the merged rectangles in green
    # for rect in merged_rectangles_x:
    #     x_values, y_values = get_rectangle_coords(rect)
    #     ax.plot(x_values, y_values, 'g-', lw=3)  
    
    # # Label the rectangles with their number
    # for i, rect in enumerate(merged_rectangles_x):
    #     center_x, center_y = get_center(rect)
    #     ax.text(center_x, center_y, str(i), color='red', ha='center', va='center', fontsize=12)
    
    # # Etiquetas y título
    # ax.set_xlabel('Coordinate X')
    # ax.set_ylabel('Coordinate Y')
    # ax.set_title('Original and Merged Rectangles')
    
    # # Show chart
    # plt.grid(True)
    # plt.show()
    
    # Find and merge nearby rectangles
    merged_rectangles_y = find_and_merge_nearby_rectangles(y_oriented)
    
    # # Displaying the merged and original rectangles
    # fig, ax = plt.subplots()
    
    # # Draw all original rectangles in gray
    # for rect in y_oriented:
    #     x_values, y_values = get_rectangle_coords(rect)
    #     ax.plot(x_values, y_values, 'k-', lw=2)  
    
    # # Draw the merged rectangles in green
    # for rect in merged_rectangles_y:
    #     x_values, y_values = get_rectangle_coords(rect)
    #     ax.plot(x_values, y_values, 'g-', lw=3) 
    
    # # Label the rectangles with their number
    # for i, rect in enumerate(merged_rectangles_y):
    #     center_x, center_y = get_center(rect)
    #     ax.text(center_x, center_y, str(i), color='red', ha='center', va='center', fontsize=12)
    
    # # Tags and Title
    # ax.set_xlabel('Coordinate X')
    # ax.set_ylabel('Coordinate Y')
    # ax.set_title('Original and Merged Rectangles')
    
    # # Show chart
    # plt.grid(True)
    # plt.show()
    
    all_rect = merged_rectangles_x + merged_rectangles_y

    # Coordinate Z
    for i, rect in enumerate(all_rect):
        all_rect[i] = [(point[0], point[1], mean_z) if len(point) == 2 else point for point in rect]
    
    # # Create the figure and the axes
    # plt.figure(figsize=(8, 8))
    
    # # Plot the points
    # for conjunto in all_rect:
    #     xs = [p[0] for p in conjunto]
    #     ys = [p[1] for p in conjunto]
    #     plt.plot(xs + [xs[0]], ys + [ys[0]], marker='o') 
    
    # # Setting up the axes
    # plt.xlabel('Axis X')
    # plt.ylabel('Axis Y')
    # plt.title('2D Point Projection')
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')  
    
    # # Show the graph
    # plt.show()

    # # Plot
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for rect in all_rect:
    #     x_rect = [p[0] for p in rect] + [rect[0][0]]
    #     y_rect = [p[1] for p in rect] + [rect[0][1]]
    #     ax.plot(x_rect, y_rect, 'b-', linewidth=2)
    #     ax.fill(x_rect, y_rect, 'b', alpha=0.1)
    #     ax.plot(x_rect, y_rect, 'ro')
    
    # ax.set_title("Large non-overlapping rectangles")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.grid(True)
    # plt.show()
    
    
    return all_rect








