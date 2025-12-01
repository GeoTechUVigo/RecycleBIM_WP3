# -*- coding: utf-8 -*-

import numpy as np


# Convert a numpy in a list
def convert_in_list(vectors):
    """
    Converts a list of vectors (tuples or arrays) into a list of lists.

    Parameters
    ----------
    vectors : list of tuples or arrays
        A list of vectors, where each vector can be a tuple or an array 
        containing numerical values.

    Returns
    -------
    vectors_without_numpy : list of lists
        A list of vectors, where each vector has been converted from a tuple 
        or array to a Python list.

    """
    vectors_without_numpy = []
    
    for vector in vectors:
        vector_without_numpy = list(vector)
        vectors_without_numpy.append(vector_without_numpy)
        
    return vectors_without_numpy

# Definition of the equation of the line
def plane_equation_xy(vectors, x,  y):
    """
    Computes the values of the plane equations for given vectors at 
    specific x and y coordinates.

    Parameters
    ----------
    vectors : list of tuples
        A list of vectors, where each vector is a tuple containing the 
        coefficients (a, b, c, d) of a plane equation.
    x : float
        The x-coordinate at which to evaluate the plane equations.
    y : float
        The y-coordinate at which to evaluate the plane equations.

    Returns
    -------
    equations : list of float
        A list of evaluated values for each plane equation at the specified
        x and y coordinates.

    """
    equations = []

    for vector in vectors:
        a , b, c, d = vector  
        equation = a * x + b * y + d
        equations.append(equation)

    return equation

# Round coefficients of each equation to two decimal:
def round_planes_coefficients(planes):
    """
    Rounds the coefficients of the plane equations to two decimal places.

    Parameters
    ----------
    planes : list of lists
        A list of plane equations, where each equation is represented as a 
        list of coefficients.

    Returns
    -------
    rounded_planes : list of lists
        A list of plane equations, where each coefficient has been rounded to 
        two decimal places.

    """
    rounded_planes = [[round(coeff, 2) for coeff in equation] for equation in planes]
    return rounded_planes

# See if two lines are parallel
def are_planes_parallel(plane1, plane2):
    """
    Checks if two planes are parallel by comparing their coefficients.

    Parameters
    ----------
    plane1 : list
        A list representing the coefficients (a, b, c) of the first
        plane equation.
    plane2 : list
        A list representing the coefficients (a, b, c) of the second 
        plane equation.

    Returns
    -------
    bool
        Returns True if the planes are parallel, and False otherwise.

    """
    # Coefficients a, b of the equations of the planes
    coef1 = plane1[:2]
    coef2 = plane2[:2]
    # Check if the slopes are equal
    if coef1[0]*coef2[1] == coef1[1]*coef2[0]:
        return True
    else:
        return False

# Calculation of the angle between lines
def calculate_angle(line1, line2):
    """
    Calculates the angle between two lines based on their direction vectors.


    Parameters
    ----------
    line1 : list
        A list representing the coefficients (A, B) of the first line equation,
        defining the direction vector.
    line2 : list
        A list representing the coefficients (A, B) of the second line equation,
        defining the direction vector.

    Returns
    -------
    angle_degree : float
        The angle between the two lines in degrees. Returns NaN if the angle
        cannot be computed (e.g., division by zero).

    """
    # Convert lists to NumPy arrays
    line1 = np.array(line1)
    line2 = np.array(line2)

    # Extract the direction components of the lines
    direction1 = line1[:2]
    direction2 = line2[:2]

    # Calculate the dot product and the magnitudes of the vectors
    product_point = np.dot(direction1, direction2)
    magn1 = np.linalg.norm(direction1)
    magn2 = np.linalg.norm(direction2)

    # Calculate the angle in radians
    if -1 <= product_point / (magn1 * magn2) <= 1:
        angle_radians = np.arccos(product_point / (magn1 * magn2))
    else:
        angle_radians = np.nan
    # Convert angle to degrees
    angle_degree = np.degrees(angle_radians)

    return angle_degree

# Check if two planes are perpendicular
def are_planes_perpendicular(plane1, plane2):
    """
    Checks if two planes are perpendicular by examining the dot product of 
    their normal vectors.

    Parameters
    ----------
    plane1 : list
        A list representing the coefficients (a, b, c) of the first plane equation.
    plane2 : list
        A list representing the coefficients (a, b, c) of the second plane equation.

    Returns
    -------
    bool
        Returns True if the planes are perpendicular, and False otherwise.

    """
    # Coefficients a, b of the equations of the planes
    coef1 = plane1[:2]
    coef2 = plane2[:2]

    # Check if the normal vectors are perpendicular
    return coef1[0] * coef2[0] + coef1[1] * coef2[1] == 0

def find_intersection_point(plane1, plane2):
    """
    Finds the intersection point of two planes, if it exists.

    Parameters
    ----------
    plane1 : list
        A list representing the coefficients (a, b, d) of the first plane 
        equation.
    plane2 : list
        A list representing the coefficients (a, b, d) of the second plane 
        equation.

    Returns
    -------
    tuple or None
        A tuple (x, y) representing the intersection point of the planes, or 
        None if the planes are parallel.

    """
    a1, b1, d1 = plane1[:3]
    a2, b2, d2 = plane2[:3]

    # Solve the system of linear equations to find the point of intersection
    determinant = a1 * b2 - a2 * b1

    if determinant != 0:
        x = (b1 * d2 - b2 * d1) / determinant
        y = (a2 * d1 - a1 * d2) / determinant
        return x, y
    else:
        # Parallel planes, there is no single point of intersection
        return None
    
def distance_lines_paralell(line1, line2):
    """
    Calculates the distance between two parallel lines.

    Parameters
    ----------
    list
       A list representing the coefficients (A, B, C) of the first line equation.
   line2 : list
       A list representing the coefficients (A, B, C) of the second line equation.

   Returns
   -------
   float
       The perpendicular distance between the two parallel lines.

    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    
    # Calculate the distance between the two parallel lines
    distance = abs(C2 - C1) / ((A1**2 + B1**2)**0.5)
    return distance

# Calculate centroid of a two-dimensional point cloud
def calculate_centroid(point_cloud):
    """
    Calculates the centroid (geometric center) of a point cloud.


    Parameters
    ----------
    point_cloud : ndarray
        A 2D NumPy array where each row represents a point in the point cloud, 
        and the columns represent the x and y coordinates of the points.

    Returns
    -------
    x_c : float
        The x-coordinate of the centroid.
    y_c : float
        The y-coordinate of the centroid.

    """
    # Add the coordinates
    sum_x = np.sum(point_cloud[:, 0])
    sum_y = np.sum(point_cloud[:, 1])

    # Count number of points
    n = len(point_cloud)

    # Calculate the weighted average
    x_c = sum_x / n
    y_c = sum_y / n

    return x_c, y_c

# Calculate distance between centroids
def calculate_distance_centroids(centroid1, centroid2):
    """
    Calculates the Euclidean distance between two centroids.

    Parameters
    ----------
    centroid1 : tuple
        A tuple representing the (x, y) coordinates of the first centroid.
    centroid2 : tuple
        A tuple representing the (x, y) coordinates of the second centroid.

    Returns
    -------
    distance : float
        The Euclidean distance between the two centroids.

    """
    # Calculate differences in coordinates
    diff_x = centroid2[0] - centroid1[0]
    diff_y = centroid2[1] - centroid1[1]

    # Calculate Euclidean distance 
    distance = np.sqrt(diff_x**2 + diff_y**2)

    return distance


def distance(point1, point2):
    """
    Calculates the Euclidean distance between two points in a 2D plane.


    Parameters
    ----------
    point1 : tuple
        A tuple representing the (x, y) coordinates of the first point.
    point2 : tuple
        A tuple representing the (x, y) coordinates of the second point.

    Returns
    -------
    float
        The Euclidean distance between the two points.

    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def points_in_buffer(points, intersection_point, buffer_radius):
    """
    Finds the points that lie within a specified buffer radius around a given
    intersection point.

    Parameters
    ----------
    points : list of tuples
        A list of tuples, where each tuple represents the (x, y) coordinates 
        of a point.
    intersection_point : tuple
        A tuple representing the (x, y) coordinates of the intersection point.
    buffer_radius : float
        The radius of the buffer around the intersection point.

    Returns
    -------
    points_within_buffer : list of tuples
        A list of points that lie within the specified buffer radius from the 
        intersection point.

    """
    points_within_buffer = []
    for point in points:
        dist = distance(point, intersection_point)
        if dist <= buffer_radius:
            points_within_buffer.append(point)
    return points_within_buffer


