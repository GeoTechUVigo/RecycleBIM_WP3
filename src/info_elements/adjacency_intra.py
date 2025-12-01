# -*- coding: utf-8 -*-

def intersection(line1, line2):
    """
    Calculate the intersection point of two lines

    Parameters
    ----------
    line1 : list
        Coefficients of the general equation of one of the lines.
    line2 : list
        Coefficients of the general equation of another line.

    Returns
    -------
    x : float
        x-coordinate of the intersection point.
    y : float
        y-coordinate of the intersection point.

    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9: 
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return (x, y)

def adjacent(lines_position_t):
    """
    Check which lines intersect and, consequently, indicate which 
    walls are adjacent.

    Parameters
    ----------
    lines_position_t : list
        Coefficients of the general equation of the lines.

    Returns
    -------
    adjacency_complete : list
        Indicates the index of the walls that are adjacent.

    """
    adjacency_complete = []
    for lines in lines_position_t:
        adjacency = []
        n = len(lines)
    
        for i in range(n):
            for j in range(i + 1, n):
                inter = intersection(lines[i], lines[j])
                if inter:
                    adjacency.append([i, j])
        adjacency_complete.append(adjacency)
    
    return adjacency_complete
    
    
    
    