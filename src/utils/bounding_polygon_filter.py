# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.path import Path


def bounding_polygon_filter(cloud, polygon, zmax, zmin, eliminate='outside'):
    """
    Function to apply bounding polygon filter.

    Parameters
    ----------
    cloud : PointCloud object of Open3D
        Open3d cloud to filter.
    polygon : Geometric Object
        Shapely polygon.
    zmax : int
        Maximum heigth of polygon filter.
    zmin : int
        Minimum heigth of polygon filter.
    eliminate : str, optional
        Choose if filter keeps inside points or outside points of polygon box
            Options: 'inside' --> Remove points from inside polygon box
                     'outside' --> Remove points from outside polygon box any 
                                   other posibility will be interpreted as 
                                   'outside'. 
            The default is 'outside'.

    Returns
    -------
    cloudFiltered : PointCloud object of Open3D
        Open3d filtered cloud.
    iii : numpy array
        Index of points to eliminate.

    """

    if not isinstance(cloud, np.ndarray):
        points = np.asarray(cloud.points)[:,0:3]
    else:
        points = cloud
    
    idx_heigth = np.logical_and(points[:, 2]<zmax, points[:, 2]>zmin)
    
    poly_coords = np.asarray(polygon.exterior.coords)
    p = Path(poly_coords) # Make a polygon
    idx = p.contains_points(points[:, 0:2])
    idx_final = np.logical_and(idx_heigth==True, idx==True)
    
    if eliminate=='outside': 
        cloudFiltered = cloud.select_by_index(np.where(idx_final==True)[0])
        iii = np.where(idx_final==True)[0]
    elif eliminate=='inside': 
        cloudFiltered = cloud.select_by_index(np.where(idx_final==True)[0], invert=True)
        iii = np.where(idx_final==False)[0]
    else: cloudFiltered = cloud.select_by_index(np.where(idx_final==True)[0])
        
    return cloudFiltered, iii