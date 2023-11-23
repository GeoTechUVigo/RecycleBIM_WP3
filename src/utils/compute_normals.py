# -*- coding: utf-8 -*-

import pandas as pd
from pyntcloud import PyntCloud

def compute_normals(pts_arr, k=2):
    """
    Calculate normals on a set of three-dimensional points.

    Parameters
    ----------
    pts_arr : numpy array
        Three dimensional dot set.
    k : int, optional
        the k points closest to each point in the cloud will be searched. 
        The default is 2.

    Returns
    -------
    Numpy array
        Calculated normal values.

    """
    
    pynt_pcd = PyntCloud(pd.DataFrame(pts_arr, columns=['x', 'y', 'z']))
    k_neighbors = pynt_pcd.get_neighbors(k=k)
    
    normals = pynt_pcd.add_scalar_field("normals", k_neighbors= k_neighbors)
    
    return pynt_pcd.points.loc[:, normals].values

