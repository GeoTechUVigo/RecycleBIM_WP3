# -*- coding: utf-8 -*-


import numpy as np

def search_indexes(df, array):
    """
    Looks for points that are in the numpy in a dataframe and recovers
    the indexes.

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    array : Array of float64
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    indexes = np.where((df['x'].values[:, None] == array[:, 0]) &
                       (df['y'].values[:, None] == array[:, 1]) &
                       (df['z'].values[:, None] == array[:, 2]))
    return indexes[0]
