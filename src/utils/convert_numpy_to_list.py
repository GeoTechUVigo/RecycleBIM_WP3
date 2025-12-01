# -*- coding: utf-8 -*-

import numpy as np

def convert_numpy_to_list(d):
    """
    Recursively converts all numpy.ndarray objects in a dictionary or list to 
    Python lists.


    Parameters
    ----------
    d : dict, list, or numpy.ndarray
        The input data that may contain numpy arrays, lists, or other types of 
        data.

    Returns
    -------
    dict, list, or other
        A new structure where all numpy arrays have been converted to lists.

    """
    if isinstance(d, dict):
        return {k: convert_numpy_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_numpy_to_list(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d
