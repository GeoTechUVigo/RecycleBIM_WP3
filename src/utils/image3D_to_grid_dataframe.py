# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools

def image_3d_to_grid_dataframe(image_3d):
    """
    Generate voxelized data from image 3d. 

    Parameters
    ----------
    image_3d : numpy array
        Is the image 3D generated.

    Returns
    -------
    grid_df : DataFrame
        Is the voxelized data.

    """
    
    rng_lst = image_3d.shape

    vxl_idx_arr = np.asarray([item for item in itertools.product(*[range(x) for x in rng_lst])])
    vxl_idx_arr[:, [0,2]] = vxl_idx_arr[:, [2,0]]
    
    vxl_idx_arr = np.hstack((vxl_idx_arr, image_3d.flatten()[:, np.newaxis]))
    grid_df = pd.DataFrame(vxl_idx_arr, columns=['i', 'j', 'k','scalar'])
    
    return grid_df


