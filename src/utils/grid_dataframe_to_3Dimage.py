# -*- coding: utf-8 -*-

import numpy as np

def grid_dataframe_to_3d_image(grid_df):
    """
    Generate 3D image from voxelized data. 

    Parameters
    ----------
    grid_df : DataFrame
        Is the voxelized data.

    Returns
    -------
    image_3d : numpy array
        Is the image 3D generated.

    """
    
    image_3d = np.zeros((grid_df.loc[:, 'i':'k'].max().values[::-1]+1), int)

    uni_lbls = np.unique(grid_df.iloc[:, 3].values)

    for lbl in uni_lbls:        
        # DataFrame Indices of indoor empty voxels
        index_indoor_empty_voxels = grid_df[grid_df.iloc[:,3]==lbl].index
        grid_coord_indoor_empty_voxels = grid_df.loc[index_indoor_empty_voxels, 'i':'k']
   
        image_3d[grid_coord_indoor_empty_voxels.loc[:, 'k'].values,
                 grid_coord_indoor_empty_voxels.loc[:, 'j'].values,
                 grid_coord_indoor_empty_voxels.loc[:, 'i'].values] = lbl
 
    
    return image_3d