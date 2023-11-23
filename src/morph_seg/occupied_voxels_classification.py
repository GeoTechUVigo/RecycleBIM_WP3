# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import mode

from src.utils import get_grid_index


def occupied_voxels_classification(labelled_vox_df, vxl_idx_arr, whole_dilated_image, rng_lst, occ_lbl):
    """
    Classify occupied voxels from classified empty voxel according 
    proximity and add room labels to VoxelGrid matrix.

    Parameters
    ----------
    labelled_vox_df : DataFrame
        Data structure also contains labeled axes (rows and columns).
        Arithmetic operations align on both row and column labels. Can
        be thought of as a dict-like container for Series objects. The
        primary pandas data structure.
    whole_dilated_image : numpy array 
        Dilation performed on individual segmented rooms.
    occ_lbl : int

    Returns
    -------
    room_lab_arr : numpy array
        Room labels added to VoxelGrid array.
    segmented_3d_image : numpy array
        
    """
    # Classify occupied voxels from classified empty voxel according proximity
    occupied_voxels = labelled_vox_df.loc[labelled_vox_df.scalar==occ_lbl, 'i':'k'].values

    segmented_3d_image = np.zeros_like(whole_dilated_image)

    # Add room labels to VoxelGrid matrix
    room_lab_arr = np.ones(vxl_idx_arr.shape[0])*-1

    for n_vox in range(occupied_voxels.shape[0]):                
        i, j, k = occupied_voxels[n_vox]                
        vxl_idx = get_grid_index.get_grid_index(i, j, k, rng_lst[2]+1, 
                                            rng_lst[1]+1, rng_lst[0]+1)                
        assigned = False
        level = 1
        while not assigned:
            nbr_values = whole_dilated_image[max(0,k-level):k+level+1,
                                             max(0,j-level):j+level+1, 
                                             max(0,i-level):i+level+1]     
           
            if not any(nbr_values.flatten()):
                level += 1
            else:
                id_room = mode(nbr_values.flatten()[nbr_values.flatten() > 0], keepdims=False)
                segmented_3d_image[k, j, i] = id_room[0]
                room_lab_arr[vxl_idx] = id_room[0]
                assigned= True
               
    return room_lab_arr, segmented_3d_image