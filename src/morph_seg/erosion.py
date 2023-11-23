# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import binary_erosion



def morphological_erosion(labelled_vox_df, se, width_door, empty_in_lbl, voxel_size):
        """
        The first morphological operation consists in a erosion applied to
        indoor empty space with the aim of breaking the empty space 
        continuity where it is weakest, generally in doorways.

        Parameters
        ----------
        labelled_vox_df : DataFrame
            Data structure also contains labeled axes (rows and columns).
            Arithmetic operations align on both row and column labels. Can
            be thought of as a dict-like container for Series objects. The
            primary pandas data structure.               
        se : array_like
            Structuring element used for the erosion. Non-zero elements are 
            considered True. If no structuring element is provided, an element
            is generated with a square connectivity equal to one.
        width_door : float
            Estimated door size
            Is measured in meters.
        empty_in_lbl : int 
            
        voxel_size : float
            Voxel size
            Is measured in meters.

        Returns
        -------
        image_erode : ndarray of bools
            Erosion of the input by the structuring element.
        se_erosion : array_like
            Structuring element used for the erosion. Non-zero elements are 
            considered True. If no structuring element is provided, an element
            is generated with a square connectivity equal to one.

        """
        # Generate 3D image from voxelized data 
        image_3d = np.zeros((labelled_vox_df.loc[:, 'i':'k'].max().values[::-1]+1), int)

        # DataFrame Indices of indoor empty voxels
        index_indoor_empty_voxels = labelled_vox_df[labelled_vox_df.scalar==empty_in_lbl].index
        grid_coord_indoor_empty_voxels = labelled_vox_df.loc[index_indoor_empty_voxels, 'i':'k']

        image_3d[grid_coord_indoor_empty_voxels.loc[:, 'k'].values,
                  grid_coord_indoor_empty_voxels.loc[:, 'j'].values,
                  grid_coord_indoor_empty_voxels.loc[:, 'i'].values] = 1
        
        # Define structuring element 
        if se is None:
            res_vox = voxel_size
            l = np.ceil((width_door/res_vox)/2.0) + np.ceil((width_door/res_vox)*0.1)+1
            size_cube = int(l)            
            se_erosion = np.ones((size_cube, size_cube, size_cube))
        else:
            se_erosion = se
            
        image_erode = binary_erosion(image_3d, structure=se_erosion).astype(image_3d.dtype)
    
        return image_erode, se_erosion

