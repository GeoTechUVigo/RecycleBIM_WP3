# -*- coding: utf-8 -*-

import numpy as np


def point_cloud_classification(pcd_in, pcd_clean_arr, room_lab_arr, voxel_grid, voxel_size):
    """
    The classification of the point cloud is based on a proximity
    assignment between the voxels belonging to each room and the
    nearby points.
    Occupied voxels are labelled regarding proximity of labelled
    empty voxel. For each occupied voxel, a set of neighboring
    voxels are evaluated. If any of them was labelled in the previous
    steps, occupied voxel is classified in the same way. In case there
    are not a unique label, the most frequent one is selected to
    classify occupied voxel. 

    Parameters
    ----------
    pcd_clean_arr : numpy array
        Cleaned point cloud.
    room_lab_arr : numpy array
        Room labels added to VoxelGrid array..
    voxel_size : float
        Voxel size
        Is measured in meters.

    Returns
    -------
    class_pts_arr : numpy array
        Indicates the room number to which it belongs.
    idx_in_pcd : numpy array
        Index of the room.

    """
    class_pts_arr = np.array([], dtype=int)

    idx_in_pcd = np.array([], dtype=int)
    # Minimum xyz coordinates of cleaned point cloud
    xyz_min = pcd_clean_arr.min(axis=0) 
        
    vxl_pts = np.rint((pcd_in.loc[:, 'x':'z'].values - xyz_min)/voxel_size)

    # VoxelGrid bound limits
    [i_min, j_min, k_min] = voxel_grid.get_voxel(voxel_grid.get_min_bound())
    [i_max, j_max, k_max] = voxel_grid.get_voxel(voxel_grid.get_max_bound())
    
    rng_lst = [k_max-k_min, j_max-j_min, i_max-i_min]
    
    # Remove indexes exceeding bounds
    i_rem = np.where((vxl_pts[:, 0]>i_max) | (vxl_pts[:, 0]<i_min))[0]
    j_rem = np.where((vxl_pts[:, 1]>j_max) | (vxl_pts[:, 1]<j_min))[0]
    k_rem = np.where((vxl_pts[:, 2]>k_max) | (vxl_pts[:, 2]<k_min))[0]
    
    ijk_rem = np.concatenate((i_rem, j_rem, k_rem))
    
    idx_bool_filt = ~np.isin(np.arange(vxl_pts.shape[0]), ijk_rem)
    
    vxl_pts =  vxl_pts[idx_bool_filt]
    
    vxl_idx = np.dot(vxl_pts, np.array([1, rng_lst[2]+1, (rng_lst[2]+1)*(rng_lst[1]+1)]).T).astype(int)
    
    vxl_lbls_arr = room_lab_arr[vxl_idx].astype(int)
    
    label_pcd_in_arr = np.ones(pcd_in.shape[0]) * -1
    
    idx_in_pcd = pcd_in.index.values[idx_bool_filt][vxl_lbls_arr != -1]
    class_pts_arr = vxl_lbls_arr[vxl_lbls_arr != -1]
    
    label_pcd_in_arr[idx_in_pcd] = class_pts_arr

    # Add room labels to DataFrame of the input point cloud
    pcd_in['room'] = label_pcd_in_arr
    
    return class_pts_arr, idx_in_pcd
