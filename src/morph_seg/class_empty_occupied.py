# -*- coding: utf-8 -*-

import open3d as o3d
from parameters import parameters_Bilbao as pm
import numpy as np
import pandas as pd
import itertools
from src.utils import array_to_o3d
from src.utils import bounding_polygon_filter


def class_empty_occupied(pcd_in, clean_idx, boundary_pol, floor_plane, 
                         ceiling_plane):
    """
    Classifies voxels in a clean point cloud as empty or occupied, and 
    classifies empty voxels as interior or exterior, using a boundary polygon
    and two planes (floor and ceiling) to determine the spatial classification.

    Parameters
    ----------
    pcd_in : pd.DataFrame
        DataFrame containing the (x, y, z) coordinates of the input point cloud.
    clean_idx : list or np.ndarray
        Indexes of the rows that correspond to the cleaned point cloud within 
        the DataFrame `pcd_in`.    
    boundary_pol :o3d.geometry.Polygon
        Boundary polygon used to filter out empty cloud points that are outside
        the defined area.
    floor_plane :np.ndarray
       4-dimensional vector representing the ground plane, in the form 
       [a, b, c, d], where ax + by + cz + d = 0 defines the plane.
    ceiling_plane : np.ndarray
       4-dimensional vector representing the roof plane, in the form [a, b, c, d],
       where ax + by + cz + d = 0 defines the plane.

    Returns
    -------
    labelled_vox_df : pd.DataFrame
        DataFrame containing the coordinates of the voxels (i, j, k) along with
        their class label (empty or occupied, and in the case of empty voxels,
        whether it is interior or exterior).
    vxl_idx_arr : np.ndarray
        Array of voxel indices with their corresponding labels. It has the form
        with the columns: index i, index j, index k, and the voxel label.
    rng_lst : list
        List with the dimensions of the voxel grid 
        (k_max - k_min, j_max - j_min, i_max - i_min).
    pcd_clean_arr : np.ndarray
        Array of the cleaned point cloud extracted from `pcd_in` according to 
        the `clean_idx` indexes.
    voxel_grid : o3d.geometry.VoxelGrid
        Voxel grid generated from the cleaned point cloud.
    """
    # Clean point cloud
    # xyz coordinates of cleaned point cloud
    pcd_clean_arr = pcd_in.loc[clean_idx, 'x': 'z'].values

    # Create VoxelGrid filtered point cloud    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        array_to_o3d.array_to_o3d_point_cloud(pcd_clean_arr), pm.voxel_size)

    # Classification of voxel as empty or occupied
    [i_min, j_min, k_min] = voxel_grid.get_voxel(voxel_grid.get_min_bound())
    [i_max, j_max, k_max] = voxel_grid.get_voxel(voxel_grid.get_max_bound())

    # Generate indexes from all voxels (occupied(10) and empty(0)) 
    rng_lst = [k_max-k_min, j_max-j_min, i_max-i_min]

    vxl_idx_arr = np.asarray([
        item for item in itertools.product(*[range(x+1) for x in rng_lst])])
    vxl_idx_arr[:, [0,2]] = vxl_idx_arr[:, [2,0]]

    # Save in class attribute
    pm.vox_labels['i'] = vxl_idx_arr[:, 0]
    pm.vox_labels['j'] = vxl_idx_arr[:, 1]
    pm.vox_labels['k'] = vxl_idx_arr[:, 2]           

    # Compute centroids
    vxl_cent = vxl_idx_arr * pm.voxel_size + voxel_grid.get_min_bound()

    xyz_min = pcd_clean_arr.min(axis=0)
    ijk_occ_vxl = np.rint((pcd_clean_arr - xyz_min)/pm.voxel_size)
    idx_occ_vxl = np.unique(np.dot(ijk_occ_vxl, 
                                    np.array([1, rng_lst[2]+1, 
                                              (rng_lst[2]+1)*(rng_lst[1]+1)]).T
                                    ).astype(int))

    vox_lbl = np.zeros(vxl_idx_arr.shape[0], dtype=int)
    vox_lbl[idx_occ_vxl] = pm.occ_lbl

    idx_empty = np.in1d(range(len(vox_lbl)), idx_occ_vxl)

    empty_cent = vxl_cent[~idx_empty, :]

    # Empty cloud
    empty_cloud = array_to_o3d.array_to_o3d_point_cloud(empty_cent)
    empty_cloud.paint_uniform_color([.0, .3, .9])
                                                                                              
        
    # Classify empty voxels as indoor (1) or outside (0)
    filt_empty_cloud, idx_filt = bounding_polygon_filter.bounding_polygon_filter(
        empty_cloud, boundary_pol, pm.zmax, pm.zmin, eliminate='outside')
    in_empty_pcd = empty_cloud.select_by_index(idx_filt)
    out_empty_pcd = empty_cloud.select_by_index(idx_filt, invert=True)
    in_empty_pcd.paint_uniform_color([.0, 0.9, .3])
    out_empty_pcd.paint_uniform_color([.0, 1., .5])

    # Compute distance from obstacle points to planes
    obs_v = np.ones((len(in_empty_pcd.points), 4))
    obs_v[:, :3] = np.asarray(in_empty_pcd.points)
    dist = -np.dot(obs_v, floor_plane)/np.linalg.norm(floor_plane[:3])
    idx1 = np.where(dist > 0)[0]
    in_filt_pcd1 = in_empty_pcd.select_by_index(idx1, invert=True)

    obs_v = np.ones((len(in_filt_pcd1.points), 4))
    obs_v[:, :3] = np.asarray(in_filt_pcd1.points)
    dist = -np.dot(obs_v, ceiling_plane)/np.linalg.norm(ceiling_plane[:3])
    idx2 = np.where(dist < 0)[0]
    in_filt_pcd2 = in_filt_pcd1.select_by_index(idx2, invert=True)

    # Index of indoor empty voxels
    idx1_inv = np.in1d(range(len(in_empty_pcd.points)), idx1)
    idx2_inv = np.in1d(range(len(in_filt_pcd1.points)), idx2)

    in_idx = np.where(~idx_empty)[0][idx_filt][~idx1_inv][~idx2_inv]

    # Set inner empty labels
    vox_lbl[in_idx] = pm.empty_in_lbl

    # Save in class attribute
    pm.vox_labels['occ'] = vox_lbl

    vxl_idx_arr = np.hstack((vxl_idx_arr, vox_lbl[:, np.newaxis]))
    labelled_vox_df = pd.DataFrame(vxl_idx_arr, columns=['i', 'j', 'k','scalar'])
    
    return labelled_vox_df, vxl_idx_arr, rng_lst, pcd_clean_arr, voxel_grid