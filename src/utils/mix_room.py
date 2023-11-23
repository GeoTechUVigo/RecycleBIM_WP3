# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import pandas as pd
import itertools
from matplotlib import cm

from src.utils import array_to_o3d
from src.utils import bounding_polygon_filter
from src.preprocessing import subsampling as sb
from src.preprocessing import clean_exteriors
from src.elements_seg import extract_floor_ceiling as efc
from src.utils import remove_no_continuous as rop
from src.morph_seg import erosion
from src.morph_seg import individualisation
from src.morph_seg import dilation
from src.morph_seg import occupied_voxels_classification as ovc
from src.morph_seg import point_cloud_classification as pcc




def mix_room(pcd, n_room, nt, ransac_th, eps, pt, voxel_size, alpha, zmax, zmin,
             dist, vox_labels, se, width_door, empty_in_lbl, occ_lbl):
    """
    The room number that is poorly divided is selected and the parameters are 
    changed so that it is divided well.

    Parameters
    ----------
    pcd : DataFrame
        Is the pointcloud.
    n_room : str
        Room that is poorly segmented.
    nt : float
        Minimum value to be considered a candidate to floor and celling via 
        normal estimation.
    ransac_th : float
        Maximum distance from point to model in order to be considered as inlier.
    eps : float
        Size to apply DBSCAN.
    pt : int
        Minimum points for DBSCAN.
    voxel_size : float
        Size of the voxel grid to filter the cloud.
    alpha : float
        Alpha value.
    zmax : int
        Maximum heigth of polygon filter.
    zmin : int
        Minimum heigth of polygon filter.
    dist : float
        Maximum distance a point can have to an estimated plane to be considered an inlier.
    vox_labels : DataFrame
        Contains information about the occupancy of each voxel in 3D space.
    se : array_like
        Structuring element used for the erosion. Non-zero elements are 
        considered True. If no structuring element is provided, an element
        is generated with a square connectivity equal to one.
    width_door : float
        Size to build a structure.
    empty_in_lbl : int
        Empty room.
    occ_lbl : int
        Occupied room.

    Returns
    -------
    class_selected_rows : DataFrame
        Are the rooms separated correctly.

    """
    # Select the room that has a bad division
    selected_rows = pcd[pcd['id_room'] == n_room]
    
    # Resets the indexes
    selected_rows = selected_rows.reset_index(drop=True)
    
    # Remove column 'id_room'
    selected_rows = selected_rows.drop(columns=['id_room'])
        
    
    pcd_columns = selected_rows.columns.tolist()
   
    # Voxelización
    pcd_down_o3d, vox_idx = sb.voxel_downsample(
        array_to_o3d.array_to_o3d_point_cloud(selected_rows.loc[:, 'x':'z'].values),
        voxel_size, order_by_index=(True))
    
    # Clean exteriors
    pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(pcd_down_o3d, 
                                                                epsilon = eps,
                                                                points_min = pt, 
                                                                return_index=True)
    # o3d.visualization.draw_geometries([pcd_down_clean])   
    

    # Extract floor, celling and obstacles
    floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil = efc.extract_floor_ceiling(pcd_down_clean, nt,
                                                                                        search_param=o3d.geometry.KDTreeSearchParamKNN(40), 
                                                                                        ransac_th=ransac_th, return_index=True, timestamp=None, 
                                                                                        visualize=False)
    # o3d.visualization.draw_geometries([floor, obstacles, ceiling], window_name="Cleaned point cloud after outliers filtering by DBSCAN")
    

    # Remove non continuous ceiling
    boundary_pol, floor_plane, ceiling_plane, selected_rows, clean_idx = rop.remove_no_continuous(selected_rows, floor, obstacles, ceiling, idx_floor, idx_obs,
                                                                                       idx_ceil, eps, pt, alpha, voxel_size, zmax, zmin, 
                                                                                       dist, vox_idx, idx_clean, visualize = False, save = False)    
   
    
    # Clean point cloud
    # xyz coordinates of cleaned point cloud
    pcd_clean_arr = selected_rows.loc[clean_idx, 'x': 'z'].values

    # Create VoxelGrid filtered point cloud    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        array_to_o3d.array_to_o3d_point_cloud(pcd_clean_arr), voxel_size)


    # Classification of voxel as empty or occupied
    [i_min, j_min, k_min] = voxel_grid.get_voxel(voxel_grid.get_min_bound())
    [i_max, j_max, k_max] = voxel_grid.get_voxel(voxel_grid.get_max_bound())

    # Generate indexes from all voxels (occupied(10) and empty(0)) 
    rng_lst = [k_max-k_min, j_max-j_min, i_max-i_min]

    vxl_idx_arr = np.asarray([item for item in itertools.product(*[range(x+1) for x in rng_lst])])
    vxl_idx_arr[:, [0,2]] = vxl_idx_arr[:, [2,0]]

    # Save in class attribute
    vox_labels['i'] = vxl_idx_arr[:, 0]
    vox_labels['j'] = vxl_idx_arr[:, 1]
    vox_labels['k'] = vxl_idx_arr[:, 2]           

    # Compute centroids
    vxl_cent = vxl_idx_arr * voxel_size + voxel_grid.get_min_bound()


    xyz_min = pcd_clean_arr.min(axis=0)
    ijk_occ_vxl = np.rint((pcd_clean_arr - xyz_min)/voxel_size)
    idx_occ_vxl = np.unique(np.dot(ijk_occ_vxl, np.array([1, rng_lst[2]+1, 
                                                          (rng_lst[2]+1)*(rng_lst[1]+1)]).T).astype(int))

    vox_lbl = np.zeros(vxl_idx_arr.shape[0], dtype=int)
    vox_lbl[idx_occ_vxl] = 10


    idx_empty = np.in1d(range(len(vox_lbl)), idx_occ_vxl)

    empty_cent = vxl_cent[~idx_empty, :]

    # o3d empty cloud
    empty_cloud = array_to_o3d.array_to_o3d_point_cloud(empty_cent)

    # if visualize:
    #     empty_cloud.paint_uniform_color([.0, .3, .9])
    #     o3d.visualization.draw_geometries([empty_cloud], 
    #                                       window_name="Occupied voxels: {} Empty voxels: {}".format(len(idx_occ_vxl), 
    #                                                                                                 len(empty_cloud.points)))
        
    # Classify empty voxels as indoor (1) or outside (0)
    filt_empty_cloud, idx_filt = bounding_polygon_filter.bounding_polygon_filter(empty_cloud, 
                                                                   boundary_pol, 
                                                                   zmax, zmin, 
                                                                   eliminate='outside')
    in_empty_pcd = empty_cloud.select_by_index(idx_filt)
    out_empty_pcd = empty_cloud.select_by_index(idx_filt, invert=True)

    # if visualize:
    #     in_empty_pcd.paint_uniform_color([.0, 0.9, .3])
    #     out_empty_pcd.paint_uniform_color([.0, 1., .5])
    #     o3d.visualization.draw_geometries([in_empty_pcd], window_name="Indoor empty space after boundary filter")

    # o3d.io.write_point_cloud(save + "\\Indoor empty space after boundary filter.xyz", in_empty_pcd)

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
    vox_lbl[in_idx] = 1

    # Save in class attribute
    vox_labels['occ'] = vox_lbl

    # o3d.visualization.draw_geometries([in_filt_pcd2], window_name="indoor empty space after distance point-to-plane filter")
        
    # if save:
    #     o3d.io.write_point_cloud(save + '\\indoor_empty_space.xyz', in_filt_pcd2)

    vxl_idx_arr = np.hstack((vxl_idx_arr, vox_lbl[:, np.newaxis]))
    labelled_vox_df = pd.DataFrame(vxl_idx_arr, columns=['i', 'j', 'k','scalar'])

    
    # 3D morphological erosion
    image_erode, se_erosion = erosion.morphological_erosion(labelled_vox_df, se, 
                                                width_door, empty_in_lbl, 
                                                voxel_size)
    

    erode_image = image_erode
    erode_image_df = labelled_vox_df.copy()
    erode_image_df.loc[:, 'scalar'] = 0
    erode_image_df.loc[:, 'scalar'] = erode_image.flatten()
    erode_pcd = array_to_o3d.array_to_o3d_point_cloud(erode_image_df.loc[erode_image_df.scalar==1, 'i':'k'].values)
    # erode_pcd.paint_uniform_color([.7, .7, .7])
    # o3d.visualization.draw_geometries([erode_pcd], window_name='Erode empty space')    
    # if save:
    #         erode_image_df.to_csv(save + '\\erode_im.csv', sep = ",", index=False) #añadí sep = ","
    # o3d.io.write_point_cloud(save + "\\erosion.xyz", erode_pcd) 
        
    
    # Individualisation
    labels_out = individualisation.room_individualisation(image_erode, voxel_size)

    labelled_erode_im_df = labelled_vox_df.copy()
    labelled_erode_im_df.loc[:, 'scalar'] = 0
    labelled_erode_im_df.loc[:,'scalar'] = labels_out.flatten()

    o3d_lbl_erode = array_to_o3d.array_to_o3d_point_cloud(labelled_erode_im_df.loc[labelled_erode_im_df.scalar>0,'i':'k'].values)
    erode_lbls = labelled_erode_im_df.loc[labelled_erode_im_df.scalar>0 ,'scalar'].values
    max_label = erode_lbls.max()
    colors = cm.get_cmap("tab20")(erode_lbls / (max_label if max_label > 0 else 1))
    colors[erode_lbls < 0] = 0
    o3d_lbl_erode.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_lbl_erode], window_name="Individualised spaces") 
    #     if save:
    #         labelled_erode_im_df.to_csv(save + '\\erode_labeled_im.csv', index=False)
    # o3d.io.write_point_cloud(save + "\\individualisation.xyz", o3d_lbl_erode)

    
    # 3D Morphological dilation
    dilated_image = dilation.morphological_dilation(image_erode, labels_out, se_erosion)

    # Save in class attribute
    vox_labels['empty_room'] = dilated_image.flatten()


    dilated_labelled_df = erode_image_df.copy()
    dilated_labelled_df.loc[:, 'scalar'] = 0
    dilated_labelled_df.loc[:, 'scalar'] = dilated_image.flatten()
        
    o3d_lbl_dilated = array_to_o3d.array_to_o3d_point_cloud(dilated_labelled_df.loc[dilated_labelled_df.scalar>0,'i':'k'].values)
    dilated_lbls = dilated_labelled_df.loc[dilated_labelled_df.scalar>0 ,'scalar'].values
    max_label = dilated_lbls.max()
    colors = cm.get_cmap("tab20")(dilated_lbls / (max_label if max_label > 0 else 1))
    colors[dilated_lbls < 0] = 0
    o3d_lbl_dilated.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_lbl_dilated], window_name='Dilated spaces')
            
    # if save:
    #     dilated_labelled_df.to_csv(save + '\\dilated_labelled_im.csv', index=False)
    # o3d.io.write_point_cloud(save + "\\ditation.xyz", o3d_lbl_dilated)
    

    # Occupied voxels classification
    room_lab_arr, segmented_3d_image = ovc.occupied_voxels_classification(labelled_vox_df, vxl_idx_arr,
                                                                                dilated_image, rng_lst, occ_lbl)

    vox_labels['occ_room'] = segmented_3d_image.flatten()


    classified_vox_df = dilated_labelled_df.copy()  
    classified_vox_df.loc[:, 'scalar'] = 0
    classified_vox_df.loc[:,'scalar'] = segmented_3d_image.flatten()
        
    o3d_class_vxl = o3d.geometry.PointCloud()
    o3d_class_vxl.points = o3d.utility.Vector3dVector(classified_vox_df.loc[classified_vox_df.scalar>0,'i':'k'].values)
    class_lbls = classified_vox_df.loc[classified_vox_df.scalar>0 ,'scalar'].values
    max_label = class_lbls.max()
    print(f"Point cloud has {max_label + 1} rooms")
    colors = cm.get_cmap("tab20")(class_lbls / (max_label if max_label > 0 else 1))
    colors[class_lbls < 0] = 0
    o3d_class_vxl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_class_vxl], window_name="Classified occupied voxels")       
    # if save:
    #     classified_vox_df.to_csv(save + '\\classified_vox_im.csv', index=False)             
    # o3d.io.write_point_cloud(save + "\\classification.xyz", o3d_class_vxl)


    # Point cloud classification
    class_pts_arr, idx_in_pcd = pcc.point_cloud_classification(selected_rows, pcd_clean_arr, 
                                                                     room_lab_arr, voxel_grid, 
                                                                     voxel_size)
    pcd_columns.append('id_room')
    # Classified input point cloud
    class_selected_rows = pd.DataFrame(np.hstack((selected_rows.loc[idx_in_pcd, 'x':'walls'].values,
                                           class_pts_arr[:, np.newaxis])),
                                columns=pcd_columns)
  


    return class_selected_rows


































