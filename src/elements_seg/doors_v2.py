# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree, distance
import matplotlib as plt

from src.utils import array_to_o3d
from src.utils import grid_dataframe_to_3Dimage
from src.utils import compute_normals
from src.utils import image3D_to_grid_dataframe

def door_detection(vox_labels, voxel_size, voxel_grid, empty_in_lbl, 
                   occ_lbl, pts_door,eps_door, min_pt_door, visualize, save):
    """
    Doors determine the adjacency relationships of the rooms in a building.
    Aware of this, a door detection process has been developed from the
    previous morphological room segmentation method in order to generate 
    the highest topological level of the proposed hierarchical navigation.   

    Parameters
    ----------
    vox_labels : DataFrame
        Contains information about the occupancy of each voxel in 3D space.
    voxel_size : float
        Voxel size
        Is measured in meters.
    voxel_grid : PointCloud object of Open3D
        Structure that organizes the points in a point cloud into 
        three-dimensional cells called "voxels."
    empty_in_lbl : int
        Empty room.
    occ_lbl : int
        Occupied room.
    pts_door : float
        Distance threshold that you define to classify a point as a "door". 
        Points that are at a distance less than pts_door will be considered 
        "non-doors", and points that are at a distance equal to or greater
        than pts_door will be considered "doors".   
    eps_door : float
        Maximum distance between two samples for one to be considered a neighbor
        of the other. Basically, it determines how close two points must be to
        be considered part of the same cluster. If eps_door is very small,
        the algorithm may not find many clusters and group many points into 
        the same cluster. If it is too large, the algorithm may not find 
        distinct clusters.
    min_pt_door : int
        Minimum number of points that must be inside eps_door for a point to be 
        considered a "center point" of the cluster. In other words, it is the
        minimum number of points that must be close to a point for it to be
        considered part of a cluster. If the number of points within eps_door 
        around a point is less than min_pt_door, that point will be considered 
        a "border point" and will not be included in the cluster.
    visualize : str
        True : show pictures
        False : not show pictures.
    save : str
        Save the new point cloud.

    Returns
    -------
    bbx_door_lst : list of numpy arrays
        Grid bounding box coordinates to xyz coordinates.
    adj_room_lst : list of numpy arrays
        Rooms adjacent to door.

    """
    
    door_min = 1/voxel_size
    
    occ_vox_df =  vox_labels.loc[:, ('i', 'j', 'k', 'occ')]
    labelled_vox_df = vox_labels.loc[:, ('i', 'j', 'k', 'empty_room')]
    classified_vox_df = vox_labels.loc[:, ('i', 'j', 'k', 'occ_room')]
    
    # Rename label name column to 'scalar'
    occ_vox_df.columns = ['i', 'j', 'k', 'scalar']
    labelled_vox_df.columns = ['i', 'j', 'k', 'scalar']
    classified_vox_df.columns = ['i', 'j', 'k', 'scalar']
    
    # Generate 3D image from voxelized data 
    image_3d = np.zeros((occ_vox_df.loc[:, 'i':'k'].max().values[::-1]+1), int)

    # DataFrame Indices of indoor empty voxels
    index_indoor_empty_voxels = occ_vox_df[occ_vox_df.scalar==empty_in_lbl].index
    grid_coord_indoor_empty_voxels = occ_vox_df.loc[index_indoor_empty_voxels, 'i':'k']

    image_3d[grid_coord_indoor_empty_voxels.loc[:, 'k'].values,
             grid_coord_indoor_empty_voxels.loc[:, 'j'].values,
             grid_coord_indoor_empty_voxels.loc[:, 'i'].values] = 1
    
    dilated_image = grid_dataframe_to_3Dimage.grid_dataframe_to_3d_image(labelled_vox_df)
    
    # Put occupied voxel to 0
    image_3d[image_3d==occ_lbl] = 0
    
    # Compute non-classified empty voxels (voxels between walls, door space, ...    )
    unrecoverd_voxel = np.logical_xor(image_3d, dilated_image).astype(int)
    unrecovered_voxel_df = labelled_vox_df.copy()
    unrecovered_voxel_df.loc[:, 'scalar'] = 0
    unrecovered_voxel_df.loc[:,'scalar'] = unrecoverd_voxel.flatten()
    
    # Visualization on open3D visualiser
    unrecovered_vxl_o3d = array_to_o3d.array_to_o3d_point_cloud(
    unrecovered_voxel_df.loc[unrecovered_voxel_df.scalar==1, 'i':'k'].values)
    if visualize:
        unrecovered_vxl_o3d.paint_uniform_color([1., 0. , 0.])
        o3d.visualization.draw_geometries([unrecovered_vxl_o3d], window_name='Unrecovered Empty Voxels')    
    if save:
        o3d.io.write_point_cloud(save + '\\unrecovered_empty_voxels.xyz', unrecovered_vxl_o3d)
    
    # Filter by normals (according to z direction)
    unrecovered_normals = compute_normals.compute_normals(
        unrecovered_voxel_df.loc[unrecovered_voxel_df.scalar==1, 'i':'k'].values,4)
    idx_nor_discard = np.where(unrecovered_normals[:,2]>0.10)[0]
    idx_nor_select =  np.where(unrecovered_normals[:,2]<=0.10)[0]
    
    
    vertical_pcd = unrecovered_vxl_o3d.select_by_index(idx_nor_select)
    no_vertical_pcd = unrecovered_vxl_o3d.select_by_index(idx_nor_select, invert=True)
        
    
    if visualize:        
        o3d.visualization.draw_geometries([vertical_pcd], window_name='Vertical pcd')
    
        
    # Discard removed voxels close to occupied voxels 
    occ_tree = cKDTree(classified_vox_df.loc[classified_vox_df.scalar > 0, 'i':'k'].values)
    
    dd, ii = occ_tree.query(np.asarray(vertical_pcd.points), k=1)
    
    
    idx_no_door = np.where(dd < pts_door)[0]
    idx_door = np.where(dd >= pts_door)[0]
    
    cand_door_vxl = vertical_pcd.select_by_index(idx_door)
    disc_vert_vxl = vertical_pcd.select_by_index(idx_door, invert=True)
    
    disc_vert_vxl.paint_uniform_color([1.,0.,0.])
    cand_door_vxl.paint_uniform_color([1.,1.,0.])
    
    if visualize:
        o3d.visualization.draw_geometries([disc_vert_vxl, cand_door_vxl], window_name='Candidate doors')
     
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as ctxtman:
            labels = np.array(cand_door_vxl.cluster_dbscan(eps=eps_door,
                                                           min_points=min_pt_door, 
                                                           print_progress=True))
    
    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    colors = plt.colormaps.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    
    if visualize:
        cand_door_vxl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([cand_door_vxl])    
    if save:            
        cand_coloured_arr = np.hstack((np.asarray(cand_door_vxl.points)[labels>=0], 
                                        np.asarray(cand_door_vxl.colors)[labels>=0]))
        cand_coloured_df = pd.DataFrame(cand_coloured_arr, columns=['x', 'y', 'z', 'R', 'G', 'B'])
        cand_coloured_df.to_csv(save + '\\coloured_cand.csv', sep=' ', index=False)
    
    disc_cand_lst = []
    
    disc_idx = np.array([],int)
    lbl_room_idx = np.array([], int)
    idx_doors_lst = []
    outside_pcd = o3d.geometry.PointCloud()
    outside_pcd.points = o3d.utility.Vector3dVector(occ_vox_df.loc[occ_vox_df.scalar==0, 'i':'k'].values)
    
    bb_door_lst = []
    
    # Rooms adjacent to door
    adj_room_lst = []
    
    # 3D dilated_image ->  DataFrame -> o3d pointcloud
    dilated_labelled_df = image3D_to_grid_dataframe.image_3d_to_grid_dataframe(dilated_image)        
    o3d_lbl_dilated = array_to_o3d.array_to_o3d_point_cloud(
        dilated_labelled_df.loc[dilated_labelled_df.scalar>0,'i':'k'].values)
    
    # Classified vox dataframe -> o3d point cloud
    o3d_class_vxl = array_to_o3d.array_to_o3d_point_cloud(
        classified_vox_df.loc[classified_vox_df.scalar>0,'i':'k'].values)
    
    # Labels of dilated voxels
    dilated_lbls = dilated_labelled_df.loc[dilated_labelled_df.scalar>0 ,'scalar'].values
    
    for lbl_cand in range(max_label+1):    
        cand_pcd = cand_door_vxl.select_by_index(np.where(labels==lbl_cand)[0])
                
        # Aligned bounding box fitted to candidate cluster
        cand_aabb = cand_pcd.get_axis_aligned_bounding_box()
        cand_aabb_pts = np.asarray(cand_aabb.get_box_points())
        
        width = np.max(cand_aabb_pts[:, 1]) - np.min(cand_aabb_pts[:, 1])
        if width < 7:
            
            # Minimum axis variation
            np.min(np.asarray(cand_pcd.points), axis=0)
            min_axis = np.argmin(cand_aabb.get_max_bound() - cand_aabb.get_min_bound())

            # Assume door height(min_axis=2) is longer than the other dimensions
            if min_axis != 2:
                cand_aabb_pts[cand_aabb_pts[:,min_axis]<cand_aabb.get_center()[min_axis], min_axis] = cand_aabb.get_min_bound()[min_axis] - 2
                cand_aabb_pts[cand_aabb_pts[:,min_axis]>cand_aabb.get_center()[min_axis], min_axis] = cand_aabb.get_max_bound()[min_axis] + 2
                
                # Extrusion bounding box along minimal axis variation
                extr_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(cand_aabb_pts, axis=0), max_bound=np.max(cand_aabb_pts, axis=0))
                
                # Check adjacency to dilated rooms 
                idx_adj = extr_aabb.get_point_indices_within_bounding_box(o3d_lbl_dilated.points)
                idx_occ = extr_aabb.get_point_indices_within_bounding_box(o3d_class_vxl.points)
                idx_out = extr_aabb.get_point_indices_within_bounding_box(outside_pcd.points)
                adj_rooms = dilated_lbls[idx_adj]
                
                uni_adj, count_adj = np.unique(adj_rooms, return_counts=True) 
                       
                if (len(idx_occ) > len(cand_pcd.points)) or len(uni_adj) != 2: 
                    disc_cand_lst.append(lbl_cand)
                    disc_idx = np.append(disc_idx, np.where(labels==lbl_cand)[0])
                    
                elif ((count_adj[0]/len(idx_adj)) < 0.01) or ((count_adj[1]/len(idx_adj)) < 0.01): # antes estaba 0.1
                    disc_cand_lst.append(lbl_cand)
                    disc_idx = np.append(disc_idx, np.where(labels==lbl_cand)[0])
                    
                elif (cand_aabb_pts[len(cand_aabb_pts) - 1][0] - cand_aabb_pts[0][0] > 50.0):
                    disc_cand_lst.append(lbl_cand)
                    disc_idx = np.append(disc_idx, np.where(labels==lbl_cand)[0])                
                                   
                else:
                    print("detected door: {}".format(lbl_cand))
                    if len(bb_door_lst):
                        # centroid of previous detected doors on plane XY
                        doors_cent = np.asarray([door_bb.get_center()[:2] for door_bb in bb_door_lst])
                        # centroid of candidate door on plane XY
                        cand_cent = extr_aabb.get_center()[:2]
                        
                        # Distance between centroids
                        dist_cent = distance.cdist(cand_cent[:, np.newaxis].T, doors_cent)
                        if np.any(dist_cent < door_min):
                            print("repetead door")
                            door_idx = int(np.where(dist_cent < door_min)[1])
                            bb_door = bb_door_lst[door_idx]
                            
                            # generate bounding box
                            bb_pts = np.vstack((np.asarray(extr_aabb.get_box_points()),
                                                np.asarray(bb_door.get_box_points())))
                            xyz_bb_min = bb_pts.min(axis=0)
                            xyz_bb_max = bb_pts.max(axis=0)
                            
                            new_bb_pts = np.array(np.meshgrid([xyz_bb_min[0], xyz_bb_max[0]], 
                                                              [xyz_bb_min[1], xyz_bb_max[1]], 
                                                              [xyz_bb_min[2], xyz_bb_max[2]])).T.reshape(-1,3)
                            
                            new_aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                                o3d.utility.Vector3dVector(new_bb_pts))
                            
                            # update door's bounding box list
                            bb_door_lst[door_idx] = new_aabb
                            idx_doors_lst[door_idx] = np.concatenate((idx_doors_lst[door_idx], np.where(labels==lbl_cand)[0]))
                            
                            continue
                            
                        
                    bb_door_lst.append(extr_aabb)
                    idx_doors_lst.append(np.where(labels==lbl_cand)[0])
                    adj_room_lst.append(uni_adj)
            else:
                disc_cand_lst.append(lbl_cand)
                disc_idx = np.append(disc_idx, np.where(labels==lbl_cand)[0])
            
        else:
            continue
        
    # If points close to door clusters belong to two different room, door cluster is considered a door
    
    if visualize:
        doors_vxl = cand_door_vxl.select_by_index(np.hstack(idx_doors_lst))
        doors_vxl.paint_uniform_color([1.,0.,0.])
        o3d.visualization.draw_geometries([o3d_lbl_dilated.select_by_index(
            np.where(dilated_lbls>1)[0]), doors_vxl, *bb_door_lst])
        o3d.visualization.draw_geometries([ doors_vxl, *bb_door_lst])   
    
    if save: 
        o3d.io.write_point_cloud(save + "\\rooms_without_doors.xyz", 
                                 o3d_lbl_dilated.select_by_index(np.where(dilated_lbls>1)[0]))
        o3d.io.write_point_cloud(save + "\\doors.xyz", doors_vxl)
    
    
    bbx_door_lst = [ np.asarray(bbx_door.get_box_points()) for bbx_door in bb_door_lst]
    
    # Grid bounding box coordinates to xyz coordinates
    bmin = voxel_grid.get_min_bound()
    
    bbx_door_pts_lst = [bmin + vxl_idx * voxel_size + (voxel_size/2) for vxl_idx_arr in bbx_door_lst for vxl_idx in vxl_idx_arr]

    
    if visualize:
        o3d_door_pts = array_to_o3d.array_to_o3d_point_cloud(np.asarray(bbx_door_pts_lst))
            
    if save:
        o3d.io.write_point_cloud(save + "\\door_pts.xyz", o3d_door_pts)
    
    return bbx_door_lst, bbx_door_pts_lst, adj_room_lst