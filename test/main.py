# -*- coding: utf-8 -*-

#  Paths

path = r"C:/Users/HP/OneDrive - Universidade de Vigo/RecycleBIM_UVigo_shared-folder/0_GitHub/Code/data/" # Case study folder
folder_cloud = r"/csBilbao"
folder_results = r"C:/Users/HP/OneDrive - Universidade de Vigo/RecycleBIM_UVigo_shared-folder/0_GitHub/Code/results" # Results


import sys
from pathlib import Path


# Añadir ruta del proyecto (src/)
try:
    project_path = str(Path(__file__).parent.parent)
except NameError:
    project_path = str(Path.cwd().parent)
    
sys.path.append(project_path)

from src.storey_seg import segmentation_trajectory

#%%
# Packages (Python)
import laspy as lp
import pandas as pd
import numpy as np
import time
import os
import json

from scipy.signal import find_peaks
import open3d as o3d

import alphashape as alph
import itertools
import matplotlib
matplotlib.use('Agg')  # Establece backend sin GUI
import matplotlib as plt
from scipy.spatial import cKDTree, distance
import cc3d
from scipy.stats import mode
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point

# Packages 
from parameters import parameters_Bilbao as pm

from src.storey_seg import segmentation_trajectory

from src.utils import array_to_o3d
from src.utils import remove_no_continuous as rop
from src.utils import bounding_polygon_filter
from src.utils import convert_numpy_to_list

from src.preprocessing import subsampling as sb
from src.preprocessing import clean_exteriors

from src.elements_seg import stairs
from src.elements_seg import extract_floor_ceiling as efc

from src.morph_seg import class_empty_occupied as ceo
from src.morph_seg import erosion
from src.morph_seg import individualisation
from src.morph_seg import dilation
from src.morph_seg import occupied_voxels_classification as ovc
from src.morph_seg import point_cloud_classification as pcc

from src.morph_seg import mix_room
from src.morph_seg import pick_in_pc

from src.elements_seg import walls_index
from src.info_elements import intersection_points_index
from src.info_elements import points_global_index
from src.info_elements import mod_lines
from src.info_elements import points_room_order

from src.info_elements import adjacency_walls
from src.info_elements import adjacency_intra
from src.info_elements import graphs
from src.info_elements import infoBIM
from src.info_elements import overlaps_walls
from src.elements_seg import windows
from src.info_elements import infoBIM_doors_windows
from src.elements_seg import doors_with_trajectory
from src.info_elements import graph
from src.improve_ifc.stairs_to_IFC  import stairs_to_IFC

#%% Folders that are created to save the results

# Folder that contains results
try:
    os.mkdir(folder_results)
    print("Folder has already been created.")
    print("The folder already exists.")
except Exception as e:
    print("Error creatring folder:", e)

# Folder with the results from Bilbao
path_results_csBilbao = folder_results + folder_cloud
try:
    os.mkdir(path_results_csBilbao)
    print("Folder has already been created.")
except FileExistsError:
    print("The folder already exists.")
except Exception as e:
    print("Error creatring folder:", e)

# Folder with the results of the segmentation by storeys
path_Storeys = path_results_csBilbao + "\\Storeys"
try:
    os.mkdir(path_Storeys)
    print("Folder has already been created.")
except FileExistsError:
    print("The folder already exists.")
except Exception as e:
    print("Error creatring folder:", e)  

#%% Storey Segmentation (with trajectory)
# Paths
path_building = path + folder_cloud + "\\Bilbao_all_AutoCleaned.txt"
path_trajectory = path + folder_cloud + "\\Trajectory.txt" 

# It identifies where to cut
intervals = segmentation_trajectory.seg_traj(path_building, path_trajectory, 
                                              pm.nt, pm.ransac_th_2, pm.init_n, 
                                              pm.iterat, pm.h_2, pm.g_res, 
                                              pm.threshold_size)

# Save all DataFrames to CSV files
dfs =[]
for s, (key, df) in enumerate(intervals.items()):
    df['id_storey'] = s
    df.to_csv(f'{path_Storeys}\\Storey_{s}.csv', index=False)
    dfs.append(df)
    
# Concatenate all DataFrames in the list into one
final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(f'{path_Storeys}\\Storeys.csv', index=False)


#%% The stairs are removed 
folder_clean_storey = folder_results + folder_cloud + "\\Storeys"

stories = []
for number_storey in range(0, 1):  
    file = f"Storey_{number_storey}.csv"  # Archive name
    storey = os.path.join(folder_clean_storey, file)
    
    pcd_in =  pd.read_csv(storey, sep=',', usecols=[0,1,2,3] )
    stories.append(pcd_in['id_storey'].max())
last_storey = max(stories)


for number_storey in range(0, 1):  
    file = f"Storey_{number_storey}.csv"  # Archive name
    storey = os.path.join(folder_clean_storey, file)
    
    pcd_in =  pd.read_csv(storey, sep=',', usecols=[0,1,2,3] 
                          )
    # Folder with the results of the analyzed storey
    path_storey = (folder_results + folder_cloud +
                   f"\\Storey_{number_storey}_Last")
    try:
        os.mkdir(path_storey)
        print("Folder has already been created.")
    except FileExistsError:
        print("The folder already exists.")
    except Exception as e:
        print("Error creatring folder:", e)
    
    # Folder with the results of the storey elements
    path_elements = path_storey + "\\elements"
    try:
        os.mkdir(path_elements)
        print("Folder has already been created.")
    except FileExistsError:
        print("The folder already exists.")
    except Exception as e:
        print("Error creatring folder:", e)
    
    
    boundary_pol = stairs.extract_boundary_ceiling_or_floor(pcd_in,
                                                            last_storey)    
    
    st_without_stairs, stairs = stairs.remove_stairs(pcd_in, boundary_pol, pm.s_p,
                                                     pm.nt, pm.eps_s, pm.pts_s,
                                                     pm.ransac_s, pm.init_n,
                                                     pm.iterat_s, path_elements, 
                                                     visualize=False, save=True
                                                     )
    
    pcd_file = folder_results + folder_cloud  + f"/Storey_{number_storey}_Last/elements/stairs.txt"
    point_cloud_building = pd.read_csv(folder_results + folder_cloud + f"/Storeys/Storey_{number_storey}.csv", sep=',')[['x','y','z']]

    #Stairs
    (length,
     width,
     p_stairs,
     p_stairs_o) = stairs_to_IFC(pcd_file, point_cloud_building, pm.nt, pm.threshold, pm.init_n,
                                 pm.iterat, pm.max_dist_stairs, pm.eps_stairs, 
                                 pm.pts_stairs, pm.vertical_step,
                                 pm.horizontal_step, n_storey=number_storey)
                                 
    stairs_to_IFC.file_stairs_IFC(length, width, number_storey,
                    p_stairs, p_stairs_o)
    
    
    #%% Room Segmentation
    start = time.time()
    pcd_in = st_without_stairs
    
    n_storey = int(pcd_in['id_storey'].unique()[0])
    last_column = pcd_in.columns[-1]
    
    # Check if the last column has the attribute 'id_storey'
    if 'id_storey' in last_column:
        print("The last column has the attribute 'id_storey'")
    else:
        print("The last column has not the attribute'id_storey'")
        pcd_in['id_storey'] = int(n_storey)
        print(pcd_in)
    
    pcd_in['id_storey'] = pcd_in['id_storey'].astype(int)
    pcd_in_columns = pcd_in.columns.tolist()
    
    # Voxelización
    pcd_down_o3d, vox_idx = sb.voxel_downsample(
        array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[:, 'x':'z'].values), 
        pm.voxel_size, order_by_index=(True))
    
    # Clean exteriors
    pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(
                                                        pcd_down_o3d,
                                                        epsilon = pm.eps_1, 
                                                        points_min = pm.pt, 
                                                        return_index=True)
    
    # Extract floor, celling and obstacles
    [floor, obstacles, ceiling, idx_floor, idx_obs, 
     idx_ceil] = efc.extract_floor_ceiling(pcd_down_clean, pm.nt, 
                                           search_param=pm.s_p,
                                           ransac_th=pm.ransac_th,
                                           iterat=pm.iterat,
                                           return_index=True, 
                                           timestamp=None, 
                                           visualize=False)
    
    # Remove non continuos celling
    [boundary_pol, floor_plane, ceiling_plane, pcd_in,
     clean_idx] = rop.remove_no_continuous(pcd_in, floor, obstacles, ceiling, 
                                           idx_floor, idx_obs, idx_ceil,
                                           pm.eps_2,  pm.pt, pm.alpha,
                                           pm.voxel_size, pm.zmax,
                                           pm.zmin, pm.dist, pm.init_n, 
                                           vox_idx, idx_clean,
                                           visualize = False, save = False)    
    
    pcd_to_windows = pcd_in[['x','y','z']]
    se = pm.str_el(n_storey)
    
    [labelled_vox_df, vxl_idx_arr, rng_lst, pcd_clean_arr, voxel_grid
     ] = ceo.class_empty_occupied(pcd_in, clean_idx, boundary_pol, floor_plane, 
                                  ceiling_plane)
    
    # 3D morphological erosion
    image_erode, se_erosion = erosion.morphological_erosion(labelled_vox_df,
                                                            se, 
                                                            pm.width_door, 
                                                            pm.empty_in_lbl, 
                                                            pm.voxel_size)
    erode_image = image_erode
    erode_image_df = labelled_vox_df.copy()
    erode_image_df.loc[:, 'scalar'] = 0
    erode_image_df.loc[:, 'scalar'] = erode_image.flatten()
    erode_pcd = array_to_o3d.array_to_o3d_point_cloud(
        erode_image_df.loc[erode_image_df.scalar==1, 'i':'k'].values)
    erode_pcd.paint_uniform_color([.7, .7, .7])
    # o3d.visualization.draw_geometries([erode_pcd], 
    # window_name='Erode empty space')  
    
    # Individualisation
    labels_out = individualisation.room_individualisation(image_erode, 
                                                          pm.voxel_size)
    labelled_erode_im_df = labelled_vox_df.copy()
    labelled_erode_im_df.loc[:, 'scalar'] = 0
    labelled_erode_im_df.loc[:,'scalar'] = labels_out.flatten()
    o3d_lbl_erode = array_to_o3d.array_to_o3d_point_cloud(
        labelled_erode_im_df.loc[labelled_erode_im_df.scalar>0,'i':'k'].values)
    erode_lbls = labelled_erode_im_df.loc[
        labelled_erode_im_df.scalar>0 ,'scalar'].values
    max_label = erode_lbls.max()
    colors = plt.colormaps.get_cmap("tab20")(
        erode_lbls / (max_label if max_label > 0 else 1))
    colors[erode_lbls < 0] = 0
    o3d_lbl_erode.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_lbl_erode], 
    # window_name="Individualised spaces")
    
    # 3D Morphological dilation
    dilated_image = dilation.morphological_dilation(image_erode, labels_out, 
                                                    se_erosion)
    # Save in class attribute
    pm.vox_labels['empty_room'] = dilated_image.flatten()
    dilated_labelled_df = erode_image_df.copy()
    dilated_labelled_df.loc[:, 'scalar'] = 0
    dilated_labelled_df.loc[:, 'scalar'] = dilated_image.flatten()
    o3d_lbl_dilated = array_to_o3d.array_to_o3d_point_cloud(
        dilated_labelled_df.loc[dilated_labelled_df.scalar>0,'i':'k'].values)
    dilated_lbls = dilated_labelled_df.loc[
        dilated_labelled_df.scalar>0 ,'scalar'].values
    max_label = dilated_lbls.max()
    colors = plt.colormaps.get_cmap("tab20")(
        dilated_lbls / (max_label if max_label > 0 else 1))
    colors[dilated_lbls < 0] = 0
    o3d_lbl_dilated.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_lbl_dilated], 
    # window_name='Dilated spaces')   
    
    # Occupied voxels classification
    room_lab_arr, segmented_3d_image = ovc.occupied_voxels_classification(
        labelled_vox_df, vxl_idx_arr, dilated_image, rng_lst, pm.occ_lbl)
    pm.vox_labels['occ_room'] = segmented_3d_image.flatten()
    classified_vox_df = dilated_labelled_df.copy()  
    classified_vox_df.loc[:, 'scalar'] = 0
    classified_vox_df.loc[:,'scalar'] = segmented_3d_image.flatten()
    o3d_class_vxl = o3d.geometry.PointCloud()
    o3d_class_vxl.points = o3d.utility.Vector3dVector(
        classified_vox_df.loc[classified_vox_df.scalar>0,'i':'k'].values)
    class_lbls = classified_vox_df.loc[classified_vox_df.scalar>0 ,
                                       'scalar'].values
    max_label = class_lbls.max()
    print(f"Point cloud has {max_label + 1} rooms")
    colors = plt.colormaps.get_cmap("tab20")(
        class_lbls / (max_label if max_label > 0 else 1))
    colors[class_lbls < 0] = 0
    o3d_class_vxl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([o3d_class_vxl], 
    # window_name="Classified occupied voxels")
    
    # Point cloud classification
    class_pts_arr, idx_in_pcd = pcc.point_cloud_classification(
        pcd_in, pcd_clean_arr, room_lab_arr, voxel_grid, pm.voxel_size)
    pcd_in_columns.append('id_room')
    
    # Classified input point cloud
    class_pcd_df = pd.DataFrame(
        np.hstack((pcd_in.loc[idx_in_pcd, 'x':'id_storey'].values, 
                   class_pts_arr[:, np.newaxis])), columns=pcd_in_columns)
    class_pcd_df['id_room'] = class_pcd_df['id_room'].astype(int)
    
    # class_pcd_df.to_csv(path_storey + "\\segmetation_pointcloud.csv",
    # index=False)
    
    end = time.time()
    elapsed = end-start
    print(elapsed)
    
    #%% Is necessary segment rooms again?
    del [class_lbls, class_pts_arr, classified_vox_df, dilated_image, 
         dilated_labelled_df, dilated_lbls, erode_image, erode_image_df, 
         erode_lbls, erode_pcd, idx_in_pcd, image_erode, labelled_erode_im_df,
         labelled_vox_df, labels_out, max_label, pcd_clean_arr, rng_lst, 
         room_lab_arr, se_erosion, segmented_3d_image, vox_idx, voxel_grid, 
         vxl_idx_arr]
    
    while True:
        # Ask the user if they want to perform an action
        answer = input(
            "Is there a room that needs to be modified? ('yes' or 'no'): ")
        # Check the answer
        if answer.lower() == 'yes':
            print("Have to modify the input parameters.")
            pcd = class_pcd_df
            n_room = pick_in_pc.select_room(pcd)
            d = int(input("Size door: "))
            se = np.ones((d,d,d))
            pm.vox_labels = pd.DataFrame() 
            class_selected_rows = mix_room.mix_room(
                pcd, n_room, pm.nt, pm.ransac_th, pm.eps_2, pm.pt, 
                pm.voxel_size,  pm.alpha, pm.zmax, pm.zmin, pm.dist,
                pm.vox_labels, se, pm.empty_in_lbl, pm.occ_lbl, pm.width_door)
            
            # Remove room from dataframe
            pcd_r = pcd[pcd['id_room'] != n_room]
            ex_column = pcd_r.pop('id_room')
            pcd_r['id_room'] = ex_column
    
            m = pcd_r['id_room'].max()
            for i in range(
                    min(class_selected_rows['id_room'].unique().astype(int)), 
                    max(class_selected_rows['id_room'].unique().astype(int))+1):
                class_selected_rows.loc[class_selected_rows['id_room'] == i, 
                                        'id_room'] = m + i
    
            class_pcd_df = pd.concat([pcd_r, class_selected_rows], 
                                     ignore_index=True)
            
            class_pcd_df['id_room'] = class_pcd_df['id_room'].replace(
                class_pcd_df['id_room'].max(), n_room)
    
            # class_pcd_df.to_csv(path_storey + "\\segmetation_pointcloud_" + 
            #                     str(d) + ".csv", index=False)
            
        elif answer.lower() == 'no':
            print("No need to modify input parameters.")
            break
    
    #%%
    # Create a Series with all values
    [floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil
     ] = efc.extract_floor_ceiling(array_to_o3d.array_to_o3d_point_cloud(
         class_pcd_df.loc[:, 'x':'z'].values), pm.nt, search_param=pm.s_p, 
         ransac_th=pm.ransac_th, iterat=pm.iterat, return_index=True,
         timestamp=None, 
         visualize=False)
         
    new_values = pd.Series([0] * len(class_pcd_df.iloc[idx_floor]),
                           index=class_pcd_df.iloc[idx_floor].index, 
                           name='id_element')
    new_values_1 = pd.Series([1] * len(class_pcd_df.iloc[idx_ceil]), 
                             index=class_pcd_df.iloc[idx_ceil].index, 
                             name='id_element')
    new_values_2 =  pd.Series([2] * len(class_pcd_df.iloc[idx_obs]), 
                              index=class_pcd_df.iloc[idx_obs].index, 
                              name='id_element')
    # Add the Series as a new column to the DataFrame
    class_pcd_df['id_element'] = 'NaN'
    class_pcd_df.loc[new_values.index, 'id_element'] = new_values
    class_pcd_df.loc[new_values_1.index, 'id_element'] = new_values_1
    class_pcd_df.loc[new_values_2.index, 'id_element'] = new_values_2
    
    class_pcd_df['id_storey'] = class_pcd_df['id_storey'].astype(int)
    class_pcd_df['id_room'] = class_pcd_df['id_room'].astype(int)
    class_pcd_df.to_csv(path_storey + "\\segmetation_pointcloud_last.csv", 
                        index=False)
    
    #%% Elements
    # Wall detection with RANSAC
    [equation_df_walls_t, df_walls_t, lines_position_t, z_0, walls_df, 
     class_pcd_df] = walls_index.walls(class_pcd_df, pm.ex_v, 
                                       pm.max_distance, pm.threshold,
                                       pm.init_n, pm.iterat, pm.h, pm.dist_pl,
                                       pm.min_angle, pm.max_angle,
                                       pm.dist_centroids_1, 
                                       path_elements, n_storey)
    
    # Wall intersection points
    [points_intersection, room_data_list, walls_adjacency
     ] = intersection_points_index.intersect(class_pcd_df, lines_position_t, 
                                             df_walls_t, pm.b_f, pm.min_p_int,
                                             pm.max_dist_int, pm.min_angle_d, 
                                             pm.max_angle_d, pm.perfect_angle,
                                             pm.flat_angle, pm.ex_v)
    
    # Reparameterization of the walls so that they form right angles 
    lines_position_t = points_global_index.adjust_points(
        class_pcd_df, points_intersection, room_data_list, lines_position_t, 
        df_walls_t, z_0, pm.angle_points_global, pm.ang_1, pm.ang_2)
    
    # Bucle para ejecutar la función y manejar el IndexError
    while True:
        try:
            [lines_position_t, df_walls_t, equation_df_walls_t, walls_df, 
             class_pcd_df] = mod_lines.mod_lines( class_pcd_df,
                                                 lines_position_t, 
                                                 df_walls_t, 
                                                 equation_df_walls_t, 
                                                 walls_df, pm.dist_p)
            break  
        except IndexError:
            continue  
    
    # Modification so that rooms where only 3 walls have been detected have 4.
    [lines_position_t, df_walls_t, equation_df_walls_t, walls_df, class_pcd_df
     ] = mod_lines.minimum_4_walls(lines_position_t, class_pcd_df, df_walls_t, 
                                   equation_df_walls_t, walls_df, pm.um, 
                                   pm.um_d)
    
    # Intersection points
    [points_intersection, room_data_list, walls_adjacency
     ] = intersection_points_index.intersect(class_pcd_df, lines_position_t, 
                                             df_walls_t, pm.b_f, pm.min_p_int,
                                             pm.max_dist_int, pm.min_angle_d, 
                                             pm.max_angle_d, pm.perfect_angle,
                                             pm.flat_angle, pm.ex_v)
    
    # If there is any empty list that does not take it into account
    empty_lista = []
    for ind_i, i in enumerate(points_intersection):
        if len(i) == 0:
            empty_lista.append(ind_i)    
    
    for index in sorted(empty_lista, reverse=True):
        points_intersection.pop(index)
    
    # Points of intersection of each room are ordered
    points_f = points_room_order.points_room_order(points_intersection, 
                                                   path_elements)
    
    # Iterate over positions in ascending order
    for index in sorted(empty_lista):
        points_f.insert(index, [])
    
    points_tuples = [[tuple(point) for point in sublist] 
                     for sublist in points_f]
    
    # Save information in 'info_data'
    with open(path_elements + "\\info_data.json", "w") as json_file:
        json.dump(room_data_list, json_file, indent=4)
    
    mod_points_intersection = intersection_points_index.only_intersection_points(
        points_tuples, z_0)
    
    # Floor intersection points and ceiling intersection points
    [list_points_ceiling, list_points_floor
     ] = intersection_points_index.floor_ceiling_plane_json(class_pcd_df, 
                                                            mod_points_intersection, 
                                                            pm.min_ratio, 
                                                            pm.threshold,
                                                            pm.iterat)
    
    # Save information in json in 'information'
    information_json = points_room_order.ordered_walls(class_pcd_df, pm.ex_v, 
                                                       lines_position_t, 
                                                       points_f, df_walls_t,
                                                       path_elements, 
                                                       list_points_ceiling,
                                                       list_points_floor)
    
    # Convert all NumPy arrays to lists
    converted_data = convert_numpy_to_list.convert_numpy_to_list(
        information_json)
    
    # Save to JSON file
    with open(path_elements + '\\information.json', 'w') as f:
        json.dump(converted_data, f, indent=4)
        
    # Save information in 'info_data'
    with open(path_elements + "\\info_data.json", "r") as file:
        room_json = json.load(file)
        
    for room, points_floor, points_ceiling in zip(room_json, list_points_floor, 
                                                  list_points_ceiling):
        room["floor_projection"] = [np.round(
            point, 2).tolist() for point in points_floor]
        room["ceiling_projection"] = [np.round(
            point, 2).tolist() for point in points_ceiling]
    
    with open(path_elements + "\\info_data.json", "w") as file:
        json.dump(room_json, file, indent=4)
        
    empty_lista = []
    for ind_i, i in enumerate(list_points_floor):
        if len(i) == 0:
            empty_lista.append(ind_i)    
    
    for index in sorted(empty_lista, reverse=True):
        list_points_floor.pop(index)
        
    empty_lista = []
    for ind_i, i in enumerate(list_points_ceiling):
        if len(i) == 0:
            empty_lista.append(ind_i)    
    
    for index in sorted(empty_lista, reverse=True):
        list_points_ceiling.pop(index)
    

    #%% See which walls are adjacent
    connection_rooms, connection_rooms_walls = adjacency_walls.parallel_d_room(
        class_pcd_df, lines_position_t, df_walls_t, equation_df_walls_t, 
        pm.ex_v, pm.dist_th, pm.min_dist_w)
    
    # Eliminar duplicados usando set (esto puede cambiar el orden)
    connection_rooms_unique = list(set(map(tuple, connection_rooms)))
    connection_rooms_unique.sort()
                               
    connection_rooms_walls_unique = list(set(map(tuple,
                                                 connection_rooms_walls)))
    connection_rooms_walls_unique.sort()
    
    
    adjacency_complete = adjacency_intra.adjacent(lines_position_t)
    
    # Graphs
    graphs.inter_graph(connection_rooms_unique, path_storey, 
                       title_graph="inter_adjacency.html", 
                       page_title="INTER ROOM ADJACENCY GRAPH")
    
    graphs.intra_graph(adjacency_complete, lines_position_t, path_storey, 
                       title_graph="intra_adjacency.html", 
                       page_title="INTRA ROOM ADJACENCY GRAPH")
    
    graphs.graph_inter_adjacency(connection_rooms_unique, 
                                 connection_rooms_walls_unique,
                                 lines_position_t, 
                                 path_storey, 
                                 title_graph="inter_adjacency_with_walls.html",
                                 page_title="INTER ROOM ADJACENCY GRAPH")
    
    #%% Information to BIM -- Floor and Ceiling
    [points_floor, points_ceiling, 
     flattened_points_floor, 
     flattened_points_ceiling
     ] =  infoBIM.floor_ceiling_BIM(list_points_ceiling, list_points_floor, 
                                    boundary_pol, pm.dist_bound)
    height = np.round(np.mean(
        points_ceiling[:,2])-np.mean(points_floor[:,2]), 2)
    
    
    file_name = infoBIM.save_floor_ceiling(path_elements, points_floor, 
                                           points_ceiling, n_storey, 
                                           flattened_points_floor, 
                                           flattened_points_ceiling)
    
    #%% Exterior Walls
    room_data_list = infoBIM.info_walls(class_pcd_df, pm.ex_v, 
                                        lines_position_t, 
                                        df_walls_t, pm.b_f, 
                                        pm.min_p_int, 
                                        pm.max_dist_int)
    
    walls_with_intersections = infoBIM.info_walls_exterior(room_data_list, 
                                                           class_pcd_df, z_0, 
                                                           pm.min_ratio, 
                                                           pm.threshold, 
                                                           pm.iterat)
    
    # Extract coordinates
    points = class_pcd_df[class_pcd_df['id_element'] == 2][['x', 'y']].values
    
    # Create the alpha shape
    alpha_shape_walls = alph.alphashape(points, pm.alph_boundary)
    
    walls_exteriors = infoBIM.is_exterior(class_pcd_df, alpha_shape_walls, 
                                          lines_position_t, df_walls_t, 
                                          equation_df_walls_t, pm.ex_v, 
                                          pm.dist_max, pm.distance_th)
    
    lista_puntos_orden, resultados_union = infoBIM.only_exterior_walls(
        walls_exteriors, walls_with_intersections)
    
    
    file_name = infoBIM.save_exterior_walls(file_name, lista_puntos_orden, 
                                            n_storey, height)
    
    
    #%% Interior Walls
    [connection_rooms_unique, connection_rooms_walls_unique
     ] = adjacency_walls.check_adjacency_exterior_walls(
         connection_rooms_walls_unique, connection_rooms_unique, 
         walls_exteriors)
    
    sel = infoBIM.interior_walls(connection_rooms_walls_unique, 
                                 walls_with_intersections)
    
    rectangles, mean_z = overlaps_walls.overlap(sel)
    merged_rectangles = overlaps_walls.walls_to_IFC(rectangles, mean_z)
    
    file_name, all_int_walls = infoBIM.save_interior_walls(merged_rectangles, 
                                                           file_name,
                                                           n_storey, 
                                                           height)
        
    #%% Doors with trajectory
    df = pd.read_csv(path + folder_cloud + f'\\traj_{number_storey}.txt',
                     sep=' ', 
                     usecols=[0,1,2], 
                     names=['x','y','z'])
    
    # Get the points of the path every meter and apply a buffer
    # to know which rooms are close
    
    [pairs, points_door, df_1m, list_all, list_all_t
     ] = doors_with_trajectory.detect_doors_traj(class_pcd_df, 
                                                 df,
                                                 pm.dist_traj,
                                                 pm.b_traj)
    
    graphs.inter_graph(pairs, path_storey, 
                       title_graph="inter_connectivity.html", 
                       page_title="INTER ROOM CONNECTIVITY GRAPH")
    
    filtered_points_door = doors_with_trajectory.p_traj(df_1m, list_all_t, 
                                                        merged_rectangles,
                                                        pairs, 
                                                        points_door)
    
    # Door Points (filtering out doors that are too close)
    [doors_final, non_intersecting_lines
     ] = doors_with_trajectory.detect_doors_traj_2(df_1m, filtered_points_door, 
                                                   all_int_walls, pm.w_th, 
                                                   pm.w_width, pm.w_high)
    
    doors_non_intersecting = doors_with_trajectory.orientation_lines(
        non_intersecting_lines, floor, pm.w_width, pm.w_high)
    
    order_doors = infoBIM_doors_windows.save_doors_BIM(doors_final, 
                                                       doors_non_intersecting, 
                                                       pm.th_d)
    
    file_name = infoBIM.save_doors(file_name, order_doors, n_storey)
     
    # from src.elements_seg import doors, detect_points_doors
    # pm.pts_door = 4
    # pm.eps_door = 2.5
    # pm.min_pt_door = 2
    # path_doors = path_elements
    
    # bbx_door_lst, bbx_door_pts_lst, adj_room_lst = doors.door_detection(pm.vox_labels, pm.voxel_size, voxel_grid, pm.empty_in_lbl, 
    #                                                                         pm.occ_lbl, pm.pts_door, pm.eps_door, pm.min_pt_door, path_doors, 
    #                                                                         visualize=True, save=True)
    
    # w = 0.8 # ancho de la puerta
    # h = 2. # altura de la puerta
    # d = 0.12 # profundidad de la puerta
    # # Subdividir la lista en sublistas de 8 elementos cada una
    # sublistas = detect_points_doors.subdividir_lista(bbx_door_pts_lst, 8)

    # pts_door_f = []
    # for i in sublistas:
    # 	pts_door_f.append(detect_points_doors.detect_points_doors(i, w, h, d))

    # np.savetxt(path_doors + '\\doors_pts_f.txt', np.concatenate(pts_door_f))

    # doors_final = []
    # for i in pts_door_f:
    # 	doors_final.append(np.array(i))
    # order_doors = infoBIM_doors_windows.save_doors_BIM(doors_final, 
    #                                                    doors_non_intersecting, 
    #                                                    pm.th_d)
    
    # file_name = infoBIM.save_doors(file_name, order_doors, n_storey)
    
    #%% Windows
    
    filtered_points = windows.filter_points_boundary(pcd_to_windows, 
                                                     boundary_pol, 
                                                     pm.th_boundary)
    
    all_points = windows.windows(filtered_points, pm.dist_th_w, pm.init_n, 
                                 pm.iterat, pm.iter_i, pm.min_a, pm.max_a)

    p_windows = infoBIM_doors_windows.final_points_windows(
        all_points, class_pcd_df, pm.h_f_w, pm.th_g, pm.th_w, pm.s_w, pm.s_h)
    order_windows = infoBIM_doors_windows.save_windows_BIM(p_windows, pm.th_w)  
    
    file_name = infoBIM.save_windows(file_name, order_windows, n_storey)
    
        
    #%% Graphs (with networkx)
    # Adjacent rooms graph
    graph.graph(class_pcd_df, connection_rooms_unique,
                tit='Inter-adjacency room graph')
    # Connected rooms graph
    graph.graph(class_pcd_df, pairs, 
                tit='Inter-connectivity room graph')  
        
    
    #%% Save
    
    # Save point cloud of storey without stairs
    st_without_stairs.to_csv(path_elements + '\\storey_without_stairs.txt',
                             sep=' ', index=False)    
    # Final PointCloud
    np.savetxt(path_elements + '\\list_point_floor.txt', 
               np.concatenate(list_points_floor))      
    np.savetxt(path_elements + '\\list_point_ceiling.txt', 
               np.concatenate(list_points_ceiling))       
    class_pcd_df['id_entity'].fillna(99, inplace=True)
    class_pcd_df['id_entity'] = class_pcd_df['id_entity'].astype(int)
    class_pcd_df.to_csv(path_storey + "\\segmetation_pointcloud_final.csv", 
                        index=False)
    class_pcd_df_copy = class_pcd_df.copy()
    class_pcd_df_copy.loc[class_pcd_df_copy['id_entity'] == 99, 
                          'id_element'] = 99
    class_pcd_df_copy.to_csv(path_storey + "\\segmetation_pointcloud_final_f.csv",
                             index=False)
    # Doors
    # np.savetxt(path_elements + '\\points_doors_non_intersecting.txt', 
    #            np.concatenate(doors_non_intersecting))
    # np.savetxt(path_elements + '\\points_doors.txt', 
    #            np.concatenate(doors_final))
    # np.savetxt(path_elements + '\\all_points_doors.txt', np.concatenate(order_doors))
    # Windows
    filtered_points[['x','y','z']].to_csv(path_elements + "\\boundary_020.csv", 
                                          index=False)
    np.savetxt(path_elements + '\\points_windows.txt',np.concatenate(p_windows))

    pm.vox_idx = np.array([])
    pm.clean_idx = np.array([])
    pm.vox_labels = pd.DataFrame() 


