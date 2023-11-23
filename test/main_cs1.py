# -*- coding: utf-8 -*-

path = input("Data folder path: ")
folder_cloud = "\\cs1"
folder_results = input("Results folder path: ")
name_cloud = "\\cs1_storey_0.txt"
name_f = "_pointcloud_0_01.txt"


#%%
# Packages (Python)
import numpy as np
import open3d as o3d
import pandas as pd
import laspy as lp
import alphashape as alph
import itertools
import matplotlib as plt
from scipy.spatial import cKDTree, distance
import time
import os
import cc3d
from scipy.stats import mode
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import cdist

# Packages 
from src.utils import bounding_polygon_filter
from src.utils import array_to_o3d
from src.utils import load_pointcloud as lpc
from src.preprocessing import subsampling as sb
from src.preprocessing import clean_exteriors
from src.elements_seg import extract_floor_ceiling as efc
from src.utils import remove_no_continuous as rop
from src.morph_seg import erosion
from src.morph_seg import individualisation
from src.morph_seg import dilation
from src.morph_seg import occupied_voxels_classification as ovc
from src.morph_seg import point_cloud_classification as pcc
from src.elements_seg import doors
from src.elements_seg import walls

#%%
# Parameters
voxel_size = 0.1
eps = 0.3
pt = 2
nt = 0.9
alpha = 3.2
ransac_th = 0.12
zmax = 1000
zmin = -1000
dist =  0.1

pcd_in = None
se = None
vox_idx = np.array([])
clean_idx = np.array([])
vox_labels = pd.DataFrame() 
empty_in_lbl = 1
occ_lbl = 10 
width_door = 1.5

pts_door = 2 
eps_door = 2 
min_pt_door = 2 

min_ratio = 0.05
threshold = 0.09
iterations = 1000
init_n = 3
h = 2.7

images = True
write = False
#%%

# Check if the folder 'Results' already exists
if not os.path.exists(folder_results):
    # If it doesn't exist, we create it
    os.mkdir(folder_results)
else:
    print('The folder already exists.')

if not os.path.exists(folder_results + folder_cloud):
    # If it doesn't exist, we create it
    os.mkdir(folder_results + folder_cloud)
else:
    print('The folder already exists.')

# Save start time
ini_time = time.time()

# Open a file in write mode ('w')
with open(folder_results + folder_cloud + '\\time_morph_seg' + name_f, 'w') as file:
    
    # Load pointcloud
    pcd_file = path + folder_cloud + name_cloud
    #pcd_file = folder_results + folder_cloud + name_cloud
    pcd_in = lpc.load_pointcloud(name_cloud, folder_cloud, pcd_file, visualize=False)
    #pcd_in = load_file.load_file(pcd_file, header=False)
    
    last_column = pcd_in.columns[-1]
    # Check if the last column has the attribute 'id_storey'
    if 'id_storey' in last_column:
        print("The last column has the attribute 'id_storey'")
    else:
        print("The last column has not the attribute'id_storey'")
        pcd_in['id_storey'] = 0
        print(pcd_in)
    
    
    
    pcd_in_columns = pcd_in.columns.tolist()
   
    # Time
    time_1 = time.time() - ini_time
    file.write(f"The point cloud takes {time_1} seconds to load. It has {len(pcd_in)} points. \n")
    
    # Voxelización
    pcd_down_o3d, vox_idx = sb.voxel_downsample(
        array_to_o3d.array_to_o3d_point_cloud(pcd_in.loc[:, 'x':'z'].values), 
        voxel_size, order_by_index=(True))
    
    # Time
    time_2 = time.time() - ini_time 
    time_2_1 = time_2 - time_1
    file.write(f"The point cloud takes {time_2_1} seconds to voxelizate ({time_2} seconds from start). It has {len(pcd_down_o3d.points)} voxels. \n")
    
    
    # Clean exteriors
    pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(pcd_down_o3d,
                                                                epsilon = eps, 
                                                                points_min = pt, 
                                                                return_index=True)
    
    # Time
    time_3 = time.time() - ini_time
    time_3_2 = time_3 - time_2
    file.write(f"The point cloud takes {time_3_2} seconds to clean exteriors ({time_3} seconds from start). It has {len(pcd_down_clean.points)} voxels. \n")

    # Extract floor, celling and obstacles
    floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil = efc.extract_floor_ceiling(pcd_down_clean, nt,
                                                                                        search_param=o3d.geometry.KDTreeSearchParamKNN(40), 
                                                                                        ransac_th=ransac_th, return_index=True, timestamp=None, 
                                                                                        visualize=False)
    
    # Time
    time_4 = time.time() - ini_time
    time_4_3 = time_4 - time_3
    file.write(f"The point cloud takes {time_4_3} seconds to extract floor ({len(floor.points)} voxels) and ceiling ({len(ceiling.points)} voxels). ({time_4} seconds from start) \n")
    
    
    # Remove non continuos celling
    boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx = rop.remove_no_continuous(pcd_in, floor, obstacles, ceiling, idx_floor, idx_obs,
                                                                                       idx_ceil, eps, pt, alpha, voxel_size, zmax, zmin, 
                                                                                       dist, vox_idx, idx_clean, visualize = False, save = False)    
   
    
    # Clean point cloud
    # xyz coordinates of cleaned point cloud
    pcd_clean_arr = pcd_in.loc[clean_idx, 'x': 'z'].values

    # Create VoxelGrid filtered point cloud    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(array_to_o3d.array_to_o3d_point_cloud(pcd_clean_arr), 
                                                            voxel_size)


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

    # Empty cloud
    empty_cloud = array_to_o3d.array_to_o3d_point_cloud(empty_cent)
    empty_cloud.paint_uniform_color([.0, .3, .9])
                                                                                              
        
    # Classify empty voxels as indoor (1) or outside (0)
    filt_empty_cloud, idx_filt = bounding_polygon_filter.bounding_polygon_filter(empty_cloud, 
                                                                   boundary_pol, 
                                                                   zmax, zmin, 
                                                                   eliminate='outside')
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
    vox_lbl[in_idx] = 1

    # Save in class attribute
    vox_labels['occ'] = vox_lbl

    vxl_idx_arr = np.hstack((vxl_idx_arr, vox_lbl[:, np.newaxis]))
    labelled_vox_df = pd.DataFrame(vxl_idx_arr, columns=['i', 'j', 'k','scalar'])


    # Time
    time_5 = time.time() - ini_time
    time_5_4 = time_5 - time_4
    file.write(f"The point cloud takes {time_5_4} seconds to create a DataFrame for save differents attributes ({time_5} seconds from start). \n")
    
    
    # 3D morphological erosion
    image_erode, se_erosion = erosion.morphological_erosion(labelled_vox_df, se, 
                                                width_door, empty_in_lbl, 
                                                voxel_size)
    

    erode_image = image_erode
    erode_image_df = labelled_vox_df.copy()
    erode_image_df.loc[:, 'scalar'] = 0
    erode_image_df.loc[:, 'scalar'] = erode_image.flatten()
    erode_pcd = array_to_o3d.array_to_o3d_point_cloud(erode_image_df.loc[erode_image_df.scalar==1, 'i':'k'].values)
    erode_pcd.paint_uniform_color([.7, .7, .7])
        
        
    # Time
    time_6 = time.time() - ini_time
    time_6_5 = time_6 - time_5
    file.write(f"The point cloud takes {time_6_5} seconds to  erosion ({time_6} seconds from start). It has {len(erode_pcd.points)} voxels. \n")
        
    # Individualisation
    labels_out = individualisation.room_individualisation(image_erode, voxel_size)

    labelled_erode_im_df = labelled_vox_df.copy()
    labelled_erode_im_df.loc[:, 'scalar'] = 0
    labelled_erode_im_df.loc[:,'scalar'] = labels_out.flatten()

    o3d_lbl_erode = array_to_o3d.array_to_o3d_point_cloud(labelled_erode_im_df.loc[labelled_erode_im_df.scalar>0,'i':'k'].values)
    erode_lbls = labelled_erode_im_df.loc[labelled_erode_im_df.scalar>0 ,'scalar'].values
    max_label = erode_lbls.max()
    colors = plt.colormaps.get_cmap("tab20")(erode_lbls / (max_label if max_label > 0 else 1))
    colors[erode_lbls < 0] = 0
    o3d_lbl_erode.colors = o3d.utility.Vector3dVector(colors[:, :3])


    # Time
    time_7 = time.time() - ini_time
    time_7_6 = time_7 - time_6
    file.write(f"The point cloud takes {time_7_6} seconds to individualisation ({time_7} seconds from start). It has {len(o3d_lbl_erode.points)} voxels. \n")


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
    colors = plt.colormaps.get_cmap("tab20")(dilated_lbls / (max_label if max_label > 0 else 1))
    colors[dilated_lbls < 0] = 0
    o3d_lbl_dilated.colors = o3d.utility.Vector3dVector(colors[:, :3])
            

    # Time
    time_8 = time.time() - ini_time
    time_8_7 = time_8 - time_7
    file.write(f"The point cloud takes {time_8_7} seconds to dilation ({time_8} seconds from start). It has {len(o3d_lbl_dilated.points)} voxels. \n")


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
    colors = plt.colormaps.get_cmap("tab20")(class_lbls / (max_label if max_label > 0 else 1))
    colors[class_lbls < 0] = 0
    o3d_class_vxl.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Time
    time_9 = time.time() - ini_time
    time_9_8 = time_9 - time_8
    file.write(f"The point cloud takes {time_9_8} seconds to classify occupied voxels ({time_9} seconds from start). It has {len(o3d_class_vxl.points)} voxels. \n")


    # Point cloud classification
    class_pts_arr, idx_in_pcd = pcc.point_cloud_classification(pcd_in, pcd_clean_arr, 
                                                                     room_lab_arr, voxel_grid, 
                                                                     voxel_size)
    pcd_in_columns.append('id_room')
    
    # Classified input point cloud
    class_pcd_df = pd.DataFrame(np.hstack((pcd_in.loc[idx_in_pcd, 'x':'id_storey'].values,
                                           class_pts_arr[:, np.newaxis])),
                                columns=pcd_in_columns)
    
    # Time
    time_10 = time.time() - ini_time
    time_10_9 = time_10 - time_9
    file.write(f"The point cloud takes {time_10_9} seconds to classify input pointcloud ({time_10} seconds from start). It has {len(class_pcd_df)} points. \n")


    # Extract floor, ceiling and obstacles
    floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil = efc.extract_floor_ceiling(array_to_o3d.array_to_o3d_point_cloud(class_pcd_df.loc[:, 'x':'z'].values), nt,
                                                                                        search_param=o3d.geometry.KDTreeSearchParamKNN(40), 
                                                                                        ransac_th=ransac_th, return_index=True, timestamp=None, 
                                                                                        visualize=False)
    
    # Create a Series with all values
    new_values = pd.Series([0] * len(class_pcd_df.iloc[idx_floor]),
                           index=class_pcd_df.iloc[idx_floor].index, name='id_element')
    new_values_1 = pd.Series([1] * len(class_pcd_df.iloc[idx_ceil]), 
                             index=class_pcd_df.iloc[idx_ceil].index, name='id_element')
    new_values_2 =  pd.Series([2] * len(class_pcd_df.iloc[idx_obs]), 
                              index=class_pcd_df.iloc[idx_obs].index, name='id_element')
    # Add the Series as a new column to the DataFrame
    class_pcd_df['id_element'] = 'NaN'
    class_pcd_df.loc[new_values.index, 'id_element'] = new_values
    class_pcd_df.loc[new_values_1.index, 'id_element'] = new_values_1
    class_pcd_df.loc[new_values_2.index, 'id_element'] = new_values_2

    # Time
    time_11 = time.time() - ini_time
    time_11_10 = time_11 - time_10
    file.write(f"The point cloud takes {time_11_10} seconds to extract floor, ceiling and obstacles ({time_11} seconds from start). It has {len(floor.points)} points of floor, {len(ceiling.points)} points of ceiling and {len(obstacles.points)} points of obstacles. \n")

    # Doors
    bbx_door_lst, bbx_door_pts_lst, adj_room_lst = doors.door_detection(vox_labels,
                                                                        voxel_size, voxel_grid, 
                                                                        empty_in_lbl, occ_lbl, 
                                                                        pts_door, eps_door, 
                                                                        min_pt_door, 
                                                                        visualize=False, save=False)
    
    # Time
    time_12 = time.time() - ini_time
    time_12_11 = time_12 - time_11
    file.write(f"The point cloud takes {time_12_11} seconds to find doors ({time_12} seconds from start). It has {len(bbx_door_lst)} doors. \n")

    
    # Walls
    segmented_walls_c_f, class_pcd_df = walls.walls(class_pcd_df, min_ratio, threshold, init_n, h)
    
    # Time
    time_13 = time.time() - ini_time
    time_13_12 = time_13 - time_12
    file.write(f"The point cloud takes {time_13_12} seconds to find walls ({time_13} seconds from start). It has {len(segmented_walls_c_f)} points. \n")

end_time = time.time()




"""
Next, modify attributes:
    02 -> wall
    04 -> floor
    06 -> ceiling
    09 -> doors
    11 -> others
        
"""    
    
class_pcd_df.loc[class_pcd_df['id_element'] == 0, 'id_element'] = '04'
class_pcd_df.loc[class_pcd_df['id_element'] == 1, 'id_element'] = '06'
class_pcd_df.loc[class_pcd_df['id_element'] == 2, 'id_element'] = '11'

class_pcd_df.loc[class_pcd_df['walls'] == 1, 'id_element'] = '02'

class_pcd_df = class_pcd_df.drop('walls', axis=1)


class_pcd_df['id_room'] = class_pcd_df['id_room'].astype(int).astype(str).str.zfill(2)
class_pcd_df['id_storey'] = class_pcd_df['id_storey'].astype(int).astype(str).str.zfill(2)


# Doors (In Case 3 and 4 there are not doors)
doors_df = pd.DataFrame(bbx_door_pts_lst)
pcd_points = class_pcd_df[['x', 'y', 'z']].values
doors_points = doors_df.values

# Calculates the Euclidean distance between the points of doors_df and class_pcd_df
distances = cdist(doors_points, pcd_points)

# Find the index of the closest point in class_pcd_df for each point in doors_df
closest_indices = np.argmin(distances, axis=1)
class_pcd_df['door'] = 0
class_pcd_df.loc[closest_indices, 'door'] = 1
class_pcd_df.loc[class_pcd_df['door'] == 1, 'id_element'] = '09'
class_pcd_df = class_pcd_df.drop('door', axis=1)


# Dataframe
class_pcd_df.to_csv(folder_results + folder_cloud + "\\seg" + name_f, sep=' ', header=True, index=False)


# Images
if images == True:
    o3d.visualization.draw_geometries([pcd_down_o3d], window_name="Initial PointCloud")
    o3d.visualization.draw_geometries([pcd_down_clean], window_name="Clean PointCloud")
    o3d.visualization.draw_geometries([floor, obstacles, ceiling],
                                      window_name="Cleaned PointCloud after outliers filtering by DBSCAN")
    o3d.visualization.draw_geometries([floor], window_name="Floor PointCloud")
    o3d.visualization.draw_geometries([obstacles], window_name="Obstacles PointCloud")
    o3d.visualization.draw_geometries([ceiling], window_name="Ceiling PointCloud")
    o3d.visualization.draw_geometries([empty_cloud],
                                      window_name="Occupied voxels: {} Empty voxels: {}".format(len(idx_occ_vxl), 
                                                                                                        len(empty_cloud.points)))
    o3d.visualization.draw_geometries([in_empty_pcd], window_name="Indoor empty space after boundary filter")
    o3d.visualization.draw_geometries([in_filt_pcd2], window_name="Indoor empty space after distance point-to-plane filter")
    o3d.visualization.draw_geometries([erode_pcd], window_name='Erode empty space')    
    o3d.visualization.draw_geometries([o3d_lbl_erode], window_name="Individualised spaces") 
    o3d.visualization.draw_geometries([o3d_lbl_dilated], window_name='Dilated spaces')     
    o3d.visualization.draw_geometries([o3d_class_vxl], window_name="Classified occupied voxels")
    
       
if write == True:
    o3d.io.write_point_cloud(folder_results + folder_cloud + "\\Indoor empty space after boundary filter.xyz", in_empty_pcd)
    o3d.io.write_point_cloud(folder_results + folder_cloud + '\\indoor_empty_space.xyz', in_filt_pcd2)    
    erode_image_df.to_csv(folder_results + folder_cloud + '\\erode_im.csv', sep = ",", index=False) #añadí sep = ","
    o3d.io.write_point_cloud(folder_results + folder_cloud + "\\erosion.xyz", erode_pcd) 
    labelled_erode_im_df.to_csv(folder_results + folder_cloud + '\\erode_labeled_im.csv', index=False)
    o3d.io.write_point_cloud(folder_results + folder_cloud + "\\individualisation.xyz", o3d_lbl_erode)
    dilated_labelled_df.to_csv(folder_results + folder_cloud + '\\dilated_labelled_im.csv', index=False)
    o3d.io.write_point_cloud(folder_results + folder_cloud + "\\ditation.xyz", o3d_lbl_dilated)
    classified_vox_df.to_csv(folder_results + folder_cloud + '\\classified_vox_im.csv', index=False)             
    o3d.io.write_point_cloud(folder_results + folder_cloud + "\\classification.xyz", o3d_class_vxl)

