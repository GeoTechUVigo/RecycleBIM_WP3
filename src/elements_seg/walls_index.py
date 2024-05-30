# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import pandas as pd
from shapely.geometry import Point
import alphashape as alph
import sys

from scipy.spatial import ConvexHull
import matplotlib as plt
from scipy.spatial.distance import cdist

from src.utils import functions_geom
from src.utils import functions_walls


def walls(class_pcd_df, ex_v, max_distance, threshold, init_n, 
          iterat, h, dist_pl, min_angle, max_angle, dist_centroids_1, path_walls):
    """
    Celculate walls

    Parameters
    ----------
    class_pcd_df : DataFrame
        It is the pointcloud.
    ex_v : list
        Rooms in which no walls were detected.
    max_distance : float
        Maximum contour distance.
    threshold : float
        Threshold for ransac algorithm.
    init_n : int
        Number of initial points to be considered inliers in each iteration.
    iterat : int
        Number of iterations.
    h : float
        Minimum wall height.
    dist_pl : float
        Maximum distance between two parallel walls.
    min_angle : int
        Minumum angle.
    max_angle : int
        Maximun angle.
    dist_centroids_1 : float
        Maximum distance of centroids.
    path_walls : str
        Path to save information of walls.

    Returns
    -------
    equation_df_walls_t : list
        Plane equations that represent the different walls.
    df_walls_t : list
        Index of the points of the point clouds that represent the different walls.
    lines_position_t : list
        Line equations that represent the different walls.
    z_0 : float
        Midpoint of wall height.
    walls_df : DataFrame
        DataFrame containing the rooms and the walls of each room.
    class_pcd_df : DataFrame
        DataFrame with new id_entity attribute.

    """
    i = 0 # Id storey
    k = 2 # Id of elements   
    storey_i = class_pcd_df['id_storey'] == i
    element_k = class_pcd_df['id_element'] == k
    
    equation_df_walls_t = []
    df_walls_t = []
    lines_position_t = []
    df_index = []
    all_data = []
    
    for room in [valor for valor in range(1, len(class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v]:
        print(room)

        room_j = class_pcd_df['id_room'] == room        
        ijk = storey_i & room_j & element_k

        # Now I want to study or contour for each room.
        r = class_pcd_df.loc[ijk, ['x', 'y']].values                 
        c_r = alph.alphashape(class_pcd_df.loc[ijk, ['x', 'y']].values, 2.0)
                
        # Collect all points that are within a certain distance
        polygon_border = c_r.boundary
        points_in_border = []
        # Check if each point in the point cloud is on the edge of the polygon
        for point in r:
            point_shapely = Point(point)  
            dist = polygon_border.distance(point_shapely)
            
            if dist < max_distance:
                points_in_border.append(point)    
        
        positions = []
        for point in points_in_border:
            indices = np.where((r == point).all(axis=1))
            positions.extend(indices[0])
            
                   
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(class_pcd_df.loc[ijk, ['x', 'y', 'z']].values[positions])
        
        
        if cloud.has_normals()==False: 
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(40))
            
        vertical_index = np.where(np.absolute(np.asarray(cloud.normals)[:,2])<0.1)[0]  
      
        N = len(class_pcd_df.loc[ijk, ['x', 'y', 'z']].iloc[positions].iloc[vertical_index])
        count = 0
                
        equation_df_walls = []
        df_walls = []
        target = class_pcd_df.loc[ijk, ['x', 'y', 'z']].iloc[positions].iloc[vertical_index]
        while count < (1 - 0.05) * N:           
            cloud_target = o3d.geometry.PointCloud()
            cloud_target.points = o3d.utility.Vector3dVector(target.values)
            plane_model, inliers = cloud_target.segment_plane(distance_threshold=threshold, ransac_n=init_n, num_iterations=iterat) 
            index_plane_list = target.index[inliers]
            target = target.drop(index_plane_list)
            
            count += len(inliers)
            
            z_diff = class_pcd_df.loc[ijk, ['x', 'y', 'z']].iloc[positions].iloc[vertical_index].loc[index_plane_list]['z'].max() - class_pcd_df.loc[ijk, ['x', 'y', 'z']].iloc[positions].iloc[vertical_index].loc[index_plane_list]['z'].min()

            if z_diff > h:
                equation_df_walls.append(plane_model)
                df_walls.append(index_plane_list)            
            
                
        # Up to this point we have the equations provided by RANSAC and the indices of the points belonging to each plane.
        vectors_without_numpy = functions_geom.convert_in_list(equation_df_walls)
        
        # Save only points of equations.           
        z_0 = np.median(class_pcd_df[['z']])
        lines_position = [[vector[0], vector[1], z_0 * vector[2] + vector[3]] for vector in vectors_without_numpy]
        
                          
        # Verify parallelism between all pairs of equations
        i = 0
        while i < len(lines_position):
            j = i + 1
            while j < len(lines_position):
                if functions_geom.are_planes_parallel(lines_position[i], lines_position[j]):
                    dist_parallel = functions_geom.distance_lines_paralell(lines_position[i], lines_position[j])
                    if dist_parallel < dist_pl:
                        if len(df_walls[i]) > len(df_walls[j]): 
                            equation_df_walls.pop(j)
                            df_walls.pop(j)
                            lines_position.pop(j)
                        else:
                            equation_df_walls.pop(i)
                            df_walls.pop(i)
                            lines_position.pop(i)
                        continue
                j += 1
            i += 1
        
        
        i = 0
        while i < len(lines_position):
            j = i + 1
            while j < len(lines_position):
                angulo_entre_rectas = functions_geom.calculate_angle(lines_position[i], lines_position[j])
                if min_angle < angulo_entre_rectas < max_angle:
                    cent1 = functions_geom.calculate_centroid(np.array(class_pcd_df.loc[df_walls[i]][['x', 'y']]))
                    cent2 = functions_geom.calculate_centroid(np.array(class_pcd_df.loc[df_walls[j]][['x', 'y']]))
                    dist_cent = functions_geom.calculate_distance_centroids(cent1, cent2)

                    if dist_cent < dist_centroids_1:
                        if len(df_walls[i]) > len(df_walls[j]):
                            equation_df_walls.pop(j)
                            df_walls.pop(j)
                            lines_position.pop(j)
                        else:
                            equation_df_walls.pop(i)
                            df_walls.pop(i)
                            lines_position.pop(i)
                        continue
                j += 1
            i += 1

        # Save the walls of each room
        for index, i in enumerate(df_walls):
            df = class_pcd_df.loc[i][['x', 'y', 'z']]            
            df['id_room'] = int(room)
            df['id_entity'] = index
            all_data.append(df)
            df_index.append(i)                          
      
                
        equation_df_walls_t.append(equation_df_walls)
        df_walls_t.append(df_walls)
        lines_position_t.append(lines_position)
                        
        
        for zindex, l in enumerate(df_walls):       
            class_pcd_df.loc[l, 'id_entity'] = int(zindex)
                
            
    # Combine all DataFrames into one
    walls_df = pd.concat(all_data, ignore_index=True)

    walls_df.to_csv(path_walls + "\\dataframe_walls.csv", index=False)
    
    return equation_df_walls_t, df_walls_t, lines_position_t, z_0, walls_df, class_pcd_df


            
       
            
            
    