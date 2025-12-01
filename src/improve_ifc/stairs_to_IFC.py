# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import open3d as o3d
from src.utils import array_to_o3d
import alphashape as alph
from shapely.geometry import Point


# # The point cloud corresponding to the stairs is read
# pcd_file = r"C:\Users\rosam\OneDrive - Universidade de Vigo\Escritorio\Final_Code\results\csBilbao\Storey_3\stairs\stairs.txt"
# point_cloud_building = pd.read_csv(r"C:\Users\rosam\OneDrive - Universidade de Vigo\Escritorio\Final_Code\results\csBilbao\Storeys_new\Storey_3.csv", sep=',')[['x','y','z']]

def stairs_to_IFC(pcd_file, point_cloud_building, nt, threshold, init_n,
                  iterat, max_dist_stairs, eps_stairs, pts_stairs,
                  vertical_step, horizontal_step, n_storey):
    
    stairs =  pd.read_csv(pcd_file, sep=' ', usecols=[0,1,2,3])
    points_stairs = np.array(stairs[['x','y','z']])
    
    cl = array_to_o3d.array_to_o3d_point_cloud(np.array(stairs[['x','y','z']]))
    if cl.has_normals()==False: 
       cl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(40))
        
    horizontal_index = np.where(np.absolute(np.asarray(cl.normals)[:,2])>nt)[0]  
    p_new = array_to_o3d.array_to_o3d_point_cloud(np.array(stairs[['x','y','z']].iloc[horizontal_index]))
    
    # Apply RANSAC to detect a plane in the point cloud
    plane_model, inliers = p_new.segment_plane(distance_threshold=threshold,
                                                             ransac_n=init_n,
                                                             num_iterations=iterat)
    # Create a new point cloud with the points belonging to the plane
    inlier_cloud = p_new.select_by_index(inliers)
    # Color the detected plane to visualize it
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) 
    o3d.visualization.draw_geometries([inlier_cloud, p_new])
    
    # Select or largest cluster
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as ctxtman:
        labels = np.array(inlier_cloud.cluster_dbscan(eps=eps_stairs,
                                                      min_points=pts_stairs, 
                                                      print_progress=True))
    
    # Total number of clusters (excluding noise)
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters (excluding noise)")
    
    # Filter out noise (label -1) and count the number of points in each cluster
    cluster_sizes = np.bincount(labels[labels >= 0])
    
    # Finding the index of the largest cluster
    largest_cluster_idx = np.argmax(cluster_sizes)
    print(f"Largest cluster is cluster {largest_cluster_idx} with {cluster_sizes[largest_cluster_idx]} points")
    
    # Filter the points that belong to the largest cluster
    largest_cluster_mask = (labels == largest_cluster_idx)
    
    # Create a new point cloud with only the points from the largest cluster
    largest_cluster_pcd = inlier_cloud.select_by_index(np.where(largest_cluster_mask)[0])
    o3d.visualization.draw_geometries([largest_cluster_pcd])
    
    # Points of the largest cluster
    points = np.array(largest_cluster_pcd.points)
    
    # Extract the X and Y coordinates (ignoring Z)
    xy_points = points[:, :2]
    z_point = np.round(np.max(points[:, 2]), 2)
    
    # Find the extreme values ​​in X and Y
    min_x, min_y = np.round(np.min(xy_points, axis=0))
    max_x, max_y = np.round(np.max(xy_points, axis=0))
    
    # Select the extreme points
    extreme_points = [list((min_x, min_y, z_point)) ,
                      list((max_x, min_y, z_point)) , 
                      list((max_x, max_y, z_point)) , 
                      list((min_x, max_y, z_point))]
    
    length = np.round(max_y - min_y, 2)
    width = np.round(max_x - min_x, 2)
    
    min_point_xz = np.round((np.min(points_stairs[:,0]), 
                             min_y, np.min(points_stairs[:,2])), 2)
    max_point_xz = np.round((np.max(points_stairs[:,0]),
                             min_y, np.max(points_stairs[:,2])), 2)
    
    # Looking for the two points that are on the contour of the building's point cloud.
    points_c = point_cloud_building[['x', 'y']].values
    # Create the alpha shape
    alpha_shape_walls = alph.alphashape(points_c, 2)
    
    # Ready to save points near the contour
    close_points = []
    for point in extreme_points:
        point_shapely = Point(point[0], point[1])
        distance = alpha_shape_walls.boundary.distance(point_shapely)
        if distance <= max_dist_stairs:
            close_points.append(point)
    
    # See if the two points are further to the left or further to the right (or higher or lower)
    puntos_no_en_segundo = [punto for punto in extreme_points 
                            if punto not in close_points]  
    
    # Extract the coordinates of the points
    point1 = close_points[0]
    point2 = close_points[1]
    
    # Check if they are aligned on the X or Y axis
    if point1[0] == point2[0]:
        print("The points are aligned on the X axis.")
        if list(set([punto[0] for punto in puntos_no_en_segundo]))[0] > list(
                set([punto[0] for punto in close_points]))[0]:
            print('Contour points are on the left')
        
        else:
            print('Contour points are on the right')
            first_point = min(close_points, key=lambda punto: punto[1])
            second_point = min(puntos_no_en_segundo, key=lambda punto: punto[1])
            
            first_point_i = [second_point[0], second_point[2]]
            end_point_i = [min_point_xz[0] - horizontal_step, 
                           min_point_xz[2] - vertical_step]
            
            # Initialize the starting point
            current_point = first_point_i.copy()
    
            # Create a list to store the points
            points = [np.array(current_point)]
    
            # Perform the alternating movement
            while True:
                # Downward movement
                new_point = current_point + np.array([0, -vertical_step])
                
                # Check if the new point does not exceed the end
                if new_point[1] >= end_point_i[1]:
                    points.append(new_point)
                    current_point = new_point
                else:
                    break
                
                # Movement to the left
                new_point = current_point + np.array([-horizontal_step, 0])
                
                # We check if the new point does not exceed the end
                if new_point[0] >= end_point_i[0]:
                    points.append(new_point)
                    current_point = new_point
                else:
                    break
    
            points_with_column = []
            for point in points:
                new_point = np.insert(point, 1, 4)
                points_with_column.append(new_point)
    
            p_stairs = np.vstack((np.array(first_point), points_with_column))
            
            # To the other side
            first_point_o = max(close_points, key=lambda punto: punto[1])
            second_point_o = max(puntos_no_en_segundo, key=lambda punto: punto[1])
            
            first_point_i_o = [second_point_o[0], second_point_o[2]]
            end_point_i_o = [min_point_xz[0] - horizontal_step, 
                             max_point_xz[2] + vertical_step]
            
            # Initialize the starting point
            current_point_o = first_point_i_o.copy()
    
            # Create a list to store the points
            points_o = [np.array(current_point_o)]
    
            # Perform the alternating movement
            while True:
                # Upward movement
                new_point_o = current_point_o + np.array([0, vertical_step])
                
                # Check if the new point does not exceed the end
                if new_point_o[1] <= end_point_i_o[1]:
                    points_o.append(new_point_o)
                    current_point_o = new_point_o
                else:
                    break
                 
                # Movement to the left
                new_point_o = current_point_o + np.array([-horizontal_step, 0])
                
                # Check if the new point does not exceed the end
                if new_point_o[0] >= end_point_i_o[0]:
                    points_o.append(new_point_o)
                    current_point_o = new_point_o
                else:
                    break
    
            points_with_column_o = []
            for point in points_o:
                new_point_o = np.insert(point, 1, 6)
                points_with_column_o.append(new_point_o)
    
            p_stairs_o = np.vstack((np.array(first_point_o), points_with_column_o))
      
            np.vstack((p_stairs, p_stairs_o))
            np.savetxt('pts_stairs' + str(n_storey) + '.txt', np.vstack((p_stairs,
                                                                         p_stairs_o)))
                    
    elif point1[1] == point2[1]:
        print("The points are aligned on the Y axis.")
       
    else:
        print("The points are not aligned on either the X or Y axis.")

    return length, width, p_stairs, p_stairs_o


def file_stairs_IFC(length, width, n_storey, p_stairs, p_stairs_o):
    # File name
    file_name = 'stairs_' + str(n_storey) + '.txt'
    
    # File header
    header = "storey_id,element_type,element_id,stair_width,land_width,land_length,no_pts,pt1_x,pt1_y,pt1_z,pt2_x,pt2_y,pt2_z,pt3_x,pt3_y,pt3_z,pt4_x,pt4_y,pt4_z"
    
    pts_stairs = np.vstack((p_stairs[-2], p_stairs[1], p_stairs_o[1], p_stairs_o[-2]))
    pts_stairs = np.round(pts_stairs,2)
    row = [n_storey, 'stairs', 0, (length/2)-0.1, width, length, 4] + [coord for point in pts_stairs for coord in point]
    row_str = ','.join(map(str, row))
    with open(file_name, 'w') as archivo:
        archivo.write(header + '\n')
        archivo.write(row_str + '\n')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

