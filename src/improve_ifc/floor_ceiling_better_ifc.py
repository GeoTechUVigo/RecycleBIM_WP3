# -*- coding: utf-8 -*-

import numpy as np
from parameters import parameters_Bilbao as pm
from src.utils import array_to_o3d
from src.preprocessing import subsampling as sb
from src.preprocessing import clean_exteriors
from src.elements_seg import extract_floor_ceiling as efc
from src.utils import remove_no_continuous as rop
from src.utils import remove_no_continuous as ropf

from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


def points(df_points):
    # Filter columns that contain 'pt' in their names
    columns_of_interest = [col for col in df_points.columns 
                           if col.startswith('pt')]
    
    # Extract the x, y, z coordinates from the columns
    x_coords = df_points[columns_of_interest[::3]].values 
    y_coords = df_points[columns_of_interest[1::3]].values 
    z_coords = df_points[columns_of_interest[2::3]].values 
    
    # Combine x, y, z coordinates into a single array
    points = np.stack((x_coords, y_coords, z_coords), axis=-1)
    points_2 = points[0][~np.isnan(points[0]).any(axis=1)]
    
    return points_2

# Detect contour
def improve_floor_ceiling(pcd_in_t, df, last_storey, distancia_max=0.5):
    
    for id_storey, pcd_in in pcd_in_t.groupby('id_storey'):        
        if np.unique(pcd_in[['id_storey']])[0] == 0:
            df_floor_0 = df[(df['element_type'] == 'floor') 
                            & (df['storey_id'] == 0)]
            points_2 = points(df_floor_0)

            # Voxelization
            pcd_down_o3d, vox_idx = sb.voxel_downsample(
                array_to_o3d.array_to_o3d_point_cloud(
                    pcd_in.loc[:, 'x':'z'].values), 
                pm.voxel_size, order_by_index=(True))
            
            # Clean exteriors
            pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(
                pcd_down_o3d, epsilon = pm.eps_1, points_min = pm.pt, 
                return_index=True)
            
            # Extract floor, celling and obstacles
            [floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil
             ] = efc.extract_floor_ceiling(pcd_down_clean, pm.nt, 
                                           search_param=pm.s_p, 
                                           ransac_th=pm.ransac_th,
                                           iterat=pm.iterat, 
                                           return_index=True, 
                                           timestamp=None, 
                                           visualize=False)
                                           
            [boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx
             ] = ropf.remove_no_continuous(pcd_in, floor, obstacles, ceiling, 
                                           idx_floor, idx_obs, idx_ceil, 
                                           pm.eps_2, pm.pt, pm.alpha,
                                           pm.voxel_size, pm.zmax,
                                           pm.zmin, pm.dist, pm.init_n, 
                                           vox_idx, idx_clean, 
                                           visualize = False, save = False)
                                          
            boundary_coords = np.array(boundary_pol.boundary.coords)
            # Create the polygon object
            boundary_pol = Polygon(boundary_coords)

            # Define the maximum distance from the contour
            # (in units of your coordinate system)

            # Filter points within the contour distance range
            points_near_boundary = []

            for point in points_2:
                x, y, z = point  
                p = Point(x, y) 

                # Calculate the distance from the point to the polygon outline
                distancia = boundary_pol.boundary.distance(p)

                # If the distance is less than or equal to the
                # maximum distance the point is added
                if distancia <= distancia_max:
                    points_near_boundary.append(point)

            # Convert to a numpy array if necessary
            new_points_with_z = np.round(np.array(points_near_boundary), 2)
                                          
            x = new_points_with_z[:, 0]
            y = new_points_with_z[:, 1]
            
            # Add the first point at the end to close the cycle
            x = np.append(x, x[0])
            y = np.append(y, y[0])
                
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, marker='o', linestyle='-', color='b') 
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()

    
        
        elif np.unique(pcd_in[['id_storey']])[0] == last_storey:
            df_ceiling_last = df[(df['element_type'] == 'ceiling') 
                                 & (df['storey_id'] == last_storey)]
            points_2 = points(df_ceiling_last)

            # Voxelization
            pcd_down_o3d, vox_idx = sb.voxel_downsample(
                array_to_o3d.array_to_o3d_point_cloud(
                    pcd_in.loc[:, 'x':'z'].values), 
                pm.voxel_size, order_by_index=(True))
            
            # Clean exteriors
            pcd_down_clean, idx_clean = clean_exteriors.clean_exteriors(
                pcd_down_o3d, epsilon = pm.eps_1, points_min = pm.pt, 
                return_index=True)
            
            # Extract floor, celling and obstacles
            [floor, obstacles, ceiling, idx_floor, idx_obs, idx_ceil
             ] = efc.extract_floor_ceiling(pcd_down_clean, pm.nt, 
                                           search_param=pm.s_p, 
                                           ransac_th=pm.ransac_th, 
                                           iterat=pm.iterat, 
                                           return_index=True, 
                                           timestamp=None, 
                                           visualize=False)
                                           
            [boundary_pol, floor_plane, ceiling_plane, pcd_in, clean_idx
             ] = rop.remove_no_continuous(pcd_in, floor, obstacles, ceiling, 
                                           idx_floor, idx_obs, idx_ceil, 
                                           pm.eps_2, pm.pt, pm.alpha,
                                           pm.voxel_size, pm.zmax, pm.zmin,
                                           pm.dist, pm.init_n, vox_idx,
                                           idx_clean, visualize = False, 
                                           save = False)
                                          
            boundary_coords = np.array(boundary_pol.boundary.coords)
            # Create the polygon object
            boundary_pol = Polygon(boundary_coords)

            # Define the maximum distance from the contour
            # (in units of your coordinate system)

            # Filter points within the contour distance range
            points_near_boundary = []

            for point in points_2:
                x, y, z = point  
                p = Point(x, y) 

                # Calculate the distance from the point to the polygon outline
                distancia = boundary_pol.boundary.distance(p)

                # If the distance is less than or equal to the maximum 
                # distance, the point is added
                if distancia <= distancia_max:
                    points_near_boundary.append(point)

            # Convert to a numpy array if necessary
            new_points_ceiling = np.round(np.array(points_near_boundary), 2)
                                          
            x = new_points_ceiling[:, 0]
            y = new_points_ceiling[:, 1]
            # Agregar el primer punto al final para cerrar el ciclo
            x = np.append(x, x[0])
            y = np.append(y, y[0])
                
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, marker='o', linestyle='-', color='b') 
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()

    return new_points_with_z, new_points_ceiling




