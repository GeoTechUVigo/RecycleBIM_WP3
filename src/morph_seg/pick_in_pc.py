# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

def dataframe_to_open3d_pcd(dataframe):
    """
    Converts a Pandas DataFrame with 'x', 'y', 'z', and 'id_room' columns
    to an Open3D PointCloud object.

    Parameters
    ----------
    dataframe :pandas.DataFrame
        A DataFrame containing at least the columns 'x', 'y', 'z', and 'id_room'.

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        A PointCloud object from Open3D that contains the 3D points from the 
        DataFrame, with color values corresponding to the normalized 
        'id_room' values.

    """
    points = np.array(dataframe[['x', 'y', 'z']])  
    colors = dataframe['id_room'].astype(float) / dataframe['id_room'].max()  
    colors = plt.cm.tab20(colors)[:, :3]  
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def pick_points(pcd):
    """
    Allows the user to interactively select points from a 3D point cloud 
    using Open3D's visualizer.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The Open3D PointCloud object containing the 3D points. The points
        will be displayed in the viewer for the user to select.

    Returns
    -------
    list of int
        A list of indices representing the selected points in the point cloud. 
        Each index corresponds to a point in the original point cloud.

    """
    print("Press 'Shift + left click' to select points")
    print("Press 'Q' to exit the viewer and continue the script")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  
    vis.destroy_window()
    return vis.get_picked_points()


def select_room(class_pcd_df):   
    """
    Allows the user to select a point from a point cloud and returns the room 
    ID of the selected point.


    Parameters
    ----------
    class_pcd_df : pandas.DataFrame
        A DataFrame containing 3D point cloud data.
        
    Returns
    -------
    point_room : int
        The room ID (`id_room`) of the selected point. This value corresponds
        to the room or category that the user selects in the point cloud.

    """
    pcd = dataframe_to_open3d_pcd(class_pcd_df)
    selected_indices = pick_points(pcd)
    
    point = class_pcd_df.iloc[selected_indices[0]]    
    point_room = int(point['id_room'])    
        
    return point_room
    

