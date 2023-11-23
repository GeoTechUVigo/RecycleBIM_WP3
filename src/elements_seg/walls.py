# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from src.utils import functions_walls


def walls(class_pcd_df, min_ratio, threshold, init_n, h):
    """
    It detects walls.

    Parameters
    ----------
    class_pcd_df : DataFrame
        Is the pointcloud.
    min_ratio : float
        The minimum left points ratio to end the Detection.
    threshold : float
        RANSAC threshold in meters.
    init_n : int
        Number of iteration.
    h : float
        To check if the difference between the lowest and highest Z values is
        greater. 
    Returns
    -------
    segmented_walls_c_f : DataFrame
        DataFrame of walls.
    class_pcd_df : DataFrame
        Complete DataFrame.

    """
    class_2 = class_pcd_df[class_pcd_df['id_element'] == 2]
    
    # Group the DataFrame by the 'id_room' column
    grupos_por_id_room = class_2.groupby('id_room')

    # Create a dictionary to store the separate DataFrames
    dataframes_por_id_room = {}
    segmented_walls_c = []
    # Iterate through groups
    for id_room, grupo in grupos_por_id_room:
        # Save each group to the dictionary
        dataframes_por_id_room[id_room] = grupo.copy()  # We use copy() to avoid modifying the original DataFrame

        pcd_np = np.array(dataframes_por_id_room[id_room][['x', 'y', 'z']] )   

        results = functions_walls.DetectMultiPlanes(pcd_np, min_ratio, threshold, iterations=2000)

        plans = []

        for idx, (plane_equation, plane_points) in enumerate(results):         
            # Create a DataFrame for the current clustered wall
            plane_df = pd.DataFrame(data=plane_points, columns=["x", "y", "z"])
            
            # Append the DataFrame to the plans list
            plans.append(plane_df)

        # Number of plans
        num_plans = len(plans)

        segmented_walls = []
       
        for idx, plan_df in enumerate(plans):
            # Calculate the difference between the lowest and highest Z values
            z_diff = plan_df['z'].max() - plan_df['z'].min()

            # Check if the difference is greater than a parameter
            if z_diff > h:
                # Save the DataFrame with a difference greater than h
                segmented_walls.append(plan_df)
                segmented_walls_c.append(plan_df)


    segmented_walls_c_f  = pd.concat(segmented_walls_c, ignore_index=True)       
        
    # Let's assume df1 and df2 are our DataFrames
    df1 = segmented_walls_c_f
    df2 = class_2

    # Add a column 'original_index_df2' to df2 containing the original indexes
    df2['index'] = df2.index

    # Performs the merge based on the x, y, and z columns
    merged_df = df2.merge(df1, on=['x', 'y', 'z'], how='inner', suffixes=('_df2', '_df1'))

    # merged_df now contains the points from df2 that are also in df1 along with their original df2 indices.

    walls = merged_df.set_index('index')    
    walls.index.name = None        

    class_pcd_df['walls'] = 0   
    class_pcd_df.loc[walls.index, 'walls'] = 1
    
    return segmented_walls_c_f, class_pcd_df
            
             

