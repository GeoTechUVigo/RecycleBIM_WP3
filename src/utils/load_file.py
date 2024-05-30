# -*- coding: utf-8 -*-

import pandas as pd
import laspy as lp
import numpy as np

def load_file(path_file, header=True):
    """
    Read the file where the point cloud is located.

    Parameters
    ----------
    path_file : str
        Path where the point cloud is located.
    header : bool, optional
        Indicates whether the file has a header or not. The default is True.

    Returns
    -------
    pcd : DataFrame
        Is the pointcloud.

    """
    
    if path_file.split(".")[-1] == "txt":
        if header:
            pcd = pd.read_csv(path_file, sep=' ')
        else:
            # If there is no header, prompts the user to enter the column names
            num_columns = int(input("Number of columns: "))
            name_columns = [input(f"Enter the column name {i+1}: ") for i in range(num_columns)]
            pcd = pd.read_csv(path_file, sep=' ', header=None, names=name_columns)
    

    elif path_file.split(".")[-1] == "las":
        if header:
            pcd = lp.read(path_file)
            pcd_in = pd.DataFrame(np.vstack((pcd.x, pcd.y, pcd.z)).transpose(),
                                  columns=['x','y','z'])    
        
        else:
            num_columns = int(input("Number of columns: "))
            name_columns = [input(f"Enter the column name {i+1}: ") for i in range(num_columns)]
            pcd = lp.read(path_file, header=None, names=name_columns)

    return pcd


def laspy_header(path):
    pcd = lp.read(path)
    file_fields = []
    for dim in pcd.point_format:
        file_fields.append(dim.name)        
    print(file_fields)            
    
    return file_fields


