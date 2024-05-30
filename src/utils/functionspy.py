# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd


def txt_to_df(filename, columns=None):
    """
    This function generate a DataFrame structure from a input .txt file. 
        
    Arguments
    ---------    
    filename: str         
        Absolute path to file    
    columns: list of str    
        List that contains field names to be considered
        
    """
    
    # Check if the point cloud file contains header
    try:
        with open(filename, 'r') as file_txt:
            first_ln = file_txt.readline()
            header_bool = False
    except:
        return -1, None
    
    # Remove non alphanumeric characters from first line file
    pattern = re.compile('[\W_]+', re.UNICODE)
    first_ln = pattern.sub(' ', first_ln).strip()

    first_ln_tp = tuple(first_ln.split(' '))
    first_ln_lower = first_ln.lower().split(' ')
    
    # If there are a non float value in the first line it is the header
    for col in first_ln_tp:
        try:
            float(col)
        except:
            header_bool = True
            first_ln = first_ln.lower()
            break
    
    # Process file with header
    if header_bool:
        # Load input data point cloud
        if columns is None:
            try:
                data_df = pd.read_csv(filename, sep=" ", header=0,
                                       names=first_ln_lower)
            except:
                return -1, list(first_ln_tp)
        else:
            # Check whether colums are in header
            col_set = set(columns)
            hdr_set = set(first_ln_tp)
            if col_set.issubset(hdr_set):
                try:
                    data_df = pd.read_csv(filename, sep =" ", header=0, 
                                           names=first_ln_tp, 
                                           usecols=tuple(columns)) 
                except:
                    return -1, list(first_ln_tp)
        return data_df, list(first_ln_tp)
    
    # Process file without header
    else:
        if columns is not None:
            try:
                data_df = pd.read_csv(filename, sep=" ", header=None, 
                                       names=columns, usecols=columns) 
            except:
                return -1, columns
        else:      
            try:
                # Load input data 
                data_df = pd.read_csv(filename, sep=" ", header=None, 
                                     names=['time', 'x', 'y', 'z'],
                                     usecols=['time', 'x', 'y', 'z'])
            except:
                return -1, columns
            
        return data_df, columns


def v_distance_xy(pts):
    """ 
    Return a distance vector of consecutive 2d positions from input data
    
    """
     
    if isinstance(pts, pd.DataFrame):
        pos = pts.iloc[:, :2].values
    elif isinstance(pts, list):
        pos = np.asarray(pts)[:, :2]
    elif isinstance(pts, np.ndarray):
        pos = pts[:, :2]
    
    dist_pos = np.zeros(pos.shape[0])
    dist_pos[1:] = np.sqrt(np.power(np.diff(pos[:, 0]), 2.0) + 
                           np.power(np.diff(pos[:, 1]), 2.0))
            
    vect_dist = np.cumsum(dist_pos)
    
    return vect_dist 