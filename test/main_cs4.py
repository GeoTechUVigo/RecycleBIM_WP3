# -*- coding: utf-8 -*-
path = "C:\\Users\\rosam\OneDrive - Universidade de Vigo\Escritorio\\Code_v1_oct_23\data"
folder_cloud = "\\cs4"
name_cloud = "\\Initial_cs4_005.txt"
folder_results = "C:\\Users\\rosam\\OneDrive - Universidade de Vigo\Escritorio\\Code_v1_oct_23\\Results"
pcd_file = path + folder_cloud + name_cloud
name_f = "_storey_"
#%%
# Packages (Python)
import laspy as lp
import pandas as pd
import numpy as np
import time
import os

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import open3d as o3d

# Packages 
from src.utils import load_pointcloud
from src.storey_seg import histogram


#%%
# Parameters
n_bins = 18 
dist_b = 3

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
with open(folder_results + folder_cloud + '\\time_seg' + name_f + ".txt", 'w') as file:
    # Load pointcloud
    pcd_in = load_pointcloud.load_pointcloud(name_cloud, folder_cloud, pcd_file, visualize=True)  
    
    
    # Selecting the suitable n_bins visulization
    # for i in range(5, 20):
    #     n_bins = i
    #     hgt_val_pcd, indexes = histogram.histogram(pcd_in, n_bins, dist)
    #     print(i)
    
  
    
    hgt_val_pcd, indexes = histogram.histogram(pcd_in, n_bins, dist_b)


    # Histogram peaks
    no_of_storeys = max(range(0, len(indexes)))

    # Checking the cutting values 
    for i in range(0, len(indexes)):
        if i == 0:
            print(hgt_val_pcd[0:][indexes[i]])
        elif i == no_of_storeys: 
            print(hgt_val_pcd[1:][indexes[i]])
        else:
            cp = ((hgt_val_pcd[0:][indexes[i]]) + (hgt_val_pcd[1:][indexes[i]])) / 2
           
            print(cp)   

    # Selecting the cutting points 
    heights = []

    for i in range(len(indexes)):
        if i == 0:
            heights.append(hgt_val_pcd[0:][indexes[i]])
            
        elif i == no_of_storeys : 
            heights.append(hgt_val_pcd[1:][indexes[i]])
            
        else:
            cp = ((hgt_val_pcd[0:][indexes[i]]) + (hgt_val_pcd[1:][indexes[i]])) / 2
            
            heights.append(cp)

    # Initializes a list to store the resulting DataFrames
    storeys = []

    for i in range(len(heights) - 1):
        # Filters points between specific heights
        storey_i = pcd_in[(pcd_in['z'] > heights[i]) & (pcd_in['z'] < heights[i + 1])]
        
        # Assign the corresponding storey_id
        storey_i['id_storey'] = i
        
        # Save the storey data to the specified file path
        storey_i.to_csv(folder_results + folder_cloud + "\\" + folder_cloud  +
                        name_f + str(i) + ".txt", sep=' ', header=True, index=False)


    # Time
    time_1 = time.time() - ini_time
    file.write(f"The point cloud takes {time_1} seconds to segment into storeys. \n") 
    
end_time = time.time()