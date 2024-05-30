# -*- coding: utf-8 -*-

# Packages (Python)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def histogram_scott(pcd_in, num_s, d_m, name_f):
    # Read file
    datos = np.array(pcd_in)
    
    # Calculate number of bins with Scott's formula
    n_bins = int((np.max(datos) - np.min(datos)) / (3.5 * np.std(datos) / len(datos)**(1/3)))
    
    # Maximum and minimum of Z-values
    max_z = pcd_in.loc[:, 'z'].max()
    min_z = pcd_in.loc[:, 'z'].min()
    
    # Z-histogram of point cloud data
    pcd_zhist = np.histogram(pcd_in.loc[:, 'z'].values, bins=n_bins, range=(min_z,max_z))
    
    # Plot of histogram
    plt.hist(pcd_in.loc[:, 'z'].values, bins=n_bins, color='#87CEFA')
    plt.xlabel('Z (m)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Higher frequencies
    hgt_val_pcd = pcd_zhist[1]  
    index_order = np.argsort(pcd_zhist[0])[::-1]  
    frec_order = np.array(pcd_zhist[0])[index_order]
    bins_order = np.array(pcd_zhist[1])[index_order]
    
    # If there is one floor it will have two clusters (floor-ceiling), if it has 
    # two floors it will have 4 clusters (floor-ceiling-floor-ceiling), if it has 
    # three floors it will have 6 clusters (floor-ceiling-floor-ceiling-floor-ceiling)...
    k = num_s * 2
    
    # Select the highest frequencies
    top_frec = frec_order[:k]
    top_bins = bins_order[:k]
    indexes = np.sort(top_bins)
    
    heights = []
    for i in range(0, len(indexes)): 
        if i == 0:
            heights.append(hgt_val_pcd[np.where(np.isclose(hgt_val_pcd, 
                                                           indexes[0]))[0][0]])
        
        elif i > 0 and i < len(indexes)-1:
            if indexes[i+1] - indexes[i] < d_m:
                heights.append((indexes[i+1] + indexes[i])/2)    
        
        elif i == len(indexes)-1:
            heights.append(hgt_val_pcd[np.where(np.isclose(hgt_val_pcd,
                                                           indexes[len(indexes)-1]))[0][0] + 1])
            
    # Save results
    for i in range(0, len(heights)-1):
        storey = pcd_in[(pcd_in['z'] >= heights[i]) & (pcd_in['z'] <= heights[i+1])]
        storey.to_csv(name_f + str(i) + ".txt", sep=' ', header=True, index=False)
        
    
    