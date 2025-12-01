# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def histogram_scott(pcd_in, num_s, d_m): 
    """
    Segment the initial point cloud by storeys using the frequency histogram 
    in Z as a method.

    Parameters
    ----------
    pcd_in : DataFrame
        Point cloud to be segmented.
    num_s : int
        Number of storeys.
    d_m : float
        Maximum distance between the ceiling of one storey and the floor of 
        the next storey.

    Returns
    -------
    heights : list of floats
        List with the heights at which the cut must be made.

    """
     
    datos = np.array(pcd_in)
    
    # Calculate number of bins with Scott's formula
    n_bins = int((np.max(datos) - np.min(datos)) / (
        3.5 * np.std(datos) / len(datos)**(1/3)))
    
    # Maximum and minimum of Z-values
    max_z = pcd_in.loc[:, 'z'].max()
    min_z = pcd_in.loc[:, 'z'].min()
    
    # Z-histogram of point cloud data
    pcd_zhist = np.histogram(pcd_in.loc[:, 'z'].values, bins=n_bins, 
                             range=(min_z,max_z))
    
    # Plot of histogram
    plt.hist(pcd_in.loc[:, 'z'].values, bins=n_bins, color='#87CEFA')
    plt.xlabel('Z (m)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Higher frequencies
    hgt_val_pcd = pcd_zhist[1]  
    index_order = np.argsort(pcd_zhist[0])[::-1]  
    # frec_order = np.array(pcd_zhist[0])[index_order]
    bins_order = np.array(pcd_zhist[1])[index_order]
    
    k = num_s * 2
    
    # Select the highest frequencies
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
            heights.append(hgt_val_pcd[np.where(np.isclose(
                hgt_val_pcd, indexes[len(indexes)-1]))[0][0] + 1])
    
    return heights
    
    
    
    