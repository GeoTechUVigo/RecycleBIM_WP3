# -*- coding: utf-8 -*-

import pandas as pd
import laspy as lp
import numpy as np


def load_pointcloud(name_cloud, folder_cloud, pcd_file, visualize):
    if folder_cloud == "\\cs1" :
        pcd_in = pd.read_csv(pcd_file, sep=' ')   

        
    elif folder_cloud == "\\cs2":
        if name_cloud.split("_")[0] == "\\Initial":
            pcd_in = pd.read_csv(pcd_file, sep=' ', usecols=[0,1,2,3], names=['x','y','z','time'])
        
        elif name_cloud.split("_")[0] != "\\Initial":
            pcd_in = pd.read_csv(pcd_file, sep=' ')   
    
    elif folder_cloud == "\\cs3":
        if name_cloud.split("_")[0] == "\\Initial":
            if name_cloud.split(".")[-1] == "las":
                las_file = lp.read(pcd_file)
                file_fields = []
                for dim in las_file.point_format:
                    file_fields.append(dim.name)
                    
                print(file_fields)            
                pcd_in = pd.DataFrame(np.vstack((las_file.x, las_file.y, las_file.z, 
                                                  las_file.intensity, las_file.red, 
                                                  las_file.green, las_file.blue)).transpose(),
                                      columns=['x','y','z','intensity','red','green','blue'])    
     
            elif name_cloud.split(".")[-1] == "txt":
                pcd_in = pd.read_csv(pcd_file, sep=' ',usecols=[0,1,2], names=['x','y','z'])   
        elif name_cloud.split("_")[0] != "\\Initial":
            pcd_in = pd.read_csv(pcd_file, sep=' ')  
            
    elif folder_cloud == "\\cs4":
        if name_cloud.split("_")[0] == "\\Initial":
            if name_cloud.split(".")[-1] == "las":
                las_file = lp.read(pcd_file)
                file_fields = []
                for dim in las_file.point_format:
                    file_fields.append(dim.name)
                    
                print(file_fields)            
                pcd_in = pd.DataFrame(np.vstack((las_file.x, las_file.y, las_file.z, 
                                                  las_file.intensity, las_file.red, 
                                                  las_file.green, las_file.blue)).transpose(),
                                      columns=['x','y','z','intensity','red','green','blue'])    
     
            elif name_cloud.split(".")[-1] == "txt":
                pcd_in = pd.read_csv(pcd_file, sep=' ',usecols=[0,1,2], names=['x','y','z'])  
        elif name_cloud.split("_")[0] != "\\Initial":
            pcd_in = pd.read_csv(pcd_file, sep=' ')            
    

        
    return pcd_in