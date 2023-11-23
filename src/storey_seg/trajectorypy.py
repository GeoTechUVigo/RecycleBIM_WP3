# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import ceil
from src.utils.functionspy import txt_to_df, v_distance_xy


class Trajectory(object):
    """ 
    This class stores trajectory information and provides methods to handle
    trajectory data.
        
    Attributes
    ----------    
    id_fields: list of str    
         List of field names of loaded trajectory         
    offset_time: float    
        time value (in seconds) used to scale timestamp    
    poses: DataFrame    
        Trajectory XYZ coordinates in DataFrame with column labels 'x', 
        'y' and 'z'.   
    timestamp: Series    
        Timestamp of each trajectory position    
    time_period: scalar    
        Acquisition period of trajectory data        
    traj: DataFrame    
        All loaded trajectory data such as XYZ coordinates, timestamp, etc

    Methods
    -------
    show_2dtrajectory(dss=None)    
        2D visualization of trajectory        
    time_segmentation(time_seg, name=None)    
        Performs a classification of the instance data according to the 
        information of the argument time_seg        
    trajectory_subsampling(self, dss, dim=2)    
        Performance trajectory subsampling according to the distance of dss
        argument.
"""
    # Minimum information required to create an instance
    required_cols = set(('x', 'y', 'z'))
    
    def __init__(self, traj, columns=None):
        """  
        Arguments
        ---------        
        traj : string, DataFrame, ndarray        
            Trajectory data. It can be in file, a DataFrame or a ndarray.        
        columns: list of str        
            List contains data identifiers.            
        """
        
        self.__id_fields = columns
        self.trajectory = traj
        
        self.__offset_time = 0
        self.__id_fields_file = None
        # Keys are resoluton and values are array positions
        self.__traj_sub_dict = {}
          
    
    @property
    def id_fields(self):
        """List of str: Name labels of loaded trajectory"""
        return self.__id_fields    
    
    @property
    def offset_time(self):
        """float: Time value (in seconds used to scale timestamps) """
        return self.__offset_time
    
    @property
    def poses(self):
        """DataFrame: XYZ coordinates of trajectory positions arranged in 
                      referenced columns by 'x', 'y', 'z' labels"""
        return self.__traj.loc[:, 'x':'z']
    
    @property
    def time_period(self):
        """float: period of acquisition obtained from trajectory timestamp"""
        if 'time' in self.trajectory.columns:
            return pd.Series.mean(self.trajectory.time.diff(), skipna=True)
        else:
            self.logger.error('Trajectory data does not contains time data') 
    
    @property
    def timestamp(self):
        """Series: Timestamp of loaded point clouds if it is avaliable """
        if 'time' in self.trajectory.columns:
            return self.trajectory.loc[:, 'time']
        else:
            self.logger.error('Trajectory data does not contains time data')
            
    @property
    def trajectory(self):
        """DataFrame: Loaded trajectory data"""
        return self.__traj        

    @offset_time.setter
    def offset_time(self, offset):
        if 'time' in self.id_fields:
            try:
                float(offset)
                self.__offset_time = offset
                self.__traj.loc[:,'time'] = self.__traj.loc[:, 'time'] - offset
            except:
                pass
                
        else:
            pass
            
            
    @trajectory.setter
    def trajectory(self, traj_input):
        if isinstance(traj_input, pd.DataFrame):
            columns = set(traj_input.columns)
            if self.required_cols.issubset(columns):
                self.__traj = traj_input
          
                cols = list(self.__traj.columns)
                # Time must be the first column
                if 'time' in cols and cols[0]!='time':
                    columns_reorded = cols[:]
                    columns_reorded.remove('time')
                    columns_reorded.insert(0,'time')
                    self.__traj = self.__traj[columns_reorded]
            else:
                pass
                
        elif isinstance(traj_input, np.ndarray):
            
            if self.__id_fields is not None:
                set_id_fields = set(self.__id_fields)
                if (traj_input.shape[1] == len(self.__id_fields) and 
                self.required_cols.issubset(set_id_fields)):
                    self.__traj = pd.DataFrame(traj_input, columns=self.__id_fields)
                    cols = list(self.__traj_df.columns)
                    # Time must be the first column
                    if 'time' in cols and cols[0]!='time':
                        columns_reorded = cols[:]
                        columns_reorded.remove('time')
                        columns_reorded.insert(0,'time')
                        self.__traj = self.__traj[columns_reorded]
                else:
                    pass
                    
                    
        elif isinstance(traj_input, str):
            # File extension
            ext = traj_input.split('.')[-1]
            if ext=='txt':
                traj_df, id_fields = txt_to_df(traj_input, self.__id_fields)
                if not isinstance(traj_df, pd.DataFrame):
                    self.__traj = None
                    #self.logger.error('Error procesando fichero de texto')
                 
                elif self.required_cols.issubset(set(traj_df.columns)):
                    self.__traj = traj_df
                    cols = list(traj_df.columns)
                    # Time must be the first column
                    if 'time' in cols and cols[0]!='time':
                        columns_reorded = cols[:]
                        columns_reorded.remove('time')
                        columns_reorded.insert(0,'time')
                        self.__traj = self.__traj[columns_reorded]
                        
                    if self.id_fields is None:
                        self.id_fields = traj_df.columns
                    else:
                        pass

                else:
                    pass

            else:
                pass
        else:
            self.__traj = None
            
    
    def time_segmentation(self, time_seg, name=None):
        """
        This method classifies loaded trajectory data according to 
        timestamp from the time_sec argument information
        
        Arguments
        ---------
        
        time_seg: pandas Series object, dict
        
            When time_seg is a pandas Series, it is added to trajectory 
            DataFrame. 
            
            If time_seg is of type dict a pandas Series is added to
            trajectory DataFrame with the column label assigned to name 
            argument and labels of data are setted according to the info 
            in dictionary with the following structure:
                
                key : int                    
                    Identifiers of each storey
                
                value : list of tuple                    
                    Acquisition time intervals of each storey
                
                # Example of the structure of dict time_seg
                {0: [(0.0, 5.0),(10.0, 20.0)], 1:[(5.1, 9.9)]}
                
        name: str, optional
        
            Name of column to be added
            
        """
        if isinstance(time_seg, pd.Series):
            self.__traj = self.__traj.assign(time_seg)
        elif isinstance(time_seg, dict):
            if name is not None:
                if not name in self.__traj.columns:
                    self.__traj[name] = -1
                for lbl, intervals in time_seg.items():
                    for interval in intervals:
                        idx_traj = self.__traj[((self.__traj.time>interval[0]) & 
                                                (self.__traj.time<interval[1]))].index
                        self.__traj.loc[idx_traj,name] = lbl
                self.__traj[name].astype(pd.Int64Dtype())
            else:
                pass

    
    def trajectory_subsampling(self, dss, dim=2):
        """
        Performance trajectory subsampling according to the distance 
        of dss argument.
           
        Arguments
        ---------        
        dss: float, optional         
            Subsampling distance
            
        dim: integer, default=2        
            Space dimension to calculate distance. Only 2 and 3 dimension 
            are supported
        """
        try:
            pos_traj = self.__traj.loc[self.__traj_sub_dict[dss], 'time':'z']
        except:
            # Distance in XY-plane
            if dim == 2:
                pos = self.__traj.loc[:, 'x':'y'].values
                vect_dist = v_distance_xy(pos)
            # Distance in 3D space   
            elif dim == 3:
                 pos = self.__traj.loc[:, 'x':'z'].values
             
            else:
                pass
                return -1
                
            # Number of segments
            n_seg = ceil(vect_dist[-1] / dss)
            # Index positions of reference positions
            idx_ref_pos = []
            
            for i in range(n_seg):
                # Relative distance
                rel_dist = i * dss
                v_rel_dist = vect_dist - rel_dist
                # Negative values to inf values
                idx_rel_dist = np.where(v_rel_dist>=0.0, v_rel_dist, np.inf)
                # The minimun values correspont to nearest position to rel_dist
                idx_pos = np.argmin(idx_rel_dist)
                # Add to idexes of reference positions list
                idx_ref_pos.append(idx_pos)
            
            self.__traj_sub_dict[dss] = self.trajectory.index[idx_ref_pos]
            pos_traj = self.trajectory.loc[self.__traj_sub_dict[dss],'time':'z']
        
        return pos_traj
    
