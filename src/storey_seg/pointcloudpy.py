# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import laspy

from src.storey_seg.trajectorypy import Trajectory
from src.utils.functionspy import txt_to_df


class PointCloudPy(object):  
    """ 
    This class stores pointcloud data and provides methods to handle stored
    date. Also allows to work with information of trajectory attached to the 
    point cloud if it is avaliable.
    
    Attributes
    ---------
    id_fields : list of str   
        List of field names of loaded point cloud          
    n_pts : int
        Number of points        
    offset_time : float
        Time value (in seconds) used to scale timestamp        
    pcd : DataFrame
        Loaded pointcloud data    
    timestamp: Series
        Timestamp of the stored points    
    trajectory: Trajectory object
        Trajectory associated with the input pointcloud         
    col_pcd (optional) : list of strings
        List of field names of input point cloud      
    col_traj (optional) : list of str
        List of field names of input trajectory 
        
    Methods
    -------    
    check_times(scaled=False)    
        Checks if the timestamps of the point cloud and the trajectory
        correspond temporarily
        
    time_segmentation(time_seg, name=None, traj=None)    
        Performs a classification of the instance data according to the 
        information of the argument time_seg        
        
    get_data(columns=None)    
        Allows to get specific data from the loaded point cloud
    """
    
    # Minimum information required to create an instance
    required_cols = set(('x', 'y', 'z'))
    
    def __init__(self, pcd, **kwargs):                
        self.__col_pcd =kwargs.get('col_pcd', None)
        self.pcd = pcd
        self.__col_traj = kwargs.get('col_traj', None)
        self.trajectory = kwargs.get('traj', None)
        self.__offset_time = 0
        self.__filename = None
       
        
    @property
    def id_fields(self):
        """ List of str: Name labels of loaded point cloud"""
        
        return self.__pcd.columns
    
    @property
    def n_pts(self):
        """int: Number of points of loaded point cloud"""
        
        return self.__pcd.shape[0]
   
    @property
    def offset_time(self):
        """float: Time value (in seconds used to scale timestamps)"""
        
        return self.__offset_time
    
    @property
    def pcd(self):
        """DataFrame: Loaded point cloud """
        
        return self.__pcd
            
    @property
    def timestamp(self):
        """Series: Timestamp of loaded point clouds if it is avaliable"""
        
        if 'time' in self.id_fields:
            return self.pcd.loc[:, 'time']
        else:
            pass
    
    @property
    def trajectory(self):
        """Trajectory object: Instance of Trajectory class which contains 
        information about trajectory attached to loaded point cloud"""
        
        if self.__traj is None:
            pass
        return self.__traj
             
    @pcd.setter
    def pcd(self, pcd_input):
        if isinstance(pcd_input, pd.DataFrame):
            columns = set(pcd_input.columns)
            if self.required_cols.issubset(columns):
                self.__pcd = pcd_input
                cols = list(self.__pcd.columns)
                # Time must be the first column
                if 'time' in cols and cols[0]!='time':
                    columns_reorded = cols[:]
                    columns_reorded.remove('time')
                    columns_reorded.insert(0,'time')
                    self.__pcd = self.__pcd[columns_reorded]
            else:
                pass
                """
                self.logger.error('Input DataFrame must contains at least '
                                  '[\'x\',\'y\',\'z\'] labels')
                """
                
        elif isinstance(pcd_input, np.ndarray):
            if self.__id_fields is not None:
                set_col_pcd = set(self.__col_pcd)
                if (pcd_input.shape[1] == len(self.__col_pcd) and 
                self.required_cols.issubset(set_col_pcd)):
                    self.__pcd = pd.DataFrame(pcd_input, columns=self.__col_pcd)
                    
                    #Time must be the first column
                    if 'time' in cols and cols[0]!='time':
                        columns_reorded = cols[:]
                        columns_reorded.remove('time')
                        columns_reorded.insert(0,'time')
                        self.__pcd = self.__pcd[columns_reorded]
                else:
                    pass
                    
        elif isinstance(pcd_input, str):           
            # File extension
            ext = pcd_input.split('.')[-1]
            if ext=='txt':
                pcd_df, col_pcd = txt_to_df(pcd_input, self.__col_pcd)
                if not isinstance(pcd_df, pd.DataFrame):
                    pass
                    
                elif self.required_cols.issubset(set(pcd_df.columns)):
                        self.__pcd = pcd_df
                        self.__filename = pcd_input
                        cols = list(self.__pcd.columns)
                        # Time must be the first column
                        if 'time' in cols and cols[0]!='time':
                            columns_reorded = cols[:]
                            columns_reorded.remove('time')
                            columns_reorded.insert(0,'time')
                            self.__pcd = self.__pcd[columns_reorded]
                            
                        if self.id_fields is None:
                            
                            self.__col_pcd = pcd_df.columns

                else:
                    self.logger.error('[\'x\',\'y\',\'z\'] labels are neccessary')
                    
            elif ext=='las':
                pcd_df = self.__load_las(pcd_input)
                
                if not isinstance(pcd_df, pd.DataFrame):
                    self.logger.error('Error processing file')
                elif self.required_cols.issubset(set(pcd_df.columns)):
                        self.__pcd = pcd_df
                        self.__filename = pcd_input
                        if self.id_fields is None:
                            self.__col_pcd = pcd_df.columns
                else:
                    pass  
            
            else:
                pass
        else:
            self._traj = None
            
    @trajectory.setter
    def trajectory(self, traj):
          
        if traj is None:
            self.__traj = None
        elif isinstance(traj, (str, pd.DataFrame, np.ndarray)):
            try:
                self.__traj = Trajectory(traj, self.__col_traj)
            except:
                self.__traj = None
        elif isinstance(traj, Trajectory):
            self.__traj = traj
        else:
            pass
               
    @offset_time.setter
    def offset_time(self, offset):
        if 'time' in self.id_fields:
            try:
                float(offset)
                self.__offset_time = offset
                self.__pcd.loc[:,'time'] = self.__pcd.loc[:, 'time'] - offset
            except:
                pass
        else:
            pass
    
    
   
    def __load_las(self, file_name):
        """
        Import XYZ and time data from las file to DataFrame

        Parameters
        ----------
        file_name : str
            Path to las file.

        Returns
        -------
        input_df : dataframe 
            Is the file in .las format transformed into a dataframe.

        """
        
        las_file = laspy.read(file_name)
        file_fields = []
        for dim in las_file.point_format:
            file_fields.append(dim.name)
                                
        # List of point formats supporting gps_time 
        gps_time_formats = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Point format of las file
        file_las_format = las_file.header.data_format_id
        
        if file_las_format in gps_time_formats:
            input_df = pd.DataFrame({'time': las_file.gps_time, 
                                     'x': las_file.x, 
                                     'y': las_file.y,
                                     'z': las_file.z})
        
        else:
            return -1
        
        return input_df
    
    
    def check_times(self, scaled=False):
        """
        This method checks if timestamps of pointcloud and trajectory are
        matching.

        Parameters
        ----------
        scaled : bool, optional
            Determines if time scalation must be carried out.
            The default is False.

        Returns
        -------
        bool
            True if timestamps of pointcloud and trajectory are
            matching or False if not.

        """
        
        # Check avaliability os time data in both pointcloud and trajectory
        if ('time'in self.__col_pcd) and ('time' in self.__traj.id_fields):
            
            res = self.__traj.time_period
            # Compare maximun and minimun value of time of pcd and trajectory
            diff_min = np.absolute(self.__pcd.time.min() - 
                                   self.__traj.timestamp.min())
            diff_max = np.absolute(self.__pcd.time.max() - 
                                   self.__traj.timestamp.max())
            # Checks if difference is smaller than time period
            if (diff_min <= res) and (diff_max <= res):
                min_time_pcd = self.__pcd.time.min()
                min_time_traj = self.__traj.timestamp.min()
                
                # If scaled=True time is scaled at 0.0 seconds 
                if scaled:
                    # Relative time
                    self.offset_time = min_time_pcd
                    self.__traj.offset_time = min_time_traj
                return True
            else:
                return False
           
        else:
            return False
        
    def get_data(self, columns=None):
        """
        This method returns pcd DataFrame. If columns are passed, only 
        the specified columns are returned.

        Parameters
        ----------
        columns : str, optional
            Nme of columns. The default is None.

        Returns
        -------
        DataFrame
            DataFrame of pointcloud.

        """
        
        if columns is None:
            return self.__pcd
        else:
            data_cols = self.pcd.columns
            cols = [columns[i] for i in range(len(columns)) if 
                    columns[i] in data_cols]
            if len(cols):
                if len(cols) < len(columns):
                    """
                    self.logger.error('Some specified columns are not in point'
                                      'loud data')
                    """
                return self.__pcd.loc[:, cols]
            else:
                pass

    
    def time_segmentation(self, time_seg, name=None, traj=None):
        """
        This method classifies input data according to timestamp from the 
        time_sec argument information

        Parameters
        ----------
        time_seg : pandas Series object, dict
            When time_seg is a pandas Series, if traj is None Series data 
            is added to pcd DataFrame else Series data is added to 
            trajectory. 
            
            If time_seg is of type dict a pandas Series is added to pcd
            data with the column label assigned to name argument and labels
            of data are setted according to the info in dictionary:
                
                key : int
                    label_id                
                value : list of tuples
                    list containing tuples of time intervales.
        name : str, optional
            Name of column to be added.
            The default is None.
        traj : bool, optional
            If true, time-based segmentation method is invoked for 
            trajectory instance.
            The default is None.

        Returns
        -------
        None.

        """
      
        if isinstance(time_seg, pd.Series):
            if traj is None:
                self.__pcd = self.__pcd.assign(time_seg)
            else:
                self.__traj.time_segmentation(time_seg)
                
        elif isinstance(time_seg, dict):
            if name is not None:
                
                if not name in self.__pcd.columns:
                    self.__pcd[name] = -1
                for lbl, intervals in time_seg.items():
                     for interval in intervals:
                         idx_pcd = self.__pcd[((self.__pcd.time>interval[0]) & 
                                                 (self.__pcd.time<interval[1]))].index
                         self.__pcd.loc[idx_pcd,name] = lbl
                self.__pcd[name].astype(pd.Int64Dtype())
                if traj is not None:             
                    self.__traj.time_segmentation(time_seg, name)
            else:
                pass

