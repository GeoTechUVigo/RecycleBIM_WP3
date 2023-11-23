# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from math import floor, ceil
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift
from src.storey_seg.trajectorypy import Trajectory

# Retrieve top level logger
logger = logging.getLogger('inmena_application.histdivision')

class StoreyDivision(object):
    """ 
    This class implements a histogram-based method to identify floor levels
    in a point cloud. First, a histogram is generate from the Z coordinate
    of the trajectory positions and a peaks-finding algorithm is used to 
    detect peaks in histogram so that each peak corresponds to one level.If
    more than one peak is detected a similar histogram is generated from 
    pointcloud data. Merging both information ceiling and floor height are
    calculated for each floor level.
        
    Attributes
    ----------    
    ceils_hgt: dict    
        Height of the ceilings of each storey    
    floor_hgt: dict    
        Height of the floors of each storey        
    floors_time: dict    
        Acquisition time intervals of storeys        
    if_floors: list    
        Identifiers of storeys        
    trajectory: str        
        Instance of Trajectory class containing input trajectory    
    pcd: str        
        Loaded pointcloud data        
    traj_hgt_pks: list of float    
        Height values of peaks detected in the trajectory z-histogram
    
    Methods
    -------
    get_floors_intervals():        
        Generate a dictionary with acquisition time intervals.
    
    process():        
        Invokes and controls the necessary methods to determine the 
        different floor levels the in point cloud
        
    show_pcd_hist():        
         Visualization of the point cloud z-histogram
        
    show_traj_hist():        
         Visualization of the trajectory z-histogram
            
    """
    # Minimum information required to create an instance
    required_cols = set(('z', 'time'))
    def __init__(self, traj, pcd, bin_traj, bin_pcd, dss):
        """
        Note:
            Initialization with arrays requires data structured in matrix with
            four columns corresponding to timestamp and XYZ coordinates.
            
            Initialization with DataFrame only requires columns containing 
            the data of timestamp and Z coordinate labeled as 'time' and 'z'.

        Parameters
        ----------
        traj : ndarray, DataFrame, Trajectory object            
            Trajectory associated with the input pointcloud.
        pcd : DataFrame, ndarray        
            Input pointcloud data.
        bin_traj : float            
            Bin width  to compute z-histogram of trajectory.
        bin_pcd : float        
            Bin width to compute z-histogram of point cloud.
        dss : float, optional        
            Distance (in meters) to trajectory subsampling.
            The default is 0.1.

        Returns
        -------
        None.

        """
        
        # Logger for inmaena application
        self.logger = logging.getLogger('inmena_application.histdivision.HistDivision')
        self.logger.info('creating a instance of HistDivision')
        
        # Subsampled trajectory
        self.__traj_ss = None
        
        self.__traj_dss = dss
        self.trajectory = traj
        self.pcd = pcd
        self.bin_traj = bin_traj
        self.bin_pcd = bin_pcd
        self.__id_floors = []
        self.__traj_hgt_pks = []
        self.__traj_zhist = None
        self._pcd_zhist = None
     
        self.__floors_hgt = {}
        self.__ceils_hgt = {}
        self.__floors_time = {}
        
    
    @property
    def ceils_hgt(self):
        """
        dict : Height of the ceilings of each storey. Dictionary has the 
              following structure:
                  keys : int
                        Identifiers of storeys: 0, 1, ...
                  values : float
                        Height (in meters) in the reference coordinates system 
                        of the point cloud
                  
        """
        return self.__ceils_hgt
    
    @property
    def floors_hgt(self):
        """
        dict : Height of the floors of each storey. Dictionary has the 
              following structure:
                  keys : int
                      Identifiers of storeys: 0, 1, ...
                  values : float
                      Height (in meters) in the reference coordinates system 
                      of the point cloud 
        
        """
        return self.__floors_hgt
    
    @property
    def floors_time(self):
        """
        dict : Acquisition time intervals of storeys. Dictionary has the 
             following structure:
                 keys : int
                     Identifiers of storeys: 0,1, ...
                 values : list of tuples
                     Each tuple contains two time values (in seconds) 
                     corresponding to start and end time of interval  
                             
        """
        return self.__floors_time
    
    @property
    def id_storey(self):
        """
        list: Identifiers of storeys
        
        """
        return list(self.__floors_hgt.keys())
    
    @property
    def pcd(self):
        """
        DataFrame: Loaded pointcloud data
        
        """
        return self.__pcd
    
    @property
    def trajectory(self):
        """
        Trajectory object: Loaded trajectory data
        
        """
        return self.__traj
         
    @property
    def traj_hgt_pks(self):
        """
        list: Height values of peaks detected in trajectory z-histogram.
        
        """
        return self.__traj_hgt_pks
        
    @trajectory.setter
    def trajectory(self, traj):
        if isinstance(traj, Trajectory):
            if self.required_cols.issubset(set(traj.id_fields)):
                self.__traj = traj
                #Subsampled trajectory poses
                if self.__traj_dss is not None:
                    # Trajectory subsampling
                    traj_ss = self.__traj.trajectory_subsampling(self.__traj_dss)
                    # Only 'time' and 'z' data are necessary
                    self.__traj_ss = traj_ss.loc[:, ('time','z')]
                else:
                    self.__traj_ss = self.__traj.loc[:, ('time','z')]
            else:
                self.logger.error('Input Trajectory must contains at least '
                                  '[\'time\',z\'] id fields')
                
        elif isinstance(traj, (pd.DataFrame, np.ndarray)):
            if isinstance(traj, pd.DataFrame):
                if self.required_cols.issubset(set(traj.columns)):
                    # Trajectory instance
                    self.__traj = Trajectory(traj, list(self.required_cols))
                else:
                    self.__traj = None
                    self.logger.error('Input trajectory data must contains at'
                                      'least three columns corresponding to '
                                      '[\'time\',\'z\'] coordinates')
                    
            elif isinstance(traj, np.ndarray):
                if traj.shape[1] > 3:
                    self.__traj = Trajectory(traj[:,:3],
                                             list(self.required_cols))
                else:
                    self.__traj = None
                    self.logger.error('Input trajectory data must contains at '
                                      'least three columns corresponding to '
                                      '[\'time\',\'z\'] coordinates')
                    
            if  self.__traj is not None:        
                #Subsampled trajectory poses
                if self.__traj_dss is not None:
                    traj_ss = self.__traj.trajectory_subsampling(self.__traj_dss)
                    # XYZ coordinates of points cloud and trajectory
                    try:
                        self.__traj_ss = traj_ss.loc[:, ('time', 'z')]
                    except:
                        self.__traj_ss = traj_ss.loc[:, ('x','time')]
                else:
                    try:
                        self.__traj_ss = self.__traj.loc[:, ('time', 'z')]
                    except:
                        self.__traj_ss = traj_ss.loc[:, ('x','time')]
              
        else:
            self.__traj = None
            self.__traj_ss = None
        
    @pcd.setter
    def pcd(self, pcd):
        if isinstance(pcd, pd.DataFrame):
            self.__pcd = pcd.loc[:, ('time', 'z')]
        elif isinstance(pcd, np.ndarray):
            # suppose columns array are ['time' 'x' 'y' 'z']
            if pcd.dim == 2 and pcd.shape[1] == 4:
                self.__pcd = pd.DataFrame(pcd[:, [0,3]], columns=['time','z'])
        else:
            self.__pcd = None
              
    # Private methods  
    def __get_intermediate_peaks(self, level, traj_pks, pcd_pks, pcd_val, n_max):
        """
        Determinates what detected peaks in z-histogram pointcloud 
        correspond to height of floor and ceiling of a arbitrary storey.

        Parameters
        ----------
        level : int        
            Identifier of storey.
        traj_pks : dict        
            Height values corresponding to detected trajectory peaks.
        pcd_pks : ndarray        
            Height values corresponding to detected point cloud peaks.
        pcd_val : ndarray        
            Frecuency values corresponding to detected point cloud peaks.

        Returns
        -------
        list
            Indices of floor and ceiling.
        list
            Weight of floor and ceiling.
        list
            Values corresponding to frecuency of floor and ceiling.

        """
        
        # Calculate peaks between trajectory peaks
        idx_cand_pks =  np.where((pcd_pks > traj_pks[level]) & 
                                     (pcd_pks < traj_pks[level+1]))[0]
    
        # Weight of peaks (number of points)
        weight_pks = pcd_val[idx_cand_pks].copy()
        
        # Get two maximuns in weight peaks
        idx_max = np.argpartition(weight_pks, -n_max)[-n_max:]
        
        # Heights of max values
        hgt1 = pcd_pks[idx_cand_pks[idx_max[0]]]
        hgt2 = pcd_pks[idx_cand_pks[idx_max[1]]]
        
        # Indexes of peaks of pcd
        idx_pks = np.arange(pcd_pks.size) 
        
        if hgt1 < hgt2:
            idx_floor = idx_pks[idx_cand_pks[idx_max[1]]]  
            idx_ceil = idx_pks[idx_cand_pks[idx_max[0]]]
          
        else:
            idx_floor = idx_pks[idx_cand_pks[idx_max[0]]]
            idx_ceil = idx_pks[idx_cand_pks[idx_max[1]]]
        
        hgt_floor = pcd_pks[idx_floor]
        val_floor = pcd_val[idx_floor]
         
        hgt_ceil = pcd_pks[idx_ceil]
        val_ceil = pcd_val[idx_ceil]
        
        return [idx_floor, idx_ceil], [hgt_floor, hgt_ceil], [val_ceil, val_floor]
        
    def __traj_hist(self, traj_df, min_hor_dist): 
        """
        Computes the z-histogram of the trajectory data.

        Parameters
        ----------
        traj_df : DataFrame        
            Trajectory data. Only Z coordinates are required to compute the 
            histogram.

        Returns
        -------
        list
            Height of trayectory peaks.

        """
        
        # Z-Histogram of trajectory ref
        max_z_traj_ref = traj_df.loc[:, 'z'].max() + self.bin_traj
        min_z_traj_ref = traj_df.loc[:, 'z'].min()
        
        n_bins = floor((max_z_traj_ref-min_z_traj_ref)/self.bin_traj)
    
        self.__traj_zhist= np.histogram(traj_df.loc[:, 'z'].values,
                                   bins=n_bins)
 
        # Height values histogram
        hgt_val = self.__traj_zhist[1]
        
        # Number of points histogram
        n_pts_zhist = self.__traj_zhist[0]
        
        arr_ref_traj = hgt_val[0:-1] + self.bin_traj/2.0
        
        # Min_hor_dist = 1.0
        
        if n_bins < ceil(2*min_hor_dist/self.bin_traj):
            
            idx_peaks_traj_zhist = [np.argmax(n_pts_zhist).tolist()]
        else:        
            # Mean of values of z-histogram
            mean_z_hist = np.mean(n_pts_zhist)
            
            # Peak-finding
            idx_pks = np.where(n_pts_zhist > mean_z_hist)[0]
            grp_pks= MeanShift(bandwidth=floor(2.0/self.bin_traj), 
                           bin_seeding=False).fit(np.array(idx_pks).reshape(-1,1))
            
            idx_pks_filt = []
            for lbl in np.unique(grp_pks.labels_).tolist():
                idx_grp = np.where(grp_pks.labels_ == lbl)[0]
                if idx_grp.size == 1:
                    idx_pks_filt.append(int(idx_pks[idx_grp]))
                    
                else:
                    idx_greater_pk = np.argmax(n_pts_zhist[idx_pks[idx_grp]])
                    idx_pks_filt.append(idx_pks[idx_grp[idx_greater_pk]])
            idx_peaks_traj_zhist = np.sort(np.array(idx_pks_filt))
            
        # Indexes of trayectory peaks
        if len(idx_peaks_traj_zhist) > 0:            
            # Height of trajectory peaks
            self.__traj_hgt_pks = list(arr_ref_traj[idx_peaks_traj_zhist])
            return 0
        
        else:
            return -1

    def __pcd_hist(self, n_max):
        """
        Computes the z-histogram of the trajectory data.
        
        """
        
        pcd_df = self.__pcd
         # Z-histogram of point cloud
        max_z = pcd_df.loc[:, 'z'].max()
        min_z = pcd_df.loc[:, 'z'].min()
        
        # Width bin
        res_pcd = self.bin_pcd
        
        # Number of bins of pc histogram
        n_bins = ceil((max_z-min_z)/res_pcd)
        
        # Z-histogram of point cloud data
        self.__pcd_zhist = np.histogram(pcd_df.loc[:, 'z'].values, bins=n_bins)
        
        # Height values histogram
        hgt_val_pcd = self.__pcd_zhist[1]
        
        # Midpoints of bins
        arr_ref = hgt_val_pcd[0:-1] + res_pcd/2.0
        
        # Mean histogram
        mean_pc_hist = np.mean(self.__pcd_zhist[0])
        
        # Find peaks in z-histogram
        indexes, _ = find_peaks(self.__pcd_zhist[0], 
                                height=mean_pc_hist, 
                                distance=3)
        
        # Heigth of point cloud peaks
        pcd_hgt_pks = arr_ref[indexes]
        # Vals of point clouds histograms
        pcd_val_pks = self.__pcd_zhist[0][indexes]     
        
        # Testing peaks list
        floors_hgt = []
        ceils_hgt = []
        floors_idx = []
        ceils_idx = []
        indexes_peaks = np.arange(pcd_hgt_pks.size)
        traj_hgt_pks = self.__traj_hgt_pks
        for n_pk, traj_peak in enumerate(traj_hgt_pks):
                
            if n_pk != len(traj_hgt_pks)-1:
                if n_pk == 0:
                    # Calculate height floor value
                    idx_floor_cand_pks = np.where(pcd_hgt_pks < traj_hgt_pks[n_pk])[0]
                    floor_idx_pk = np.argmin(pcd_hgt_pks[idx_floor_cand_pks])
                    floor_pk = pcd_hgt_pks[floor_idx_pk]
                    floor_idx = indexes_peaks[idx_floor_cand_pks[floor_idx_pk]]
                    # Append height value and idx
                    floors_hgt.append(floor_pk)
                    floors_idx.append(floor_idx)
                # Get ceil and floor heights between two trajectory peaks
                idx, hgts, vals = self.__get_intermediate_peaks(n_pk,
                                                              traj_hgt_pks,
                                                              pcd_hgt_pks,
                                                              pcd_val_pks,
                                                              n_max)
                # Append height and index of the ceil
                ceils_hgt.append(hgts[1])
                ceils_idx.append(idx[1])
                
                # Append height and ndex of next floor
                floors_hgt.append(hgts[0])
                floors_idx.append(idx[0])
                
            else:
                ceils_hgt.append(np.amax(pcd_hgt_pks))
                ceils_idx.append(np.argmax(pcd_hgt_pks))
                
            # Saving results in instance attributes within dictionary structure
            self.__ceils_hgt = {n:(idx, z) for n, (idx, z) in enumerate(zip(ceils_idx, ceils_hgt))}
            self.__floors_hgt = {n:(idx, z) for n, (idx, z) in enumerate(zip(floors_idx, floors_hgt))}
        
    def get_floors_intervals(self):
        """
        Returns a dictionary with acquisition time intervals is created from 
        the floor and ceiling height values. Returned dictionary has the
        following structure:
            keys: int
                Identifiers of storeys: 0,1, ...
            values: list of tuples
                Each tuple contains two time values (in seconds) 
                corresponding to start and end time of interval  

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # print("Number of storeys: ", len(self.__traj_hgt_pks)) 
        if not len(self.__traj_hgt_pks) > 1:           
            traj_df = self.__traj_ss
            
            self.__floors_time = {self.id_storey[0]: [(traj_df.time.min(),
                                                     traj_df.time.max())]}
            
        else:
            traj_df = self.__traj.trajectory
        
            # Trajectory division
            floors_hgt = [hgt_floor[1] for hgt_floor in self.__floors_hgt.values()]
            ceils_hgt = [hgt_ceil[1] for hgt_ceil in self.__ceils_hgt.values()]
            
            for level, (bottom, top) in enumerate(zip(floors_hgt, ceils_hgt)):
                # Trayectory positions between the floor and the ceil
                traj_level_df = traj_df[((traj_df.z>bottom) & (traj_df.z<top))]
                time_level = traj_level_df.time.values
                step_time = self.__traj.time_period
                
                time_diff = np.concatenate((np.zeros(1), np.diff(time_level)))
                # Finding time jumps greater than twice the acquisition time 
                # period
                cut_pos = np.where(time_diff > 2*step_time)[0]
                # Time intervals
                time_intvl = np.zeros((cut_pos.size+1, 2))
                
                init_time_intvl = np.concatenate((np.array([time_level[0]]),
                                                  np.array(time_level[cut_pos])
                                                  ))
                
                end_time_intvl = np.concatenate((time_level[cut_pos-1], 
                                                 np.array([time_level[-1]])
                                                 ))
                
                time_intvl[:, 0] = init_time_intvl
                time_intvl[:, 1] = end_time_intvl
                
                intervals_floor = []
                for n_intvl in range(time_intvl.shape[0]):
                    
                    intervals_floor.append((time_intvl[n_intvl][0],
                                            time_intvl[n_intvl][1]))
                
                # Add acqusition floor intervals to dictionary  
                self.__floors_time[level] = intervals_floor
           
        return self.__floors_time
    
    def process(self, min_hor_dist, n_max):
        """ 
        Invokes and controls the necessary methods to determine the 
        different floor levels the in point cloud. The workflow is as
        follows:
            - Compute z-histogram with trajectory positions
            
            - Find peaks on histogram computed in the previous step
            
            - If only one peaks is detected, the point cloud is treated as
              a single storey
            
            - Else, a z-histogram from point cloud data is computed.
            
            - Trajectory and pointcloud peaks are combined to determinate
              floor and ceiling heights
                
        """
        traj_df = self.__traj_ss
        
        # XYZ coordinates of points cloud and trajectory
        pcd_df = self.__pcd.loc[:, ('time','z')]
              
        # Trajectory Z-Histogram
        traj_hist_res = self.__traj_hist(traj_df, min_hor_dist)
        
        if not traj_hist_res:
            traj_hgt_pks = self.__traj_hgt_pks
        else:
            self.logger.error('Error in trajectory divsion process')

        if not (len(traj_hgt_pks) > 1):
            # A unique story
            self.__ceils_hgt = {0: pcd_df.z.max()}
            self.__floors_hgt = {0: pcd_df.z.min()}
        else:   
            # Z-Histogram of point cloud
            self.__pcd_hist(n_max) 
            
