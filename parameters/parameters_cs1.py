# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

voxel_size = 0.1
eps = 0.3
pt = 2
nt = 0.9
alpha = 3.2
ransac_th = 0.12
zmax = 1000
zmin = -1000
dist =  0.1

pcd_in = None
se = None
vox_idx = np.array([])
clean_idx = np.array([])
vox_labels = pd.DataFrame() 
empty_in_lbl = 1
occ_lbl = 10 
width_door = 1.5

pts_door = 2.5 
eps_door = 2 
min_pt_door = 2 

min_ratio = 0.05
threshold = 0.09
iterat = 2000
init_n = 3
h = 2.5 

max_distance = 0.5
min_points_to_save = 3
dist_pl = 2.5
min_angle = 0
max_angle = 40
dist_centroids_1 = 5
b_f = 0.5
min_adjacent_pairs = 6  # Minimum number of pairs of adjacent walls required
max_buffer_radius = 3
dist_centroids_2 = 4

ex_v = [1]


min_p_int = 4
max_dist_int = 8
min_angle_d = 85
max_angle_d = 95
perfect_angle = 90
flat_angle = 180


angle_points_global = 20


dist_max = 0.05

images = False
write = False

