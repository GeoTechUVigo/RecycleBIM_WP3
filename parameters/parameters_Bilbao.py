# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import open3d as o3d

# Storey Segmentation (with trajectory)
nt = 0.9
ransac_th_2 = 0.25
init_n = 3
itera_2 = 1000
h_2 = 0.75
g_res = 100
threshold_size = 10

# Room Segmentation
voxel_size = 0.05
eps_1 = 0.1 
eps_2 = 0.5
pt = 2
nt = 0.9
alpha = 3.2
ransac_th = 0.4
zmax = 1000
zmin = -1000
dist =  0.1
s_p = o3d.geometry.KDTreeSearchParamKNN(40)

pcd_in = None
vox_idx = np.array([])
clean_idx = np.array([])
vox_labels = pd.DataFrame() 
empty_in_lbl = 1
occ_lbl = 10 
width_door = 1

# pts_door = 4 
# eps_door = 2.5 
# min_pt_door = 2 

min_ratio = 0.05
threshold = 0.09

iterat = 3000
init_n = 3
h = 1.6

max_distance = 0.3 
min_points_to_save = 3
dist_pl = 0.5
min_angle = 0
max_angle = 40
dist_centroids_1 = 0.3
b_f = 0.5
min_adjacent_pairs = 6  
max_buffer_radius = 2
dist_centroids_2 = 4

ex_v = [None] 

min_p_int = 4
max_dist_int = 8
min_angle_d = 80 
max_angle_d = 100 
perfect_angle = 90
flat_angle = 180

angle_points_global = 45

ang_1 = 45
ang_2 = 135

images = True
write = False

dist_p = 1
distance_th = 0.2
dist_bound = 0.5
alph_boundary = 2
dist_th=0.35
min_dist_w = 0.3
um=0.005
um_d=0.1
# Doors
dist_traj = 0.5
b_traj=0.05
th_d = 0.1

# Windows
dist_th_w = 0.2

dist_max = 0.05
iter_i = 1 
min_a = 0.5
max_a = 6

th_boundary = 0.2

th_g = 0.5
th_w = 0.1
s_h = 0.7 
s_w = 0.4 
w_width = 0.35
w_high = 2.2
w_th = 0.4
h_f_w = 0.4

# Stairs
eps_s = 0.5
pts_s = 90
ransac_s = 0.1
iterat_s = 5000

# Stairs to IFC
max_dist_stairs = 0.5
eps_stairs = 0.05
pts_stairs = 2
vertical_step = 0.15  
horizontal_step = 0.25  




def str_el(n_storey):
    if n_storey == 0:
        se = np.ones((17,17,17))
        # se=np.ones((20,20,20))
    elif n_storey == 1:
        se = np.ones((17,17,17))
        # se=np.ones((18,18,18)) 
        # se=np.ones((19,19,19)) 
    elif n_storey == 2:
        se = np.ones((14,14,14))
        # se=np.ones((16,16,16)) 
    elif n_storey == 3:
        se = np.ones((15,15,15))
        # se=np.ones((18,18,18)) 
    elif n_storey == 4:
        se = np.ones((14,14,14))
        # se=np.ones((18,18,18))
        # se=np.ones((20,20,20))
    
    return se
            





