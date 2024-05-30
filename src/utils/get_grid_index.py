# -*- coding: utf-8 -*-

def get_grid_index(i, j, k, i_size, j_size, k_size):
    """
    Computes a unique index on a 3D grid using coordinates (i, j, k) and 
    dimension sizes (i_size, j_size, k_size).

    """
    
    idx_grid = i + i_size*j + i_size*j_size*k
    
    return idx_grid    