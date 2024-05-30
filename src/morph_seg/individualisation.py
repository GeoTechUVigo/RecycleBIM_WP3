# -*- coding: utf-8 -*-

import numpy as np
import cc3d

def room_individualisation(image_erode, voxel_size):
    """
    Create individual segmented rooms and relabelling rooms.

    Parameters
    ----------
    image_erode : ndarray of bools
        Erosion of the input by the structuring element.
    voxel_size : float
        Voxel size
        Is measured in meters.

    Returns
    -------
    labels_out : ndarray of bools
        Relabelling rooms.

    """
    labels_out = cc3d.connected_components(image_erode, connectivity=26)
    uni_lbls = np.unique(labels_out).tolist()
    
    for lbl in uni_lbls:
        n_pts = labels_out[labels_out==lbl].size
        if n_pts < int(5 /voxel_size):
            labels_out[labels_out==lbl] = 0
        else:
            print('label: {}, n points: {}'.format(lbl, n_pts))

    # Relabelling rooms
    new_lbls = [*range(len(np.unique(labels_out)))]
    old_lbls = sorted(np.unique(labels_out))
    
    for new_lbl, old_lbl in zip(new_lbls, old_lbls):             
        labels_out[np.where(labels_out==old_lbl)] = new_lbl

    return labels_out