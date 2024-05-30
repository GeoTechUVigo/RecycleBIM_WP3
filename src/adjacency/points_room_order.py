# -*- coding: utf-8 -*-

import numpy as np
import json


def is_point_on_line(point, line, tolerance=1e-2):
    """
    See for each point which walls it joins

    """
    x, y = point
    A, B, C = line
    result = A * x + B * y + C
    return abs(result) < tolerance

def clockwise_sort(points):
    """
    Arrange the points clockwise

    """
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    return sorted_points


def points_room_order(points_intersection, path_walls):
    """
    Arrange the points clockwise

    """
    
    points_f = []
    for r in points_intersection:
        pts = []
        for i in r:
            pts.append([i[0], i[1]])
        
        sorted_points = clockwise_sort(np.array(pts))
        points_f.append(sorted_points[::-1].tolist())
       
    np.savetxt(path_walls + '\\ordered_points.txt', np.concatenate(points_f))
    
    return points_f

def data_structure(class_pcd_df, lines_position_t, points_f, df_walls_t, ex_v, path_walls):
    """
    Save data

    """
    v = [valor for valor in range(1, len(class_pcd_df.groupby('id_room')) + 1) if valor not in ex_v]
    
    data = []
    
    for id_room, (line_set, points_set, walls_set) in enumerate(zip(lines_position_t, points_f, df_walls_t)):
        for id_point, point in enumerate(points_set):
            int_id_lines = []
            for id_line, line in enumerate(line_set):
                if is_point_on_line(point, line, tolerance=1e-2):
                    int_id_lines.append(id_line)
            
            point_info = {
                'id_room' : v[id_room],
                'id_point': id_point,
                'point': point,  
                'join_walls': int_id_lines,
                'equations': [line_set[line_id] for line_id in int_id_lines],
                
            }
            data.append(point_info)
    
    
    with open(path_walls + "\\data_structure.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
