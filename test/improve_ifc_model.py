# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:00:04 2025

@author: HP
"""

import csv
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt    
from shapely.geometry import LineString

from src.improve_ifc import floor_ceiling_better_ifc
from src.improve_ifc import stairs_to_IFC
from src.improve_ifc import windows_better_ifc

#%% import data

df = pd.read_csv(r"C:\Users\rosam\OneDrive - Universidade de Vigo\Escritorio\Improve_IFC_Model\info_BIM_Complete_Bilbao.txt")
pcd_in_t = pd.read_csv(r"C:\Users\rosam\OneDrive - Universidade de Vigo\Escritorio\Final_Code\results\Bilbao\Storeys\Storeys_h.csv")

df_ceiling_0 = df[(df['element_type'] == 'ceiling') & (df['storey_id'] == 0)]
# Filter columns containing 'pt' (ptk x, ptk y, ptk z with k from 1 to 38)
columns_pt = [f'pt{i}_{axis}' for i in range(1, 39) for axis in ['x', 'y', 'z']]

#%% Improve IFC - Once all the code had finished

# DOORS
# Check if any of these columns contain NaN
df_no_nan = df_ceiling_0[columns_pt].dropna(axis=1, how='any')

# Rearrange columns to have the structure (x, y, z) per point
points = df_no_nan.values.reshape(-1, 3)

min_c = np.max(points[:,2])

df_w = df[df['element_type']=='opening_door'][['storey_id', 'element_id', 
                                               'opening_in_wall', 
                                                 'pt1_x', 'pt1_y', 'pt1_z', 
                                                 'pt2_x', 'pt2_y', 'pt2_z',
                                                 'pt3_x', 'pt3_y', 'pt3_z',
                                                 'pt4_x', 'pt4_y', 'pt4_z']]

# Create a list to store the results
result = []

# Group by storey_id and element_id
grouped = df_w.groupby(['storey_id', 'element_id', 'opening_in_wall'])

# Iterate over groups
for (storey_id, element_id, opening_in_wall), group in grouped:
    entry = {
        "id_storey": int(storey_id),
        "element_id": int(element_id),
        "opening_in_wall": int(opening_in_wall),  
        "points": []
    }

    for i in range(1, 5): 
        point = [
            float(group[f'pt{i}_x'].iloc[0]), 
            float(group[f'pt{i}_y'].iloc[0]), 
            float(group[f'pt{i}_z'].iloc[0])
        ]
        if point[2] > min_c:
            entry['points'].append(point)

    result.append(entry)

# Convert the result list to JSON
json_result = json.dumps(result, indent=4)


# Select all door points
# Step 1: Create a dictionary that groups the 'points' by 'id_storey'
grouped = {}
for entry in result:
    storey = entry['id_storey']
    points = entry['points']
    if storey not in grouped:
        grouped[storey] = []
    grouped[storey].append(points)

# Step 2: Create the results list, where each sublist contains the lists of
# 'points' for each 'id_storey'
result_2 = [grouped.get(storey, []) for storey in sorted(grouped.keys())]
print(result_2)

result_2 = []
for storey, points in grouped.items():
    if storey > 0:
        d=[]
        for i in points:
            sorted_points = sorted(i, key=lambda x: x[2], reverse=True)
            d.append(sorted_points[:2])
        result_2.append(d)

for r_ind, r in enumerate(result_2):
    for line_ind, line in enumerate(r):
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        plt.plot(x_values, y_values, marker='o')
        
        mid_x = (line[0][0] + line[1][0]) / 2
        mid_y = (line[0][1] + line[1][1]) / 2
        plt.text(mid_x, mid_y, f'{line_ind + 1}', fontsize=9, ha='center',
                 va='center', fontweight='bold')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Doors in the XY plane (Storey {r_ind + 1})')
    plt.xlim(-16, -4)
    plt.ylim(-7, 17)
    plt.grid(True)
    plt.show()

# List to store the XY coordinates keeping the original structure
result_xy = []

# Extract the X, Y coordinates from the result_2 structure 
# while keeping the structure
for outer_list in result_2:
    inner_result = []  
    for inner_list in outer_list:
        point_result = []  
        for point in inner_list:
            point_result.append([point[0], point[1]])
        inner_result.append(point_result)
    result_xy.append(inner_result)


for r_ind, r in enumerate(result_xy):
    print('\n--------------------------------------------\n')
    print(f'Storey {r_ind + 1}')
    print('\n--------------------------------------------\n')
    for l_ind, l in enumerate(r):
        line = LineString(l)
        midpoint = line.interpolate(0.5, normalized=True)
        buffer = midpoint.buffer(0.1)
        # See if there are doors above
        doors_other_storey = [] 
        
        # Now we check what other lines are in the buffer
        for r_inner_ind, r_inner in enumerate(result_xy):
            for l_inner_ind, l_inner in enumerate(r_inner):
                other_line = LineString(l_inner)
                
                # We check if the line is inside the buffer
                if buffer.intersects(other_line) and other_line != line:
                    print(f"Door {l_ind+1}: Storey {r_inner_ind+1} - "
                          f"Door {l_inner_ind+1}.")
    
# FLOOR AND CEILING
# Read the txt file into a DataFrame
(new_points_floor, 
 new_points_ceiling) = floor_ceiling_better_ifc.improve_floor_ceiling(
                         pcd_in_t, 
                         df,
                         last_storey=2)
     
#######################
# Substitute the points
p_floor = np.array(new_points_floor)
# Filter the row we want to modify
mask = (df['storey_id'] == 0) & (df['element_type'] == 'floor')
# Create a range of column names
columns = []
for i in range(1, 39):
    columns.append(f'pt{i}_x')
    columns.append(f'pt{i}_y')
    columns.append(f'pt{i}_z')
p_floor_flattened = p_floor.flatten()
if len(p_floor_flattened) < len(columns):
    p_floor_flattened = np.concatenate([p_floor_flattened, 
                                        np.full(len(columns) - len(p_floor_flattened), 
                                                np.nan)])

# Replace the values ​​in the DataFrame for the coordinates 
# (pt1_x, pt1_y, pt1_z,..., pt38_x, pt38_y, pt38_z)
df.loc[mask, columns] = p_floor_flattened
#######################
# Substitute the points
p_ceiling = np.array(new_points_ceiling)
# Filter the row we want to modify
mask = (df['storey_id'] == 4) & (df['element_type'] == 'ceiling')
# Create a range of column names
columns = []
for i in range(1, 39):
    columns.append(f'pt{i}_x')
    columns.append(f'pt{i}_y')
    columns.append(f'pt{i}_z')
p_ceiling_flattened = p_ceiling.flatten()
if len(p_ceiling_flattened) < len(columns):
    p_ceiling_flattened = np.concatenate([p_ceiling_flattened,
                                          np.full(len(columns) - len(p_ceiling_flattened),
                                                  np.nan)])

# Replace the values ​​in the DataFrame for the coordinates
 # (pt1_x, pt1_y, pt1_z,..., pt38_x, pt38_y, pt38_z)
df.loc[mask, columns] = p_ceiling_flattened#.reshape(1, -1)
#######################
# Save the result
df.to_csv(r"C:\Users\rosam\OneDrive - Universidade de Vigo\Escritorio\Improve_IFC_Model\info_BIM_Complete_m.txt", index=False, sep=',')



#%% WINDOWS
# Check if any of these columns contain NaN
columnas_existentes = [col for col in columns_pt if col in df_ceiling_0.columns]
df_no_nan = df_ceiling_0[columnas_existentes].dropna(axis=1, how='any')
# Rearrange the columns to have the structure (x, y, z) per point
points = df_no_nan.values.reshape(-1, 3)

min_c = np.max(points[:,2])

df_walls = df[df['element_type'] == 'wall'].copy()

lista_de_muros = []

for idx, row in df_walls.iterrows():
    coords = []
    for i in range(1, 39):
        try:
            x = row.get(f'pt{i}_x')
            y = row.get(f'pt{i}_y')
            z = row.get(f'pt{i}_z')
    
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                coords.append([x, y, z])
                
        except (TypeError, ValueError):
            continue

    coords_array = np.array(coords)
    muro_info = {
        'element_id': row['element_id'],
        'storey_id': row['storey_id'],
        'coords': coords_array
    }
    lista_de_muros.append(muro_info)

storeys_unicos = df_walls['storey_id'].unique()
num_storeys = len(storeys_unicos)

print(f"Hay {num_storeys} storeys únicos:")
print(storeys_unicos)

df_window = df[df['element_type']=='opening_window'][['storey_id', 'element_id', 
                                                 'opening_in_wall', 
                                                 'pt1_x', 'pt1_y', 'pt1_z', 
                                                 'pt2_x', 'pt2_y', 'pt2_z',
                                                 'pt3_x', 'pt3_y', 'pt3_z',
                                                 'pt4_x', 'pt4_y', 'pt4_z']]



# Create a list to store the results
result = []
# Group by storey_id and element_id
grouped = df_window.groupby(['storey_id', 'element_id', 'opening_in_wall'])

# Iterate over groups
for (storey_id, element_id, opening_in_wall), group in grouped:
    # Create the base structure for each object
    entry = {
        "id_storey": int(storey_id),
        "element_id": int(element_id),
        "opening_in_wall": int(opening_in_wall), 
        "points": []
    }

    # Collect points from pt1_x, pt1_y, pt1_z ... pt4_x, pt4_y, pt4_z
    for i in range(1, 5): 
        point = [
            float(group[f'pt{i}_x'].iloc[0]), 
            float(group[f'pt{i}_y'].iloc[0]), 
            float(group[f'pt{i}_z'].iloc[0])
        ]
        if point[2] > min_c:
            entry['points'].append(point)

    result.append(entry)

# Convert the result list to JSON
json_result = json.dumps(result, indent=4)

# Extract all points
# Create a list to store the points in the required format
points_list = []

# Iterate over each element of the JSON
for entry in result:
    points = entry['points']
    points_list.append([tuple(point) for point in points])

# Extract only the X and Y coordinates
x_coords = [punto[0] for fila in points_list for punto in fila]
y_coords = [punto[1] for fila in points_list for punto in fila]
        
mean_xy = np.round((np.mean(x_coords), np.mean(y_coords)), 2)
    
orientaciones = [windows_better_ifc.obtener_orientacion(fila) 
                 for fila in points_list]
for i, orientacion in enumerate(orientaciones):
    print(f"Rectángulo {i+1}: {orientacion}")

rectangulos_eje_x = []
rectangulos_eje_y = []

for fila in points_list:
    orientacion = windows_better_ifc.obtener_orientacion(fila)
    if orientacion == 'Eje X':
        rectangulos_eje_x.append(fila)
    elif orientacion == 'Eje Y':
        rectangulos_eje_y.append(fila)


# Classification of rectangles according to their Y value
rectangulos_up = []
rectangulos_down = []

for rectangulo in rectangulos_eje_x:
    y_values = [p[1] for p in rectangulo]
    if all(y > mean_xy[1] for y in y_values):
        rectangulos_up.append(rectangulo)
    
    elif all(y < mean_xy[1] for y in y_values):
        rectangulos_down.append(rectangulo)
    
# Display the rectangles
windows_better_ifc.graficar_rectangulos_en_plano(rectangulos_up, 
                                                 'XZ',
                                                 "Plane XZ")
# Display the rectangles
windows_better_ifc.graficar_rectangulos_en_plano(rectangulos_down,
                                                 'XZ', 
                                                 "Plane XZ")

# Classification of rectangles according to their X value
rectangulos_rigth = []
rectangulos_left = []

for rectangulo in rectangulos_eje_y:
    x_values = [p[0] for p in rectangulo]
    if all(x > mean_xy[0] for x in x_values):
        rectangulos_rigth.append(rectangulo)
    
    elif all(x < mean_xy[0] for x in x_values):
        rectangulos_left.append(rectangulo)

# Display the rectangles
windows_better_ifc.graficar_rectangulos_en_plano(rectangulos_rigth,
                                                 'YZ',
                                                 "Plane YZ")

# Display the rectangles
windows_better_ifc.graficar_rectangulos_en_plano(rectangulos_left,
                                                 'YZ',
                                                 "Plane YZ")

rect = [rectangulos_rigth, rectangulos_left, rectangulos_down, rectangulos_up]

"""
Right now there are 4 groups to enhance the windows:
    rectangulos_up
    rectangulos_down
    rectangulos_right
    rectangulos_left
"""

cents = [[np.round(windows_better_ifc.calcular_centro(rect_0),2) for rect_0 in rectangulo] for rectangulo in rect]

umbral = 0.7
w_final = []

for j, centros in enumerate(cents):
    grupos = windows_better_ifc.agrupar_centros(centros, umbral)

    for grupo in grupos:
        print([i + 1 for i in grupo])
    
    storey_rect = []
    for i in rect[j]:
        storey_rect.append(windows_better_ifc.encontrar_rectangulo(result, i))
    
    agrupados = [ [storey_rect[i] for i in grupo] for grupo in grupos ]
    centros_agrupados = [ [centros[i] for i in grupo] for grupo in grupos ]
    
    new_p = []
    for grupo, centros_g in zip(agrupados, centros_agrupados):

        if windows_better_ifc.son_numeros_continuos(grupo) and len(grupo) > 1:
            distancia = windows_better_ifc.calcular_distancia(centros_g[0], 
                                                              centros_g[1])
            
            if len(grupo) == 2:
                if 2 in grupo and 3 in grupo and 1 not in grupo and 4 not in grupo:
                    new_p.append(centros_g[0]-(0, distancia))
                    new_p.append(centros_g[1]+(0, distancia))                
    
                elif 1 in grupo and 2 in grupo and 3 not in grupo:
                    new_p.append(centros_g[1]+(0, distancia))
                    new_p.append(centros_g[1]+(0, 2*distancia))
                    
                elif 3 in grupo and 4 in grupo and 2 not in grupo:
                    new_p.append(centros_g[0]-(0, distancia))
                    new_p.append(centros_g[0]-(0, 2*distancia))
    
            elif len(grupo) == 3:
                if 4 not in grupo:
                    new_p.append(centros_g[2]+(0, distancia))
    
                elif 1 not in grupo:
                    new_p.append(centros_g[0]-(0, distancia))
    
        else:
            print("Los números no son consecutivos.")
            rango_fijo = {1, 2, 3, 4}  
            numeros_presentes = set(grupo)
            faltantes_en_grupo = list(rango_fijo - numeros_presentes)
            print(faltantes_en_grupo)
            if len(faltantes_en_grupo) == 2:
                if 1 not in faltantes_en_grupo and 3 not in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[0],
                                                                      centros_g[1])/2
                    new_p.append(centros_g[0]+(0, distancia))
                    new_p.append(centros_g[1]+(0, distancia))
    
                elif 2 not in faltantes_en_grupo and 4 not in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[0], 
                                                                      centros_g[1])/2
                    new_p.append(centros_g[0]-(0, distancia))
                    new_p.append(centros_g[1]-(0, distancia))
    
                elif 2 not in faltantes_en_grupo and 3 not in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[0],
                                                                      centros_g[1])/3
                    new_p.append(centros_g[0]-(0, distancia))
                    new_p.append(centros_g[1]+(0, distancia))
    
                elif 1 not in faltantes_en_grupo and 4 not in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[0],
                                                                      centros_g[1])/3
                    new_p.append(centros_g[0]+(0, distancia))
                    new_p.append(centros_g[0]+(0, 2*distancia))
                    
            if len(grupo) == 3 and len(grupo) == len(set(grupo)) and len(grupo) !=1:
                if 2 in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[1], 
                                                                      centros_g[2])
                    new_p.append(centros_g[0]+(0, distancia))
    
                elif 3 in faltantes_en_grupo:
                    distancia = windows_better_ifc.calcular_distancia(centros_g[0],
                                                                      centros_g[1])
                    new_p.append(centros_g[2]-(0, distancia))
    
    if j < len(rect) and rect[j]:  # Asegura que rect[j] existe y no está vacío

        if windows_better_ifc.obtener_orientacion(rect[j][0]) == 'Eje X':
            windows_better_ifc.graficar_rectangulos_en_plano_2(rect[j], 'XZ', 
                                                               'Windows and centers of the new inferred windows', 
                                                               new_p)
            w = []
            for n_p in new_p:
                x = n_p[0]
                y = n_p[1]
                w.append([[x - 0.4, y - 0.7],
                          [x - 0.4, y + 0.7], 
                          [x + 0.4, y + 0.7], 
                          [x + 0.4, y - 0.7]])
            
            if w:
                w_rounded = np.round(w, 2)
            
                y_values = [y for sublist in rect[j] for _, y, _ in sublist]
                mean_y = np.mean(y_values)
            
                w_rounded = np.array(w_rounded)
            
                w_expanded = np.zeros((w_rounded.shape[0], 
                                       w_rounded.shape[1],
                                       3))
                w_expanded[..., 0] = w_rounded[..., 0]
                w_expanded[..., 1] = mean_y
                w_expanded[..., 2] = w_rounded[..., 1]
            
                w_final.append(w_expanded)
    
            
        elif windows_better_ifc.obtener_orientacion(rect[j][0]) == 'Eje Y':
            windows_better_ifc.graficar_rectangulos_en_plano_2(rect[j], 'YZ', 
                                            'Windows and centers of the new '
                                            'inferred windows',
                                            new_p)
            w = []
            for n_p in new_p:
                x = n_p[0]
                y = n_p[1]
                w.append([[x - 0.4, y - 0.7], 
                          [x - 0.4, y + 0.7], 
                          [x + 0.4, y + 0.7],
                          [x + 0.4, y - 0.7]])
    
            if w:
                w_rounded = np.round(w, 2)
                w_rounded = np.array(w_rounded)
        
                x_values = [x for sublist in rect[j] for x, _, _ in sublist]
                mean_x = np.mean(x_values)
        
                print("w_rounded:", w_rounded)
                print("w_rounded shape:", w_rounded.shape)
        
                w_expanded = np.zeros((w_rounded.shape[0],
                                       w_rounded.shape[1],
                                       3))
        
                w_expanded[..., 0] = mean_x
                w_expanded[..., 1] = w_rounded[..., 0]
                w_expanded[..., 2] = w_rounded[..., 1]
        
                w_final.append(w_expanded)
            else:
                print("Advertencia: no se encontraron nuevos puntos"
                      " inferidos para Eje Y")


with open('w_final.txt', 'w') as f:
    for array in w_final:
        np.savetxt(f, array.reshape(-1, array.shape[-1]), fmt='%10.5f')
        f.write("\n\n")  

todas_las_ventanas = [ventana for grupo in w_final for ventana in grupo]

asignaciones = []  # (indice_ventana, id_muro, distancia)
umbral_distancia = 0.1  
ventanas_asignadas = set()
ids_unicos = df_w['element_id'].unique()

for i, ventana in enumerate(todas_las_ventanas):
    nuevo_id = i
    while nuevo_id in ids_unicos:
        nuevo_id += 1  

    centro_ventana = windows_better_ifc.centroide_rectangulo(ventana)
    mejor_muro = None
    menor_dist = float('inf')
    storey_asignado = None
    
    ids_unicos = set(ids_unicos)  
    ids_unicos.add(nuevo_id)  

    for muro in lista_de_muros:
        coords = muro['coords']
        if len(coords) < 3:
            continue  # no se puede definir plano

        normal, d = windows_better_ifc.calcular_plano(coords[0],
                                                      coords[1], 
                                                      coords[2])
        dist = windows_better_ifc.distancia_punto_a_plano(centro_ventana, 
                                                          normal, 
                                                          d)

        if dist < menor_dist:
            menor_dist = dist
            mejor_muro = muro['element_id']
            storey_asignado = muro['storey_id']

    if menor_dist < umbral_distancia:
        asignaciones.append((nuevo_id, mejor_muro, 
                             storey_asignado, menor_dist))
    else:
        asignaciones.append((nuevo_id, None, None, menor_dist))

ventanas_asignadas = set() 
ids_unicos = df_w['element_id'].unique()

asignaciones = []
for i, ventana in enumerate(todas_las_ventanas):
    nuevo_id = i
    while nuevo_id in ids_unicos:
        nuevo_id += 1  
        
    ids_unicos = set(ids_unicos)
    ids_unicos.add(nuevo_id)  
    
    for muro in lista_de_muros:
        if windows_better_ifc.ventana_dentro_bbox_muro(ventana, 
                                                       muro['coords'],
                                                       epsilon=0.1):
            print(f"Ventana {nuevo_id} está DENTRO del muro {muro['element_id']}")
            asignaciones.append((
                nuevo_id,
                muro['element_id'],      
                muro['storey_id'],     
            ))
            asignada = True
            break  

    if not asignada:
        asignaciones.append((nuevo_id, None, None, None)) 

element_type = "opening_window"
is_external = 1
height = ""

with open('ventanas_new.txt', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Encabezado opcional:
    writer.writerow(['storey_id','element_type','element_id',
                     'is_external','height','opening_in_wall','no_pts',
                     'pt1_x','pt1_y','pt1_z',
                     'pt2_x','pt2_y','pt2_z',
                     'pt3_x','pt3_y','pt3_z',
                     'pt4_x','pt4_y','pt4_z'])
    
    for (nuevo_id, mejor_muro, storey_asignado), ventana in zip(asignaciones, 
                                                                todas_las_ventanas):
        if mejor_muro is None or storey_asignado is None:
            continue  # saltar ventanas no asignadas
            
        # Elemento tipo ventana (según tu formato)
        element_type = "opening_window"
        is_external = 1
        height = ""
        opening_in_wall = mejor_muro
        no_pts = 4            

        if len(ventana) != 4:
            continue  #

        row = [
            storey_asignado, element_type, nuevo_id, is_external, height,
            opening_in_wall, no_pts
        ]

        for pt in ventana:
            row.extend([round(coord, 6) for coord in pt])  

        writer.writerow(row)
