# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt    


# Function to determine the orientation
def obtener_orientacion(fila):
    x = [p[0] for p in fila]
    y = [p[1] for p in fila]
    z = [p[2] for p in fila]

    # Calculate the differences between consecutive points
    dx = [x[i+1] - x[i] for i in range(len(x)-1)]  
    dy = [y[i+1] - y[i] for i in range(len(y)-1)]  
    dz = [z[i+1] - z[i] for i in range(len(z)-1)]  

    # See which axis has the greatest difference 
    # (indicating the main orientation)
    suma_dx = sum(abs(d) for d in dx)
    suma_dy = sum(abs(d) for d in dy)
    suma_dz = sum(abs(d) for d in dz)
    # print(suma_dx,suma_dy, suma_dz)
    
    if suma_dx > suma_dy and suma_dx > suma_dz:
        return 'Eje X'  
    elif suma_dy > suma_dx and suma_dy > suma_dz:
        return 'Eje Y'  
    elif suma_dz > suma_dx and suma_dz > suma_dy:
        return 'Eje Z'  
    else:
        return 'Indeterminado' 

def graficar_rectangulos_en_plano(rectangulos, plano, titulo):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for idx, rectangulo in enumerate(rectangulos):
        if plano == 'XZ':
            x = [p[0] for p in rectangulo]
            z = [p[2] for p in rectangulo]
            ax.plot(x + [x[0]], z + [z[0]], marker='o') 
            centroide_x = sum(x) / len(x)
            centroide_z = sum(z) / len(z)
        elif plano == 'YZ': 
            y = [p[1] for p in rectangulo]
            z = [p[2] for p in rectangulo]
            ax.plot(y + [y[0]], z + [z[0]], marker='o')  
            centroide_y = sum(y) / len(y)
            centroide_z = sum(z) / len(z)
        
        if plano == 'XZ':
            ax.text(centroide_x, centroide_z, str(idx+1),
                    fontsize=12, ha='center', va='center')
        elif plano == 'YZ':
            ax.text(centroide_y, centroide_z, str(idx+1),
                    fontsize=12, ha='center', va='center')

    if plano == 'XZ':
        # ax.set_xlim(-17, -3)
        # ax.set_ylim(-16, 4) 
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    elif plano == 'YZ':
        # ax.set_xlim(-8, 18)  
        # ax.set_ylim(-16, 4)  
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        
    ax.set_title(titulo)
    plt.show()


def graficar_rectangulos_en_plano_2(rectangulos, plano, titulo, new_p=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for idx, rectangulo in enumerate(rectangulos):
        if plano == 'XZ':  
            x = [p[0] for p in rectangulo]
            z = [p[2] for p in rectangulo]
            ax.plot(x + [x[0]], z + [z[0]], marker='o')  

            centroide_x = sum(x) / len(x)
            centroide_z = sum(z) / len(z)
        elif plano == 'YZ':  # Eje YZ
            y = [p[1] for p in rectangulo]
            z = [p[2] for p in rectangulo]
            ax.plot(y + [y[0]], z + [z[0]], marker='o') 

            centroide_y = sum(y) / len(y)
            centroide_z = sum(z) / len(z)
        
        if plano == 'XZ':
            ax.text(centroide_x, centroide_z, str(idx+1), fontsize=12,
                    ha='center', va='center')
        elif plano == 'YZ':
            ax.text(centroide_y, centroide_z, str(idx+1), fontsize=12,
                    ha='center', va='center')
    
    if new_p is not None:
        new_x = [p[0] for p in new_p]
        new_y = [p[1] for p in new_p]
        
        if plano == 'XZ':
            ax.scatter(new_x, new_y, color='blue', label="Puntos Nuevos") 
        elif plano == 'YZ':
            ax.scatter(new_x, new_y, color='blue', label="Puntos Nuevos")  
    
    if plano == 'XZ':
        # ax.set_xlim(-17, -3)
        # ax.set_ylim(-16, 4)  
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    elif plano == 'YZ':
        # ax.set_xlim(-8, 18) 
        # ax.set_ylim(-16, 4) 
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        
    ax.set_title(titulo)
    
    plt.show()        

def calcular_centro(rectangulo):
    if obtener_orientacion(rectangulo) == 'Eje X':
        y_values = [p[0] for p in rectangulo]
    elif obtener_orientacion(rectangulo) == 'Eje Y':
        y_values = [p[1] for p in rectangulo]
    
    z_values = [p[2] for p in rectangulo]
    centro_y = np.mean(y_values)
    centro_z = np.mean(z_values)
    return centro_y, centro_z

# Function to group centers according to the X axis
def agrupar_centros(centros, umbral):
    grupos = []
    
    for i, centro in enumerate(centros):
        agrupado = False
        for grupo in grupos:
            if abs(centros[grupo[0]][0] - centro[0]) < umbral:
                grupo.append(i)  
                agrupado = True
                break
        
        if not agrupado:
            grupos.append([i])  #

    return grupos

def son_iguales(lista1, lista2):
    if len(lista1) != len(lista2):
        return False
    
    for i in range(len(lista1)):
        if lista1[i] != list(lista2[i]):
            return False
    
    return True

def encontrar_rectangulo(points_data, rectangulo_down_0):
    for data in points_data:
        resultado = son_iguales(data['points'], rectangulo_down_0)
        if resultado==True:
            print(data['id_storey'])
            return data['id_storey'] 

def son_numeros_continuos(resultados):
    resultados_ordenados = sorted(resultados)
    
    for i in range(len(resultados_ordenados) - 1):
        if resultados_ordenados[i] + 1 != resultados_ordenados[i + 1]:
            return False
    return True

def calcular_distancia(centro1, centro2):
    if len(centro1) == 2: 
        distancia = centro2[1] - centro1[1]
    
    else:
        raise ValueError("Las coordenadas deben ser 2D o 3D")
    
    return distancia


def encontrar_faltantes(agrupados):
    faltantes = []
    
    for grupo in agrupados:
        minimo, maximo = min(grupo), max(grupo)
        rango_completo = set(range(minimo, maximo + 1))
        numeros_presentes = set(grupo)
        faltantes_en_grupo = rango_completo - numeros_presentes
        faltantes.append(sorted(faltantes_en_grupo))
    
    return faltantes

def pertenece_a_algun_muro(ventana, lista_de_muros):
    centro = calcular_centro(ventana)
    for muro in lista_de_muros:
        if muro.contiene_punto(centro):
            return True
    return False


def calcular_plano(p1, p2, p3):
    # Vector normal al plano
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, p1)
    return normal, d

def distancia_punto_a_plano(p, normal, d):
    return abs(np.dot(normal, p) + d)

def centroide_rectangulo(rect):
    return np.mean(rect, axis=0)

def ventana_dentro_bbox_muro(ventana, coords_muro, epsilon=0.1):
    ventana = np.array(ventana)
    coords_muro = np.array(coords_muro)

    min_muro = coords_muro.min(axis=0) - epsilon
    max_muro = coords_muro.max(axis=0) + epsilon

    return np.all((ventana >= min_muro) & (ventana <= max_muro))
