# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 07:04:49 2023

@author: Elisabeth
"""

import csv
import numpy as np
import cv2
import pandas as pd
import math
import os
from scipy.ndimage import measurements


# Fonction pour calculer le gradient de température en fonction de la distance
def calculate_gradient(distance_to_water):
    gradient_values = [-0.25, -0.14, -0.07, -0.05, 0.0]
    max_distance = 100.0
    num_zones = len(gradient_values)
    
    gradient = np.zeros_like(distance_to_water, dtype=float)
    for i in range(num_zones):
        mask = (distance_to_water > i * max_distance / num_zones)
        gradient[mask] = gradient_values[i]
    
    return gradient

# Fonction pour calculer la distance entre deux points (x1, y1) et (x2, y2)
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Charger la matrice depuis le fichier CSV
#Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\Image result", "map_eau_Tl6.csv")
file_path = os.path.join(root, rel_path)
matrice = open(file_path)
matrice=np.loadtxt(matrice,delimiter=",")

#Detect clusters
lw, num = measurements.label(matrice) #num: nb initial of cluster 
                                  #lw: matrix of cluster (each one has its own number)
#Threshold size of cluster to keep 
seuil = 3  # /!\ en nbre de pixels

#Creating the output data of this file 
Output = np.zeros((4,num)) #array of K columns and 4 lines (type, center(x,y), radius in m)

#Initialisation of final number of clusters (after cleaning)
nb_cluster = 0

#Creation final matrix of cluster
matF = np.zeros(np.shape(lw))

#Loop to treat each cluster
for k in range(1,num+1):
    
    #Matrix of boolean -> true if it's the k-cluster (s'il appartient au cluster actuellement traité)
    classK = (lw==k)
    
    #Coordinate and type of the kième-cluster
    x,y = np.where(classK)
    typeK = int(matrice[x[0],y[0]]) #array of K columns and 3 lines (type, center(x,y)) stocke le type du cluster
    
    # Calcul des dérivées des lignes et des colonnes de la matrice
    d_rows = np.diff(classK, axis=0)
    d_cols = np.diff(classK, axis=1)

    # Recherche des indices où il y a transition du sol (0) à l'eau (1) ou inversement
    water_start = np.argwhere(d_rows == -1) + 1
    water_end = np.argwhere(d_rows == 1) + 1
    land_start = np.argwhere(d_cols == -1) + 1
    land_end = np.argwhere(d_cols == 1) + 1

    # Définition des variations de température en fonction de la distance
    distances_intervals = [0, 25, 50, 100]  # Distances en mètres
    temperatures_intervals = [-0.25, -0.14, -0.07, -0.05]  # Variations de température en °C
    
    # Calcul du gradient de température
    matTemp = np.zeros_like(mat)
    for start, end in zip(land_start, water_end):
        for i in range(start[0], end[0]):
            for j in range(start[1], end[1]):
                distance = min(i - start[0], end[0] - i, j - start[1], end[1] - j)
                if distance <= 100:  # Si la distance est inférieure ou égale à 100 mètres
                    # Interpolation linéaire des variations de température en fonction de la distance
                    temperature = np.interp(distance, distances_intervals, temperatures_intervals)
                    matTemp[i, j] = temperature

# Affichage de la matrice de variation de température
print("Matrice matTemp :")
print(matTemp)
# Enregistrement de la matrice dans un fichier CSV
np.savetxt("matriceVariationTempEau.csv", matTemp, delimiter=",")

# for i in range(1, rows-1):
#     for j in range(1, cols-1):
#         if matrice[i, j] == 0:
#             # Recherche de la transition entre l'eau et le sol (0)
#             neighbors = matrice[i-1:i+2, j-1:j+2]
#             if np.any(neighbors != 0):
#                 distance_to_water = np.zeros_like(neighbors, dtype=float)
#                 for k in range(3):
#                     for l in range(3):
#                         if neighbors[k, l] == 0:
#                             distance_to_water[k, l] = np.min([distance(k, l, x, y) for x, y in zip(*np.where(neighbors == 1))])
#                 gradient = calculate_gradient(distance_to_water)
#                 gradient_matrice[i, j] = matrice[i, j] + gradient[1, 1]

# Afficher le résultat
print(gradient_matrice)

