import cv2
import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import csv
import os, math
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from PIL import Image
import os




##Open the class matrix

# Chemins des fichiers CSV
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Portable Elisabeth\\Desktop\\Synchronise\\INSA_IR\\Stage\\Stage AI Thailand\\AIT\\PostProcessing", "map_eau_Tl6.csv")                                                      #A verif
file_path = os.path.join(root, rel_path)
mat = open(file_path)
mat=np.loadtxt(mat,delimiter=",")


#Convert considered class in '1' (not use if it's not already the case)
mat[np.where(mat==0)]=1                       #2 avant ??
mat[np.where(mat==1)]=0
mat[np.where(mat==2)]=0                        #1 avant ??


#On identifie les endroits où il y a une transition sol/eau
# Calcul des dérivées des lignes et des colonnes de la matrice
d_rows = np.diff(mat, axis=0)
d_cols = np.diff(mat, axis=1)

# Recherche des indices où il y a transition du sol (0) à l'eau (1) ou inversement
water_start = np.argwhere(d_rows == -1) + 1
water_end = np.argwhere(d_rows == 1) + 1
land_start = np.argwhere(d_cols == -1) + 1
land_end = np.argwhere(d_cols == 1) + 1
print(d_rows)
## TEMPERATURE ANALYSIS
print("cxxc")
# Définition des variations de température en fonction de la distance par rapport à un point d'eau
distances_intervals  = [0, 25, 50, 100]  # Distances en mètres
temperatures_intervals = [-0.25, -0.14, -0.07, -0.05]  # Variations de température en °C

# Initialisation de la matrice de variation de température
matTemp = np.zeros_like(mat)

for start, end in zip(land_start, water_end):
    print("ccc")
    for i in range(start[0], end[0]):
        for j in range(start[1], end[1]):
            print("cc")
            distance = min(i - start[0], end[0] - i, j - start[1], end[1] - j)
            if distance <= 100:  # Si la distance est inférieure ou égale à 100 mètres
                # Calcul du logarithme de la distance normalisé entre 0 et 1
                log_distance = np.log(1 + distance) / np.log(1 + 100)
                # Interpolation linéaire des variations de température en fonction de la distance logarithmique
                temperature = np.interp(log_distance, distances_intervals, temperatures_intervals) 
                matTemp[i, j] = temperature
                

                    

# Affichage de la matrice de variation de température
print("Matrice matTemp :")
print(matTemp)
# Enregistrement de la matrice dans un fichier CSV
#np.savetxt("chemin_vers_le_fichier.csv", matTemp, delimiter=",")

# #Save the matrix of road's temperature
# rel_path = os.path.join("..", "Image result/EauTemp.csv")
# file_path = os.path.join(root, rel_path)
# pd.DataFrame(matMaisons).to_csv(file_path, index=False, header=False)
# #pd.DataFrame(matTemp).to_csv('C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/Test_temp03/07.csv', index=False, header=False)
