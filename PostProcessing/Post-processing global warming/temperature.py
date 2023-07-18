import cv2
import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import csv
import os, math
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import os

#On importe la matrice température de chacune des classes et on harmonise les cluster à 100
##Open the class matrix
##Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\Image result", "RouteTemp.csv")
file_path = os.path.join(root, rel_path)
matRoute = open(file_path)
matRoute = np.loadtxt(matRoute, delimiter=",")

for col in range(matRoute.shape[1]):
    for lig in range(matRoute.shape[0]):

        if matRoute[lig, col] > 20:
            matRoute[lig, col] = 100

row = np.zeros(np.shape(matRoute[0]))
matRoute = np.append(matRoute, [row], axis=0)

rel_path = os.path.join("..", "ArbreTemp2.csv")
file_path = os.path.join(root, rel_path)
matArbre = open(file_path)
matArbre = np.loadtxt(matArbre, delimiter=",")

for col in range(matArbre.shape[1]):
    for lig in range(matArbre.shape[0]):

        if matArbre[lig, col] > 4:
            matArbre[lig, col] = 200

row = np.zeros(np.shape(matArbre[0]))
matArbre = np.append(matArbre, [row], axis=0)

rel_path = os.path.join("..", "Image result/MaisonTemp2.csv")
file_path = os.path.join(root, rel_path)
matMaison = open(file_path)
matMaison = np.loadtxt(matMaison, delimiter=",")

for col in range(matMaison.shape[1]):
    for lig in range(matMaison.shape[0]):

        if matMaison[lig, col] > 2.5:
            matMaison[lig, col] = 100

row = np.zeros(np.shape(matArbre[0]))
matMaison = np.append(matMaison, [row], axis=0)

# On crée la matrice de température finale en supperposant toutes les températures

matFinaleTemp = np.zeros(np.shape(matMaison))

matFinaleTemp = matMaison + matRoute - matArbre

# Pour avoir un affichage satisfaisant, on met la valeur de tous les cluster à 5 ou -5

for col in range(matFinaleTemp.shape[1]):
    for lig in range(matFinaleTemp.shape[0]):

        if matFinaleTemp[lig, col] > 80:
            matFinaleTemp[lig, col] = 5
        if matFinaleTemp[lig, col] < -20:
            matFinaleTemp[lig, col] = -5

# Création du plot final avec affichage des valeurs de température

MatBisTemp = matFinaleTemp[:50, :50]

fig, ax = plt.subplots()
ax.matshow(MatBisTemp, cmap=plt.cm.RdYlGn_r)

# Ajout des annotations pour les valeurs de température
previous_value = None  # Stocke la valeur précédente de température

for (i, j), value in np.ndenumerate(MatBisTemp):
    if value != previous_value:  # Vérifie si la valeur est différente de la précédente
        if value != 0:  # Vérifie si la valeur n'est pas nulle
            ax.text(j, i, f'{value}°C', ha='center', va='center', fontsize=8, color='black')
        previous_value = value

#plt.savefig("Plot_TempFinale.png")
ax.axis('off')

