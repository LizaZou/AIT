import cv2
import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import csv
import os, math
import matplotlib
import matplotlib.pyplot as plt


#On importe la matrice température de chacune des classes et on harmonise les cluster à 100
##Open the class matrix
##Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\ToulouseFinal", "RouteTempAprès.csv")
file_path = os.path.join(root, rel_path)
matRoute = open(file_path)
matRoute = np.loadtxt(matRoute, delimiter=",")


rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\ToulouseFinal", "ArbreApresTemp.csv")
file_path = os.path.join(root, rel_path)
matArbre = open(file_path)
matArbre = np.loadtxt(matArbre, delimiter=",")

min_value = np.min(matArbre)
max_value = np.max(matArbre)
print("la valeur min=",min_value)
print("la valeur max=",max_value)
# Définir les nouvelles valeurs minimale et maximale souhaitées
new_min_value = 0.0
new_max_value = 1.5

# Appliquer l'échelle linéaire pour redimensionner les valeurs
matArbre2 = ((matArbre - min_value) / (new_max_value - min_value)) 

rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\ToulouseFinal", "ChampsApresTemp.csv")
file_path = os.path.join(root, rel_path)
matChamps = open(file_path)
matChamps = np.loadtxt(matChamps, delimiter=",")




rel_path = os.path.join("..", "ToulouseFinal/MaisonApresTemp.csv")
file_path = os.path.join(root, rel_path)
matMaison = open(file_path)
matMaison = np.loadtxt(matMaison, delimiter=",")


# On crée la matrice de température finale en supperposant toutes les températures

matFinaleTemp = np.zeros(np.shape(matMaison))

matFinaleTemp = matMaison + matRoute - matArbre2 + matChamps 


# Création du plot final avec affichage des valeurs de température

from matplotlib import cm

plt.figure()
cmap = cm.get_cmap('jet')
plt.imshow(matFinaleTemp, cmap=cmap, vmin=-5, vmax=5)
plt.colorbar()
plt.axis("off")

