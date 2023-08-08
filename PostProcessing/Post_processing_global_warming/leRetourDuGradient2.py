# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 04:13:01 2023

@author: Elisabeth
"""

import cv2
import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import measurements, label
from PIL import Image

#On part de la matrice de la classe arbre, déassemblée et nettoyée

print("Open the class matrix")
root = os.path.dirname(__file__)
rel_path = os.path.join("..", "map_eau_Tl6.csv")
file_path = os.path.join(root, rel_path)
matrice = open(file_path)
# Chargement de la matrice à partir du fichier CSV
matrice = np.loadtxt(file_path, delimiter=",")

# Convertir la liste en matrice numpy
matrice = np.array(matrice, dtype=int)

# Convert considered class in '1' (not used if it's not already the case)
matrice = np.where(matrice == 0, 2, matrice)
matrice = np.where(matrice == 1, 0, matrice)
matrice = np.where(matrice == 2, 1, matrice)

print("CLEAN SMALL CLUSTER")

# Detect clusters
label_matrix, num_clusters = measurements.label(matrice)
#Threshold size of cluster to keep 
seuil = 3  # /!\ en nbre de pixels
cluster_data = []

#Creating the output data of this file 
Output = np.zeros((4,num_clusters)) #array of K columns and 4 lines (type, center(x,y), radius in m)

#Initialisation of final number of clusters (after cleaning)
nb_cluster = 0

#Creation final matrix of cluster
final_matrix = np.zeros(np.shape(label_matrix))

#Loop to treat each cluster
for k in range(1, num_clusters + 1):
    
    #Matrix of boolean -> true if it's the k-cluster
    classK = (label_matrix==k)
    
    #Coordinate and type of the k-cluster
    x,y = np.where(classK)
    typeK = int(matrice[x[0],y[0]])
    
    #Coordinate of the center of the k-cluster
    a,b = measurements.center_of_mass(classK)           
    xx = x-a
    yy = y-b

    #Determination of the radius of the k-cluster
    xx = x-a
    yy = y-b
    #Using Pythagore to find the maximum distance
    sqrx = np.square(xx)
    sqry = np.square(yy)
    maxi = max(np.sqrt((sqrx+sqry))) #maxi : radius of the k-cluster
    
    if (maxi > seuil) : 
        
        #Incrementation of final cluster number
        nb_cluster += 1 
        
        #Plot the cluster
        plt.scatter(y,x)
        #Plot the center
        plt.scatter(b,a,c='black',marker ='X',s=100)
        #Plot the area 
        circle = plt.Circle((b,a), maxi, color='r', alpha=0.2)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(circle)
                
        #Fill in the output matrix
        Output[0,nb_cluster-1]=typeK
        Output[1,nb_cluster-1]=b
        Output[2,nb_cluster-1]=a
        Output[3,nb_cluster-1]=maxi
        
        #Load the final cluster matrix 
        for h in range(len(x)) :
            final_matrix[x[h],y[h]]= 1
                    
# Update of the Output data with only the clusters to keep     
for l in range(0, (num_clusters - nb_cluster)):
    Output = np.delete(Output, num_clusters - l - 1, 1)

print('Nb clusters :', nb_cluster)

#########################################################################

print("Matrice final_matrix :")
print(final_matrix)

##########################################################################
print("TEMPERATURE ANALYSIS")


#Number of different temperature areas 
nb_halo = 3 # /!\ MODIFY

#Initialisation of auxiliary matrix for each cluster
for aux in range(1,nb_cluster+1) :
    globals() ['final_matrix'+ str(aux)]= np.zeros(np.shape(final_matrix))


#Loop to treat each area
for area in range(1,nb_halo+1) : 
    
    #Loop to treat each cluster
    for clus in range(1,nb_cluster+1) :
        
        #Size of clus-cluster in pixel
        size_cluster = len(x)
        #Find the size of the area according to the size of the cluster
        taille = 50
        
#REMARQUE: A voir si on garde la taille en pixel des cluster comme paramètre de décision 
#et par quel coefficient diviser pour obtenir la taille des aires de temperature 

#print("SALUT JE SUIS AVANT LES FOR")     

        #Loop to the size of the i-area
        for size in range(taille):
            
            #Find the coordinate of the inital clus-cluster
            label_matrix2, num_cluster2 = measurements.label(final_matrix)
            classK2 = (label_matrix2==clus)
            x2,y2 = np.where(classK2)
                        
            #Localisation of cluter in the auxiliary matrix
            for m in range(len(x2)) :
                globals() ['final_matrix'+ str(clus)][x2[m],y2[m]]= 100
        
            ##Find the coordinate of the current clus-cluster (with the added area)
            label_matrix3, num_cluster3 = measurements.label(globals() ['final_matrix'+ str(clus)])

            classK3 = (label_matrix3==1)
            x3,y3 = np.where(classK3)
    
            #Loop to treat each pixel in the cluster
            for i in range(len(x3)) :
            
                #Test each neighbor → if it's '0', we are on the edge
                if (y3[i] != len(final_matrix[1])-1)  : #Test if we are on the matrix limit 
                    if (globals() ['final_matrix'+ str(clus)][x3[i],y3[i]+1]==0) : #Test one neighbor
                        globals() ['final_matrix'+ str(clus)][x3[i],y3[i]+1]=area+1 #Add the edge to the area in the auxiliary matrix 
                if (y3[i] != 0):
                    if (globals() ['final_matrix'+ str(clus)][x3[i],y3[i]-1]==0) :
                        globals() ['final_matrix'+ str(clus)][x3[i],y3[i]-1]=area+1
                if (x3[i] != 0):
                    if (globals() ['final_matrix'+ str(clus)][x3[i]-1,y3[i]]==0) :
                        globals() ['final_matrix'+ str(clus)][x3[i]-1,y3[i]]=area+1
                if (x3[i] != (len(final_matrix)-1)) :
                    if (globals() ['final_matrix'+ str(clus)][x3[i]+1,y3[i]]==0) :
                        globals() ['final_matrix'+ str(clus)][x3[i]+1,y3[i]]=area+1

#Definition of temperature rise coefficient
print("taille cluster=",size_cluster)

# Distances par rapport au cluster
#distances_intervals = [0, 25, 50, 100]  # Distances en mètre
# Conversion en pixel : 3 pixels = 1 mètre
distances_intervals = [0, 75, 150, 300, 500]
# Variation de température au sol constatées à ces distances
temperatures_intervals = [-0.25, -0.14, -0.07, -0.05, 0]  # Variations de température en °C

# Initialisation of final temperature matrix
matWater = np.zeros(np.shape(final_matrix))

print("traitement des matrices auxiliaires")

# Loop to treat each auxiliary matrix
for h in range(1, nb_cluster+1):
    # Attribuate temperature coefficient in the auxiliary matrix
    for col in range(globals()['final_matrix'+str(h)].shape[1]):
        for lig in range(globals()['final_matrix'+str(h)].shape[0]):
            value = globals()['final_matrix'+str(h)][lig, col]
            if value == 2:
                globals()['final_matrix'+str(h)][lig, col] = temperatures_intervals[0]
            elif value == 3:
                globals()['final_matrix'+str(h)][lig, col] = temperatures_intervals[1]
            elif value == 4:
                globals()['final_matrix'+str(h)][lig, col] = temperatures_intervals[2]
            elif value == 5:
                globals()['final_matrix'+str(h)][lig, col] = temperatures_intervals[3]
            # Find cluster
            elif value == 100:
                globals()['final_matrix'+str(h)][lig, col] = temperatures_intervals[4]  # /!\ MODIFY

    
    #Load the final temperature matrix 
    matWater +=  globals() ['final_matrix'+ str(h)]

#REMARQUE: Ici notre matrice finale est la somme de toute les augmentations de température générées
#à voir si on garde ce modèle 

print("j'ai fait le taf, j'affiche")

fig, ax = plt.subplots()
ax.imshow(matWater, cmap=plt.cm.Greens)
ax.axis('off')  # Désactive l'affichage des axes et des étiquettes

plt.show()

# Save the matrix of floor's temperature
rel_path = os.path.join("..", "eauTemp2007.csv")
file_path = os.path.join(root, rel_path)
pd.DataFrame(matWater).to_csv(file_path, index=False, header=False)
