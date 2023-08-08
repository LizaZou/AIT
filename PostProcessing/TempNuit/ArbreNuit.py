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


#On part de la matrice de la classe arbre, déassemblée et nettoyée

#Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\ToulouseFinal", "ArbreAvantTemp.csv")
file_path = os.path.join(root, rel_path)
mat = open(file_path)
mat=np.loadtxt(mat,delimiter=",")

# Convert considered class in '1' (not use if it's not already the case)
mat[np.where(mat==0)]=2
mat[np.where(mat==1)]=0
mat[np.where(mat==2)]=1

# CLEAN SMALL CLUSTER

#Detect clusters
lw, num = measurements.label(mat) #num: nb initial of cluster 
                                  #lw: matrix of cluster (each one has its own number)

#Threshold size of cluster to keep 
seuil = 10  # /!\ MODIFY

#Creating the output data of this file 
Output = np.zeros((4,num)) #array of K columns and 4 lines (type, center(x,y), radius in m)

#Initialisation of final number of clusters (after cleaning)
nb_cluster = 0

#Creation final matrix of cluster
matF = np.zeros(np.shape(lw))

#Loop to treat each cluster
for k in range(1,num+1):
    
    #Matrix of boolean -> true if it's the k-cluster
    classK = (lw==k)
    
    #Coordinate and type of the k-cluster
    x,y = np.where(classK)
    typeK = int(mat[x[0],y[0]])
    
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
            matF[x[h],y[h]]= 1
            
#Update of the Output data with only the clusters to keep     
for l in range(0, (num-nb_cluster)):
    Output = np.delete(Output, num-l-1, 1)
print('Nb clusters : ' + str(nb_cluster))

#########################################################################

print("Matrice matF :")
print(matF)

# Enregistrer la matrice dans un fichier CSV
#chemin_fichier_csv = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/matFARBRE0507.csv"
#pd.DataFrame(matF).to_csv(chemin_fichier_csv, index=False, header=False)
#print("La matrice a été enregistrée dans le fichier CSV :", chemin_fichier_csv)


##########################################################################
# TEMPERATURE ANALYSIS

#Number of different temperature areas 
# ... (previous code)

# TEMPERATURE ANALYSIS

# Number of different temperature areas 
# Define the number of clusters and areas
nb_halo = 3

# Define the hours and temperature values
hours = [17, 20, 22, 0, 3, 6]
temp_valeur = [-1.7, -0.5, -0.1, -0.3, -1.1, -2]

# Loop through the specified hours and temperatures
for hour, temp_change in zip(hours, temp_valeur):
    # Initialisation of auxiliary matrix for each cluster
    for aux in range(1, nb_cluster+1):
        globals()['matF' + str(aux)] = np.zeros(np.shape(matF))

    # Loop to treat each area
    for area in range(1, nb_halo + 1):
        # Loop to treat each cluster
        for clus in range(1, nb_cluster + 1):

            # Size of clus-cluster in pixel
            size_cluster = len(x)
            # Find the size of the area according to the size of the cluster
            taille = 50

            # REMARQUE: A voir si on garde la taille en pixel des clusters comme paramètre de décision 
            # et par quel coefficient diviser pour obtenir la taille des aires de temperature 

            print("SALUT JE SUIS AVANT LES FOR")

            # Loop to the size of the i-area
            for size in range(taille):

                # Find the coordinate of the inital clus-cluster
                lw2, num2 = measurements.label(matF)
                classK2 = (lw2 == clus)
                x2, y2 = np.where(classK2)

                # Localisation of cluster in the auxiliary matrix
                for m in range(len(x2)):
                    globals()['matF' + str(clus)][x2[m], y2[m]] = 100

                ## Find the coordinate of the current clus-cluster (with the added area)
                lw3, num3 = measurements.label(globals()['matF' + str(clus)])
                classK3 = (lw3 == 1)
                x3, y3 = np.where(classK3)

                # Loop to treat each pixel in the cluster
                for i in range(len(x3)):

                    # Test each neighbor → if it's '0', we are on the edge
                    if (y3[i] != len(matF[1]) - 1):  # Test if we are on the matrix limit
                        if (globals()['matF' + str(clus)][x3[i], y3[i] + 1] == 0):  # Test one neighbor
                            globals()['matF' + str(clus)][x3[i], y3[i] + 1] = area + 1  # Add the edge to the area in the auxiliary matrix
                    if (y3[i] != 0):
                        if (globals()['matF' + str(clus)][x3[i], y3[i] - 1] == 0):
                            globals()['matF' + str(clus)][x3[i], y3[i] - 1] = area + 1
                    if (x3[i] != 0):
                        if (globals()['matF' + str(clus)][x3[i] - 1, y3[i]] == 0):
                            globals()['matF' + str(clus)][x3[i] - 1, y3[i]] = area + 1
                    if (x3[i] != (len(matF) - 1)):
                        if (globals()['matF' + str(clus)][x3[i] + 1, y3[i]] == 0):
                            globals()['matF' + str(clus)][x3[i] + 1, y3[i]] = area + 1

    # Modify the temperature rise coefficients based on size_cluster and hour
    if size_cluster < 350:
        R1 = 0.58 - (0.05 * hours.index(hour))
        R2 = 0.52 - (0.05 * hours.index(hour))
        R3 = 0.22 - (0.05 * hours.index(hour))
        R4=3 - (0.15*hours.index(hour))
    elif 350 <= size_cluster < 1100:
        R1 = 1.01 - (0.10 * hours.index(hour))
        R2 = 0.94 - (0.10 * hours.index(hour))
        R3 = 0.73 - (0.10 * hours.index(hour))
        R4=3 - (0.15*hours.index(hour))
    else:
        R1 = 1.32 - (0.17 * hours.index(hour))
        R2 = 1.14 - (0.17 * hours.index(hour))
        R3 = 0.57 - (0.17 * hours.index(hour))
        R4=  3.1 - (0.15 * hours.index(hour))
    # Initialisation of final temperature matrix  
    matTree = np.zeros(np.shape(matF))
          
    # Loop to treat each auxiliary matrix
    for h in range(1, nb_cluster + 1):
        # Attribuate temperature coefficient in the auxiliary matrix
        for col in range(globals()['matF' + str(h)].shape[1]):
            for lig in range(globals()['matF' + str(h)].shape[0]):

                if (globals()['matF' + str(h)][lig, col] == 2):
                    globals()['matF' + str(h)][lig, col] = R1

                if (globals()['matF' + str(h)][lig, col] == 3):
                    globals()['matF' + str(h)][lig, col] = R2

                if (globals()['matF' + str(h)][lig, col] == 4):
                    globals()['matF' + str(h)][lig, col] = R3

                # Find cluster 
                if (globals()['matF' + str(h)][lig, col] == 100):
                    globals()['matF' + str(h)][lig, col] = R4  # /!\ MODIFY 

        # Load the final temperature matrix 
        matTree += globals()['matF' + str(h)]

    # ... (previous code for plotting, etc.)

    # Plot the temperature area 
    fig, ax = plt.subplots()
    ax.imshow(matTree, cmap=plt.cm.Greens)
    ax.axis('off')  # Désactive l'affichage des axes et des étiquettes

    plt.show()

    # Save the matrix of road's temperature
    rel_path = os.path.join("..", "ToulouseFinal/ArbreAprèsTemp_{}h.csv".format(hour))
    file_path = os.path.join(root, rel_path)
    pd.DataFrame(matTree).to_csv(file_path, index=False, header=False)
