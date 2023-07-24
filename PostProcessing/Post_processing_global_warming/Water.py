#import cv2
#import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import measurements


#On part de la matrice de la classe arbre, déassemblée et nettoyée

#Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Portable Elisabeth\\Desktop\\Synchronise\\INSA_IR\\Stage\\Stage_AI_Thailand\\AIT\\PostProcessing", "map_arbre_Tl3.csv")
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
seuil = 100  # /!\ MODIFY

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
nb_halo = 3  # /!\ MODIFY

# Initialisation of auxiliary matrix for each cluster
for aux in range(1, nb_cluster+1):
    globals()['matF' + str(aux)] = np.zeros(np.shape(matF))
    
# Distances par rapport au cluster
#distances_intervals = [0, 25, 50, 100]  # Distances en mètre
# Conversion en pixel : 3 pixels = 1 mètre
distances_intervals = [0, 75, 150, 300, 500]
# Variation de température au sol constatées à ces distances
temperatures_intervals = [-0.25, -0.14, -0.07, -0.05, 0]  # Variations de température en °C

# # Initialisation of final temperature matrix
# matWater = np.zeros(np.shape(matF))
print("c'est de la merde 1")
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
                print("c'est de la merde 2")

                # Ajout du calcul du gradient autour du cluster
                for h in range(1, nb_cluster+1):
                    for col in range(globals()['matF' + str(h)].shape[1]):
                        for lig in range(globals()['matF' + str(h)].shape[0]):
                            # Trouver les coordonnées du pixel le plus proche du cluster
                            cluster_pixels = np.where(globals()['matF' + str(h)] == 100)
                            min_distance = np.inf
                            nearest_pixel_x, nearest_pixel_y = None, None
                        
                            for x, y in zip(cluster_pixels[1], cluster_pixels[0]):
                                distance = np.sqrt((x - col)**2 + (y - lig)**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_pixel_x, nearest_pixel_y = x, y
                        
                            # Parcourez les distances par rapport au bord du cluster
                            for i in range(len(distances_intervals)):
                                distance_from_border = np.sqrt((nearest_pixel_x - col)**2 + (nearest_pixel_y - lig)**2)
                                if distance_from_border <= distances_intervals[i]:
                                    globals()['matF' + str(h)][lig, col] = temperatures_intervals[i]
                                    break
                print("c'est de la merde 3")

                #Loop to treat each auxiliary matrix
                for h in range(1, nb_cluster+1) :
                    #Attribuate temperature coefficient in the auxiliary matrix
                    for col in range(globals() ['matF'+ str(h)].shape[1]):
                        for lig in range(globals() ['matF'+ str(h)].shape[0]):
                            
                            # Vérifiez si le pixel appartient au cluster
                            if (globals()['matF' + str(h)][lig, col] == 100):
                                cluster_center_x, cluster_center_y = col, lig
                                
                                # Parcourez les distances par rapport au cluster
                                for i in range(len(distances_intervals)):
                                    distance = np.sqrt((col - cluster_center_x)**2 + (lig - cluster_center_y)**2)
                                    if distance <= distances_intervals[i]:
                                        globals()['matF' + str(h)][lig, col] = temperatures_intervals[i]
                            
                                        if (globals()['matF' + str(h)][lig, col] == 2):
                                            globals()['matF' + str(h)][lig, col] = temperatures_intervals[0]
                            
                                        if (globals()['matF' + str(h)][lig, col] == 3):
                                            globals()['matF' + str(h)][lig, col] = temperatures_intervals[1]
                            
                                        if (globals()['matF' + str(h)][lig, col] == 4):
                                            globals()['matF' + str(h)][lig, col] = temperatures_intervals[2]
                            
                                        if (globals()['matF' + str(h)][lig, col] == 5):
                                            globals()['matF' + str(h)][lig, col] = temperatures_intervals[3]
                                            
                                        # Find cluster
                                        if (globals()['matF' + str(h)][lig, col] == 100):
                                            globals()['matF' + str(h)][lig, col] = temperatures_intervals[4]  # /!\ MODIFY
                print("c'est de la merde 4")


# #Definition of temperature rise coefficient
# print("taille cluster=",size_cluster)

# # Distances par rapport au cluster
# #distances_intervals = [0, 25, 50, 100]  # Distances en mètre
# # Conversion en pixel : 3 pixels = 1 mètre
# distances_intervals = [0, 75, 150, 300, 500]
# # Variation de température au sol constatées à ces distances
# temperatures_intervals = [-0.25, -0.14, -0.07, -0.05, 0]  # Variations de température en °C

# # Initialisation of final temperature matrix
# matWater = np.zeros(np.shape(matF))

# #Initialisation of final temperature matrix  
# matWater = np.zeros(np.shape(matF))

# Ajout du calcul du gradient autour du cluster
# for h in range(1, nb_cluster+1):
#     for col in range(globals()['matF' + str(h)].shape[1]):
#         for lig in range(globals()['matF' + str(h)].shape[0]):
#             # Parcourez les distances par rapport au cluster
#             for i in range(len(distances_intervals)):
#                 distance = np.sqrt((col - cluster_center_x)**2 + (lig - cluster_center_y)**2)
#                 if distance <= distances_intervals[i]:
#                     globals()['matF' + str(h)][lig, col] = temperatures_intervals[i]
#                     break
                
# # Ajout du calcul du gradient autour du cluster
# for h in range(1, nb_cluster+1):
#     for col in range(globals()['matF' + str(h)].shape[1]):
#         for lig in range(globals()['matF' + str(h)].shape[0]):
#             # Trouver les coordonnées du pixel le plus proche du cluster
#             cluster_pixels = np.where(globals()['matF' + str(h)] == 100)
#             min_distance = np.inf
#             nearest_pixel_x, nearest_pixel_y = None, None
        
#             for x, y in zip(cluster_pixels[1], cluster_pixels[0]):
#                 distance = np.sqrt((x - col)**2 + (y - lig)**2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     nearest_pixel_x, nearest_pixel_y = x, y
        
#             # Parcourez les distances par rapport au bord du cluster
#             for i in range(len(distances_intervals)):
#                 distance_from_border = np.sqrt((nearest_pixel_x - col)**2 + (nearest_pixel_y - lig)**2)
#                 if distance_from_border <= distances_intervals[i]:
#                     globals()['matF' + str(h)][lig, col] = temperatures_intervals[i]
#                     break

# #Loop to treat each auxiliary matrix
# for h in range(1, nb_cluster+1) :
#     #Attribuate temperature coefficient in the auxiliary matrix
#     for col in range(globals() ['matF'+ str(h)].shape[1]):
#         for lig in range(globals() ['matF'+ str(h)].shape[0]):
            
#             # Vérifiez si le pixel appartient au cluster
#             if (globals()['matF' + str(h)][lig, col] == 100):
#                 cluster_center_x, cluster_center_y = col, lig
                
#                 # Parcourez les distances par rapport au cluster
#                 for i in range(len(distances_intervals)):
#                     distance = np.sqrt((col - cluster_center_x)**2 + (lig - cluster_center_y)**2)
#                     if distance <= distances_intervals[i]:
#                         globals()['matF' + str(h)][lig, col] = temperatures_intervals[i]
            
#                         if (globals()['matF' + str(h)][lig, col] == 2):
#                             globals()['matF' + str(h)][lig, col] = temperatures_intervals[0]
            
#                         if (globals()['matF' + str(h)][lig, col] == 3):
#                             globals()['matF' + str(h)][lig, col] = temperatures_intervals[1]
            
#                         if (globals()['matF' + str(h)][lig, col] == 4):
#                             globals()['matF' + str(h)][lig, col] = temperatures_intervals[2]
            
#                         if (globals()['matF' + str(h)][lig, col] == 5):
#                             globals()['matF' + str(h)][lig, col] = temperatures_intervals[3]
                            
#                         # Find cluster
#                         if (globals()['matF' + str(h)][lig, col] == 100):
#                             globals()['matF' + str(h)][lig, col] = temperatures_intervals[4]  # /!\ MODIFY

    
    #Load the final temperature matrix 
    matWater +=  globals() ['matF'+ str(h)]

# REMARQUE: Ici notre matrice finale est la somme de toutes les augmentations de température générées
# à voir si on garde ce modèle 

# ... (previous code for plotting, etc.)


#REMARQUE: Ici notre matrice finale est la somme de toute les augmentations de température générées
#à voir si on garde ce modèle 

# print('Final temperature matrix : ')
# print(matTemp)




#Plot the temperature area 
#fig, ax = plt.subplots()
#ax.matshow(matF, cmap=plt.cm.Greens) #shade of red according to the temperature

fig, ax = plt.subplots()
ax.imshow(mat, cmap=plt.cm.Greens)
ax.axis('off')  # Désactive l'affichage des axes et des étiquettes

plt.show()
#Save the matrix of road's temperature
rel_path = os.path.join("..", "ToulouseFinal/tempApresGradient.csv")
file_path = os.path.join(root, rel_path)
pd.DataFrame(mat).to_csv(file_path, index=False, header=False)


# fig, ax = plt.subplots()

# min_val, max_val = 0, 10

# intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

# ax.matshow(intersection_matrix, cmap=plt.


