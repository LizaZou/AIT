#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:15:36 2022

@author: katalizfol
"""
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


##Open the class matrix
root = os.path.dirname(__file__)
rel_path = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing\\Image result", "ROUTEFINALINCH.csv")
file_path = os.path.join(root, rel_path)
mat = open(file_path)
mat=np.loadtxt(mat,delimiter=",")
mat[np.where(mat==0)]=2
mat[np.where(mat==1)]=0
mat[np.where(mat==2)]=1

# CLEAN SMALL CLUSTER

#Detect clusters
lw, num = measurements.label(mat) #num: nb initial of cluster 
                                  #lw: matrix of cluster (each one has its own number)

#Threshold size of cluster to keep 
seuil =  25  # /!\ MODIFY

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

#Detect others-clusters
matF[np.where(matF==0)]=2
matF[np.where(matF==1)]=0
matF[np.where(matF==2)]=1
lw_oth, num_oth = measurements.label(matF)

#Threshold size of others-cluster to keep 
seuil_oth = 100 # /!\ MODIFY

#Loop to treat each others-cluster
for k in range(1,num_oth+1):
    
    #Matrix of boolean -> true if it's the k-cluster
    classK_oth = (lw_oth==k)
    
    #Coordinate of the k-cluster
    x,y = np.where(classK_oth)
    
    #Coordinate of the center of the k-cluster
    a,b = measurements.center_of_mass(classK_oth)
    #Determination of the radius of the k-cluster
    xx = x-a
    yy = y-b
    #Using Pythagore to find the maximum distance
    sqrx = np.square(xx)
    sqry = np.square(yy)
    maxi = max(np.sqrt((sqrx+sqry))) #maxi : radius of the k-cluster
    
    #Delete the small others-cluster if radius of the cluster is smaller than the threshold
    if (maxi <= seuil_oth) :

        #Load the clean small cluster matrix 
        for h in range(len(x)) :
            matF[x[h],y[h]] = 0
            
matF[np.where(matF==0)]=2
matF[np.where(matF==1)]=0
matF[np.where(matF==2)]=1            
            
# #Update of the Output data with only the clusters to keep     
# for l in range(0, (num-nb_cluster)):
#     Output = np.delete(Output, num-l-1, 1)
# print('Nb clusters : ' + str(nb_cluster))

#Initialisation final temperature matrix for road 
matRoute= np.zeros(np.shape(mat))

#Change the value of cluster 
for col in range(mat.shape[1]):
    for lig in range(mat.shape[0]):
        
        if (matF[lig,col]==1):
            matRoute[lig,col]= 4
            
#REMARQUE: Dans ce modele, on considére qu'une route donne une augmentation de 2°C sur le cluster 
            
#Plot the temperature area 
 #shade of red according to the temperature
fig, ax = plt.subplots()
ax.imshow(matRoute, cmap=plt.cm.Blues)
ax.axis('off')  # Désactive l'affichage des axes et des étiquettes

plt.show()
#Save the matrix of road's temperature
rel_path = os.path.join("..", "Image result/temperature1107.csv")
file_path = os.path.join(root, rel_path)
pd.DataFrame(matRoute).to_csv(file_path, index=False, header=False)
#plt.savefig("Plot_Rapport.png")

