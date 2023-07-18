# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:41:04 2023

@author: Ghis
"""

import csv

# Chemin du fichier CSV
chemin_fichier = "C:/FichierFinal.csv"

# Compteur pour le nombre d'occurrences
compteur = 0

# Lecture du fichier CSV
with open(chemin_fichier, "r") as fichier:
    lecteur_csv = csv.reader(fichier)
    
    # Parcours des lignes du fichier
    for ligne in lecteur_csv:
        # Parcours des valeurs dans chaque ligne
        for valeur in ligne:
            # Conversion de la valeur en nombre flottant
            try:
                nombre = float(valeur)
                # Vérification si le nombre est égal à 2.0
                if nombre == 0.0:
                    compteur += 1
            except ValueError:
                pass

# Affichage du résultat
print("Le nombre others est", compteur)
