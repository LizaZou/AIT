# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:08:10 2023

@author: Ghis
"""
import os
import pandas as pd
#from playsound import playsound

# Chemin absolu du fichier à traiter
file_path = os.path.abspath("../Ghis/OneDrive/Bureau/PostProcessing/Image result/FICHIERREMPLISSAGE5.csv")

# Charger le fichier CSV
df = pd.read_csv(file_path, header=None)

# Créer une nouvelle DataFrame avec des valeurs égales à 1.0 pour les valeurs égales à 3.0 dans le fichier CSV d'origine
df_new = df.copy()
df_new[df_new == 0.0] = 1.0
df_new[df_new == 2.0] = 1.0
df_new[df_new == 3.0] = 1.0
df_new[df_new == 4.0] = 0.0

# Chemin absolu du nouveau fichier CSV
new_file_path = os.path.abspath("../Ghis/OneDrive/Bureau/PostProcessing/Image result/ARBREFINALINCH2.csv")

# Sauvegarder la nouvelle DataFrame dans un nouveau fichier CSV
df_new.to_csv(new_file_path, index=False, header=False)

print("Le fichier ARBREFINALINCH.csv a été créé avec succès.")



