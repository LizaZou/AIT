import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import csv
from PIL import Image, ImageDraw
# Chemins des fichiers CSV
root = os.path.dirname(__file__)
rel_path = os.path.join("..", "map_maison_Tl2.csv")
file1_path = os.path.join(root, rel_path)

rel_path = os.path.join("..", "map_Arbre_GN9_Tl2.csv")
file2_path = os.path.join(root, rel_path)

rel_path = os.path.join("..", "map_Road_tle2.csv")
file3_path = os.path.join(root, rel_path)


# # Charger les fichiers CSV en DataFrames
# df1 = pd.read_csv(file1_path)
# df2 = pd.read_csv(file2_path)
# df3 = pd.read_csv(file3_path)

# # Fusionner les trois DataFrames
# merged_df = df1.copy()  # Créer une copie du premier DataFrame

# # Remplacer les valeurs de 1 par 2 dans merged_df pour les positions où df2 a des 1
# merged_df[df2 == 0] = 2

# # Remplacer les valeurs de 1 par 3 dans merged_df pour les positions où df3 a des 1
# merged_df[df3 == 0] = 3

# # Sauvegarder le DataFrame fusionné en tant que fichier CSV
# rel_path = os.path.join("..", "/../../../../../TESTFINAL2.csv")                            A VERIFIER
# file_path = os.path.join(root, rel_path)
# merged_df.to_csv(file_path, index=False)


# def afficher_image_from_csv(csv_path):
#     # Charger le fichier CSV en tant que DataFrame
#     df = pd.read_csv(csv_path)

#     # Récupérer les valeurs de l'image sous forme de tableau numpy
#     image = df.values

#     # Créer une colormap personnalisée avec les couleurs souhaitées
#     cmap = ListedColormap(['red', 'black', 'green', 'yellow'])

#     # Définir les limites des catégories
#     bounds = [0, 1, 2, 3, 4]

#     # Créer une liste des étiquettes de catégories correspondant aux limites
#     labels = ['Maison', 'Others', 'Arbre', 'Route']

#     # Créer une norme pour associer les valeurs aux couleurs
#     norm = plt.Normalize(vmin=0, vmax=4)

#     # Afficher l'image avec les couleurs personnalisées
#     plt.imshow(image, cmap=cmap, norm=norm)
#     #plt.colorbar(ticks=bounds, label='Catégorie')
#     #plt.clim(0, 4)  # Définir les limites de la colorbar
#     plt.xticks([])  # Masquer les graduations de l'axe x
#     plt.yticks([])  # Masquer les graduations de l'axe y
#     plt.show()

    
# afficher_image_from_csv(file_path)



# # # Chemins des images d'entrée
# # chemin_image1 = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/Assemblage.png"
# # chemin_image2 = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/Toulouse22.png"

# # # Ouvrir les images
# # image1 = Image.open(chemin_image1)
# # image2 = Image.open(chemin_image2)

# # # Vérifier si les dimensions des images sont différentes
# # if image1.size != image2.size:
# #     # Ajuster les dimensions des images pour les rendre identiques
# #     largeur_max = max(image1.width, image2.width)
# #     hauteur_max = max(image1.height, image2.height)

# #     image1 = image1.resize((largeur_max, hauteur_max))
# #     image2 = image2.resize((largeur_max, hauteur_max))

# # # Créer une nouvelle image qui contient les deux images d'entrée
# # nouvelle_image = Image.new("RGB", (image1.width * 2, image1.height))
# # nouvelle_image.paste(image1, (0, 0))
# # nouvelle_image.paste(image2, (image1.width, 0))

# # # Sauvegarder l'image résultante
# # #chemin_image_resultante = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/Assemblee.png"
# # #nouvelle_image.save(chemin_image_resultante)

# # #print("L'assemblage des images est terminé. L'image résultante est enregistrée à :", chemin_image_resultante)





# # Chemins des images d'entrée
# rel_path = os.path.join("..", "Image result/FusionArbreRoute2.png")
# chemin_image11 = os.path.join(root, rel_path)

# rel_path = os.path.join("..", "Image result/Maisontest3.png")
# chemin_image22 = os.path.join(root, rel_path) 

# # Ouvrir les images
# image11 = Image.open(chemin_image11)
# image22 = Image.open(chemin_image22)

# # Ajuster les dimensions des images pour les rendre identiques
# largeur_max = max(image11.width, image22.width)
# hauteur_max = max(image11.height, image22.height)

# image11 = image11.resize((largeur_max, hauteur_max))
# image22 = image22.resize((largeur_max, hauteur_max))

# # Appliquer le filtre de chaleur sur l'image 2 avec un facteur d'opacité
# facteur_opacite = 0.5  # Modifier selon les besoins
# image_fusionnee = Image.blend(image22, image11, facteur_opacite)

# # Sauvegarder l'image résultante
# rel_path = os.path.join("..", "Image result/FusionArbreRoute2Maison.png")
# chemin_image_resultante = os.path.join(root, rel_path)
# image_fusionnee.save(chemin_image_resultante)

# print("La fusion des images est terminée. L'image résultante est enregistrée à :", chemin_image_resultante)
# Chemins des fichiers CSV
rel_path = os.path.join("..", "Image result/MATFMaison.csv")
file1_path = os.path.join(root, rel_path)
chemin_maison = file1_path

rel_path = os.path.join("..", "matFARBRE0507.csv")
file1_path = os.path.join(root, rel_path)
chemin_arbre = file1_path

rel_path = os.path.join("..", "Image result/MatFRoute0407.csv")
file1_path = os.path.join(root, rel_path)
chemin_route = file1_path

# Chargement des fichiers CSV dans des DataFrames
df_maison = pd.read_csv(chemin_maison)
df_arbre = pd.read_csv(chemin_arbre)
df_route = pd.read_csv(chemin_route)

# Assemblage en respectant les priorités
df_final = df_route.copy()  # Copie du DataFrame de routes

# Remplacement des valeurs en respectant les priorités
  # Priorité pour les routes
  # Priorité pour les maisons
df_final = np.where(df_arbre == 1.0, 4.0, df_final) 
df_final = np.where(df_maison == 1.0, 3.0, df_final) 
df_final = np.where(df_route == 1.0, 2.0, df_final)
# Conversion du tableau NumPy en DataFrame Pandas
df_final = pd.DataFrame(df_final)


# Sauvegarde du DataFrame final dans un fichier CSV
rel_path = os.path.join("..", "Image result/FichierFinal6.csv")
file1_path = os.path.join(root, rel_path)
chemin_final = file1_path

df_final.to_csv(chemin_final, index=False)

print("Fichier final créé avec succès :", chemin_final)

#compteur = 0

# Lecture du fichier CSV
# with open(chemin_final, "r") as fichier:
#     lecteur_csv = csv.reader(fichier)

#     # Parcours des lignes du fichier
#     for ligne in lecteur_csv:
#         # Parcours des valeurs dans chaque ligne
#         for valeur in ligne:
#             # Conversion de la valeur en nombre flottant
#             try:
#                 nombre = float(valeur)
#                 # Vérification si le nombre est égal à 2.0
#                 if nombre == 2.0:
#                     compteur += 1
#             except ValueError:
#                 pass



# Chemin du fichier CSV

rel_path = os.path.join("..", "Image result/FichierFinal6.csv")
file1_path = os.path.join(root, rel_path)
chemin_final2 = file1_path

# Lecture du fichier CSV et création de la matrice
matrice = []
with open(chemin_final2, 'r') as fichier:
    lecteur_csv = csv.reader(fichier)
    for ligne in lecteur_csv:
        matrice.append([float(valeur) for valeur in ligne])

# Dimensions de la matrice
largeur = len(matrice[0])
hauteur = len(matrice)

# Création de l'image
image = Image.new('RGB', (largeur, hauteur))

# Parcours de la matrice pour affecter les couleurs aux pixels
for y in range(hauteur):
    for x in range(largeur):
        valeur = matrice[y][x]
        if valeur == 0.0:
            voisins = []
            rayon = 1
            while not voisins:
                for j in range(y - rayon, y + rayon + 1):
                    for i in range(x - rayon, x + rayon + 1):
                        if (i != x or j != y) and 0 <= i < largeur and 0 <= j < hauteur and matrice[j][i] != 0.0:
                            voisins.append(matrice[j][i])
                rayon += 1
        
            if voisins:
                valeur = max(set(voisins), key=voisins.count)
            else:
                valeur = 2.0  # Par défaut, remplacer les pixels sans voisins par 2.0
            
            print("Résultat enregistré:")
        if valeur == 2.0:
            couleur = (128, 128, 128)  # Gris
        elif valeur == 3.0:
            couleur = (255, 0, 0)  # Rouge
        elif valeur == 4.0:
            couleur = (0, 255, 0)  # Vert
        else:
            couleur = (0, 0, 0)  # Noir par défaut
        
        image.putpixel((x, y), couleur)


# Chemin du fichier à traiter
rel_path3 = os.path.join("..", "Image result", "FICHIERREMPLISSAGE5.csv")
file_path3 = os.path.join(root, rel_path3)
pd.DataFrame(matrice).to_csv(file_path3, index=False, header=False)
print("SALUT JE SUIS BIEN ARRIVE ICI")

plt.imshow(image)
plt.axis('off')  # Masquer les axes
plt.show()


# # Chemin des fichiers de sortie
# output_folder = os.path.join("C:\\Users\\Ghis\\Onedrive\\Bureau\\PostProcessing", "Image result")
# output_file_2 = os.path.join(output_folder, "FICHIERREMPLISSAGE3_2.csv")
# output_file_3 = os.path.join(output_folder, "FICHIERREMPLISSAGE3_3.csv")
# output_file_4 = os.path.join(output_folder, "FICHIERREMPLISSAGE3_4.csv")

# # Lecture du fichier et écriture dans les fichiers de sortie
# with open(file_path3, 'r') as file, \
#      open(output_file_2, 'w', newline='') as file_2, \
#      open(output_file_3, 'w', newline='') as file_3, \
#      open(output_file_4, 'w', newline='') as file_4:
    
#     reader = csv.reader(file)
#     writer_2 = csv.writer(file_2)
#     writer_3 = csv.writer(file_3)
#     writer_4 = csv.writer(file_4)
    
#     for row in reader:
#         new_row_2 = []
#         new_row_3 = []
#         new_row_4 = []
        
#         for value in row:
#             if value == '2.0':
#                 new_row_2.append('1.0')
#             else:
#                 new_row_2.append('0.0')
            
#             if value == '3.0':
#                 new_row_3.append('1.0')
#             else:
#                 new_row_3.append('0.0')
            
#             if value == '4.0':
#                 new_row_4.append('1.0')
#             else:
#                 new_row_4.append('0.0')
        
#         writer_2.writerow(new_row_2)
#         writer_3.writerow(new_row_3)
#         writer_4.writerow(new_row_4)

# print("Fichiers CSV créés avec succès.")



# # Chemin du fichier final CSV
# rel_path = os.path.join("..", "Image result/FichierFinal4.csv")
# file1_path = os.path.join(root, rel_path)
# chemin_final2 = file1_path

# # Chargement du fichier CSV dans un DataFrame
# df_final2 = pd.read_csv(chemin_final2)

# # Conversion du DataFrame en tableau numpy
# matrice = df_final2.values

# cmap = plt.cm.colors.ListedColormap(['yellow', 'gray', 'red', 'green'])
# bounds = [0.0, 2.0, 3.0, 4.0]
# norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# # Tracé de l'image sans les axes à côté
# plt.imshow(matrice, cmap=cmap, norm=norm)
# plt.axis('off')  # Désactiver les axes

# # Ajouter la barre de couleur
# #plt.colorbar(ticks=[0.0, 2.0, 3.0, 4.0], boundaries=bounds)

# # Ajouter le titre et les étiquettes des axes
# #plt.title("Assemblage des fichiers CSV")
# plt.xlabel("Colonnes")
# plt.ylabel("Lignes")

# # Afficher l'image
# plt.show()







