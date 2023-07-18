import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import csv

plt.close("all")
# Chemins des fichiers CSV
# file1_path = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/map_maison_Tl2.csv"
# file2_path = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/map_Arbre_GN9_Tl2.csv"
# file3_path = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/map_Road_tle2.csv"

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
# merged_df.to_csv("C:/TESTFINAL2.csv", index=False)


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

    
# afficher_image_from_csv("C:/TESTFINAL2.csv")



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
# chemin_image11 = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/FusionArbreRoute2.png"
# chemin_image22 = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/Maisontest3.png"

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
# chemin_image_resultante = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/FusionArbreRoute2Maison.png"
# image_fusionnee.save(chemin_image_resultante)

# print("La fusion des images est terminée. L'image résultante est enregistrée à :", chemin_image_resultante)




# Chemins des fichiers CSV
chemin_maison = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/MATFMaison.csv"
chemin_arbre = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/matFArbre.csv"
chemin_route = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/MatFRoute0407.csv"

# Chargement des fichiers CSV dans des DataFrames
df_maison = pd.read_csv(chemin_maison)
df_arbre = pd.read_csv(chemin_arbre)
df_route = pd.read_csv(chemin_route)

# Assemblage en respectant les priorités
df_final = df_route.copy()  # Copie du DataFrame de routes

# Remplacement des valeurs en respectant les priorités
  # Priorité pour les routes
df_final = np.where(df_maison == 1.0, 3.0, df_final)  # Priorité pour les maisons
df_final = np.where(df_arbre == 1.0, 4.0, df_final)  # Priorité pour les arbres
df_final = np.where(df_route == 1.0, 2.0, df_final)
# Conversion du tableau NumPy en DataFrame Pandas
df_final = pd.DataFrame(df_final)

# Sauvegarde du DataFrame final dans un fichier CSV
chemin_final = "C:/Users/Ghis/OneDrive/Bureau/PostProcessing/Image result/FichierFinal3.csv"
df_final.to_csv(chemin_final, index=False)

print("Fichier final créé avec succès :", chemin_final)

compteur = 0

# Lecture du fichier CSV
with open(chemin_final, "r") as fichier:
    lecteur_csv = csv.reader(fichier)

    # Parcours des lignes du fichier
    for ligne in lecteur_csv:
        # Parcours des valeurs dans chaque ligne
        for valeur in ligne:
            # Conversion de la valeur en nombre flottant
            try:
                nombre = float(valeur)
                # Vérification si le nombre est égal à 2.0
                if nombre == 4.0:
                    compteur += 1
            except ValueError:
                pass

print("Le nombre de routes final est :", compteur)


# # Chemin du fichier final CSV
# chemin_final = "C:/FichierFinal.csv"

# # Chargement du fichier CSV dans un DataFrame
# df_final = pd.read_csv(chemin_final)

# # Conversion du DataFrame en tableau numpy
# matrice = df_final.values

# # Définition des couleurs
# cmap = plt.cm.colors.ListedColormap(['yellow', 'blue','red','green'])
# bounds = [0.0, 1.0, 2.0, 3.0,   4.0]
# norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# # Tracé de l'image
# plt.imshow(matrice, cmap=cmap, norm=norm)
# plt.colorbar(ticks=[1.0, 2.0, 3.0, 4.0], boundaries=bounds)
# plt.title("Assemblage des fichiers CSV")
# plt.xlabel("Colonnes")
# plt.ylabel("Lignes")
# plt.show()
