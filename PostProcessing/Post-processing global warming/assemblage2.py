import pandas as pd
import os

# Chemins des fichiers CSV
root = os.path.dirname(__file__)
rel_path = os.path.join("..", "accura_arbre_Tl3.csv")
arbre_file = os.path.join(root, rel_path)

rel_path = os.path.join("..", "accura_maison_Tl3.csv")
maison_file = os.path.join(root, rel_path)

rel_path = os.path.join("..", "accura_route_Tl3.csv")
route_file = os.path.join(root, rel_path)

# Chargement des fichiers CSV en tant que DataFrames
df_arbre = pd.read_csv(arbre_file, header=None)
df_maison = pd.read_csv(maison_file, header=None)
df_route = pd.read_csv(route_file, header=None)

# Création d'un nouveau DataFrame pour le fichier combiné
df_combined = pd.DataFrame(index=df_arbre.index, columns=df_arbre.columns)

# Comparaison des valeurs et attribution des codes correspondants
for i in range(len(df_arbre.index)):
    for j in range(len(df_arbre.columns)):
        arbre_val = df_arbre.iloc[i, j]
        maison_val = df_maison.iloc[i, j]
        route_val = df_route.iloc[i, j]
        
        # Règles de priorité
        if arbre_val > maison_val and arbre_val > route_val:
            df_combined.iloc[i, j] = 2
        elif maison_val > arbre_val and maison_val > route_val:
            df_combined.iloc[i, j] = 0
        elif route_val > arbre_val and route_val > maison_val:
            df_combined.iloc[i, j] = 3
        else:
            # Valeurs identiques, vérification des voisins
            neighbors = [arbre_val, maison_val, route_val]
            most_common = max(set(neighbors), key=neighbors.count)
            df_combined.iloc[i, j] = most_common

# Chemin du nouveau fichier CSV combiné
rel_path = os.path.join("..", "combined.csv")
combined_file = os.path.join(root, rel_path)

# Enregistrement du DataFrame combiné en tant que fichier CSV
df_combined.to_csv(combined_file, index=False, header=False)

print("Le fichier combiné a été créé :", combined_file)
