#coding:utf8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import scipy
import scipy.stats
import prince
import matplotlib.pyplot as plt

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
# Analyse en composantes principales
# Question 1a
print("Question 1a")
Temperature = ouvrirUnFichier("data/france-temperatures.csv")
print(Temperature)

# Question 1b
print("Question 1b")
Variables = Temperature.drop(columns=["Villes"])
print(Variables.head())

# Question 1c
print("Question 1c")
# Centrer-réduire les données numériques
Variables_centre_réduire = StandardScaler().fit_transform(Variables) 
print(Variables_centre_réduire)

# Question 1d
print("Question 1d")
# Faire une A.C.P.
pca = PCA()
Variables_pca = pca.fit(Variables_centre_réduire)
print(Variables_pca)

# Question 1e
print("Question 1e")
# Affichage de la variance expliquée en pourcentage et calcul des valeurs propres
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# Question 1f
print("Question 1f")
# affichage du résultat de l’A.C.P
eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(12)], 
        "Variance expliquée" : pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    }
)
print(eig)

# Question 1g
print("Question 1g")
Temperature_pca = pca.transform(Variables_centre_réduire)
# Transformation en DataFrame pandas
Temperature_pca_df = pd.DataFrame({
    "Dim1" : Temperature_pca[:,0], 
    "Dim2" : Temperature_pca[:,1],
    "Villes" : Temperature["Villes"],
})

# Résultat (premières lignes)
print(Temperature_pca_df.head())

# Image du mapping des individus
fig, ax = plt.subplots()
Temperature_pca_df.plot.scatter("Dim1", "Dim2", ax = ax) 
for k in Temperature_pca_df.iterrows():
    if (abs(k[1]['Dim1']) > 3.5) | (abs(k[1]['Dim2']) > 1.5):
        ax.annotate(k[1]["Villes"], (k[1]['Dim1'], k[1]['Dim2']), fontsize = 9)
plt.xlabel("Dimension 1 (80%)") 
plt.ylabel("Dimension 2 (19 %)")
plt.suptitle("Premier plan factoriel (99%)")
plt.savefig("mapping_individus.png")
plt.show()

# Question 1h
print("Question 1h")
# Contribution des individus
Contribution = Temperature_pca ** 2 / np.sum( Temperature_pca ** 2, axis=0 ) * 100
print(Contribution)

# cos²
Cos2 = Temperature_pca ** 2 / np.sum(Temperature_pca ** 2, axis=1, keepdims=True)
print(Cos2)

# Question 1i
print("Question 1i")
# Calcul des coordonnées des variables et création de l’image du cercle de corrélation
# nb individus
n = Variables.shape[0] 
# nb variables
p = Variables.shape[1] 
# valeurs propres
eigval = (n-1) / n * pca.explained_variance_ 
# racine carrée des valeurs propres
sqrt_eigval = np.sqrt(eigval) 
# matrice vide pour avoir les coordonnées
corvar = np.zeros((p,p)) 
for k in range(p):
    corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
# on modifie pour avoir un dataframe
coordvar = pd.DataFrame({'id': Variables.columns, 'COR_1': corvar[:,0], 'COR_2': corvar[:,1]})
print(coordvar)

# Création d'une figure vide (avec des axes entre -1 et 1 + le titre)
fig, axes = plt.subplots(figsize = (12,12))
fig.suptitle("Cercle des corrélations")
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
# Ajout des axes
axes.axvline(x = 0, color = 'lightgray', linestyle = '--', linewidth = 1)
axes.axhline(y = 0, color = 'lightgray', linestyle = '--', linewidth = 1)
# Ajout des noms des variables
for j in range(p):
    axes.text(coordvar["COR_1"][j],coordvar["COR_2"][j], coordvar["id"][j])
# Ajout du cercle
plt.gca().add_artist(plt.Circle((0,0),1,color='blue',fill=False))
plt.savefig("Cercle des corrélations.png")
plt.show()

# Question 2
print("Question 2")
# Analyse factorielle de correspondances multiples
# Question 2a
print("Question 2a")
chiens = ouvrirUnFichier("data/chiens.csv")
print(chiens)

# Question 2b
print("Question 2b")
# Isolation de la colonne Race et Isolation des autres variables
Race = chiens["Race"]
Tableau_variables = chiens.drop(columns=["Race"])
print(Race.head())
print(Tableau_variables)

# Question 2c
print("Question 2c")
# Transformation du tableau des variables en tableau disjonctif complet
tableau_disjonctif_complet = pd.get_dummies(Tableau_variables)
print(tableau_disjonctif_complet.head())

# Question 2d
print("Question 2d")
# Calcul de l’A.C.M. à huit facteurs
ACM = prince.MCA(n_components=8).fit(tableau_disjonctif_complet)
print (ACM)

# Question 2e
print("Question 2e")
# Calcul des valeurs propres
print(ACM.eigenvalues_)

# Question 2f
print("Question 2f")
# Calcul des coordonnées des lignes et des colonnes des variables.
chiens_acm = ACM.transform(tableau_disjonctif_complet)

# Question 2g
print("Question 2g")
# Transformation en DataFrame pandas
chiens_acm_df = pd.DataFrame({
    "Dim1" : chiens_acm.iloc[:,0], 
    "Dim2" : chiens_acm.iloc[:,1],
    "Race" : chiens["Race"],
})

# Création de l’image du mapping des deux premiers facteurs
fig, ax = plt.subplots()
chiens_acm_df.plot.scatter("Dim1", "Dim2", ax = ax) 
for k in chiens_acm_df.iterrows():
        ax.annotate(k[1]["Race"], (k[1]['Dim1'], k[1]['Dim2']), fontsize = 9)
plt.xlabel("Dimension 1") 
plt.ylabel("Dimension 2")
plt.suptitle("Premier plan factoriel_ACM")
plt.savefig("mapping_individus_ACM.png")
plt.show()

# Question 2h
print("Question 2h")

# Bonus
from scipy.cluster.hierarchy import linkage, dendrogram
CAH = linkage(Temperature_pca, method="ward")
dendrogram(CAH, labels=Temperature_pca_df["Villes"].values)
fig.suptitle("Dendrogramme_CAH")
plt.xlabel("Villes") 
plt.ylabel("Distances")
plt.savefig("Dendrogramme_CAH.png")
plt.show()

