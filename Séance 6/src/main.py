#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
# Visualisation des données
# Pour résoudre le problème posé dans la question 1, il faut utiliser l'opérateur NaN de la bibliothèque Numpy dans une boucle

# Ouverture du ficher csv
Fichier = ouvrirUnFichier("data/pib-vs-energie.csv")
print(Fichier.head())

# Sélection des colonnes "PIB_2022" et "Utilisation_d_energie_2022"
PIB_2022 = list(Fichier["PIB_2022"])
Energie_2022 = list(Fichier["Utilisation_d_energie_2022"])
print(PIB_2022)
print(Energie_2022)

# Question 2
print("Question 2")
# Sélection des couples complets
PIB_Complet = []
Energie_Complet = []
for i in range(0,len(PIB_2022)):
    if not np.isnan(PIB_2022[i]) and not np.isnan(Energie_2022[i]):
        PIB_Complet.append(PIB_2022[i])
        Energie_Complet.append(Energie_2022[i])

# Question 3
print("Question 3")
# Calcul de la régression linéaire simple entre les deux colonnes
Régression = scipy.stats.linregress(Energie_Complet, PIB_Complet)
print(Régression)

# Question 4
print("Question 4")
# Calcul de la corrélation
Corrélation = scipy.stats.pearsonr(Energie_Complet, PIB_Complet)
print(Corrélation)

# Question 5
print("Question 5")

# Calcul des coordonnées(x1,y1)et(y1,y2)
x1 = min(Energie_Complet)
x2 = max(Energie_Complet)
y1 = Régression.slope*x1 + Régression.intercept
y2 = Régression.slope*x2 + Régression.intercept

# Création du graphique
fig, plot = plt.subplots()

# Création des nuages de points
plot.scatter(Energie_Complet, PIB_Complet, color="blue")

# Création de la droite de la régression linéaire
plot.plot([x1, x2], [y1, y2], color="red")
plot.set_title("Relation entre le PIB et la consommation énergétique en 2022", pad = 20)
plot.set_xlabel("Consommation énergétique")
plot.set_ylabel("PIB")
plt.savefig("Relation entre le PIB et la consommation énergétique en 2022.png")

# Bonus
print("Bonus")

# Généralisation de l’algorithme à l’ensemble des années de 1962 à 2022 et organisation des fichiers de sortie
Résultat = []
for année in range(1962,2023):
    PIB = list(pd.to_numeric(Fichier["PIB_"+str(année)], errors="coerce"))
    Energie = list(pd.to_numeric(Fichier["Utilisation_d_energie_"+str(année)], errors="coerce"))
    PIB_complet = []
    Energie_complet = []
    for i in range(0,len(PIB)):
        if not np.isnan(PIB[i]) and not np.isnan(Energie[i]):
            PIB_complet.append(PIB[i])
            Energie_complet.append(Energie[i])
    Régression = scipy.stats.linregress(Energie_complet, PIB_complet)
    Corrélation = scipy.stats.pearsonr(Energie_complet, PIB_complet)
    x1 = min(Energie_Complet)
    x2 = max(Energie_Complet)
    y1 = Régression.slope*x1 + Régression.intercept
    y2 = Régression.slope*x2 + Régression.intercept
    fig, plot = plt.subplots()
    plot.scatter(Energie_complet, PIB_complet, color="blue")
    plot.plot([x1, x2], [y1, y2], color="red")
    plot.set_title("Relation entre le PIB et la consommation énergétique en "+str(année), pad = 20)
    plot.set_xlabel("Consommation énergétique")
    plot.set_ylabel("PIB")
    plt.savefig("Relation entre le PIB et la consommation énergétique 1962-2022/Relation entre le PIB et la consommation énergétique en " + str(année) + ".png")
    plt.close(fig)
    Résultat.append([année, Régression.slope,Régression.intercept,Régression.rvalue,Régression.pvalue,Régression.stderr])
Tableau = pd.DataFrame(Résultat,columns=["Année","Slope", "Intercept", "Rvalue", "Pvalue", "Stderr"])
Tableau.to_csv("Resultats_Bonus.csv", index=False)