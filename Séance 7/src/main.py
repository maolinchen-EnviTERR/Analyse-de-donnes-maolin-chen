#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

data = pd.DataFrame(ouvrirUnFichier("data/France-Granulats-2006-2020-par-region.csv"))
# print(data)

# Création du tableau de contingence
# Contrairement à l'usage, vous ne devez pas créer de tableau croisé dynamique, puisque le fichier est déjà un tableau de contingence
tableauDeContingence = tableauDeContingence(data["Région"], {"2006": data["2006"], "2007": data["2007"], "2008": data["2008"], "2009": data["2009"], "2010": data["2010"], "2011": data["2011"], "2012": data["2012"], "2013": data["2013"], "2014": data["2014"], "2015": data["2015"], "2016": data["2016"], "2017": data["2017"], "2018": data["2018"], "2019": data["2019"], "2020": data["2020"]})
print(tableauDeContingence)

# Question 1
print("Question 1")
# Calculer les marges
# Calcul des marges en ligne
Marges_lignes = sommeDesLignes(tableauDeContingence)
print(Marges_lignes)

# Calcul des marges en colonne
Marges_colonnes = sommeDesColonnes(tableauDeContingence)
print(Marges_colonnes)

# Question 2
print("Question 2")
# Vérification de l’égalité entre le total des marges des lignes et celui des colonnes
Total_marges_lignes = sum(Marges_lignes)
Total_marges_colonnes = sum(Marges_colonnes)
print(Total_marges_lignes)
print(Total_marges_colonnes)
if Total_marges_colonnes == Total_marges_lignes:
    print("Le total des marges des lignes et le total des marges des colonnes est identique")
else:
    print("Le total des marges des lignes et le total des marges des colonnes n'est pas identique")

# Question 3
print("Question 3")
# Faire le test du chi2 avec les outils Scipy.stats
from scipy.stats import chi2_contingency 
Test_chi2 = chi2_contingency(tableauDeContingence)
print("chi2:", Test_chi2[0])
print("pvalue:", Test_chi2[1])
print("ddl:", Test_chi2[2])
print("effectifs_theoriques:", Test_chi2[3])

# Question 4
print("Question 4")
# Calcul de l'effectif total
N = sum(np.sum(tableauDeContingence))
print("N :",N)

# Calcul de  l'intensité de liaison φ2 de Pearson
chi2 = Test_chi2[0]
φ2 = chi2/N
print("φ2 :",φ2)

# Question bonus
print("Question bonus")
# Ouverture du ficher csv
Echantillons = pd.read_csv("data/Echantillonnage-100-Echantillons.csv")
print(Echantillons.head())

# Sélection des colonnes
Groupe1 = Echantillons["Pour"]
Groupe2 = Echantillons["Contre"]
Groupe3 = Echantillons["Sans opinion"]

# Analyse ANOVA
from scipy.stats import f_oneway
F,Pvalue = f_oneway(Groupe1,Groupe2,Groupe3)
print("F :",F)
print("Pvalue :",Pvalue)