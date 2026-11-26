#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import scipy
import scipy.stats
import math

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
# Ouverture du ficher csv
Temperature = ouvrirUnFichier("data/temperature.csv")
print(Temperature.head())

# Question 1 - Premier tiret
print("Question 1 - Premier tiret")
# Calcul de la corrélation entre les quatre variables
Tableau = Temperature.drop(columns=["Ville"])
print(Tableau.head())
Correlation = Tableau.corr()
print(Correlation)

# Question 1 - Deuxième tiret
print("Question 1 - Deuxième tiret")
# Affichage des statistiques descriptives
Statistique = Tableau.describe()
print(Statistique)

# Question 1 - Troisième tiret
print("Question 1 - Troisième tiret")
# Isolation du champ Ville et isolation des données
Ville = Temperature["Ville"]
Y = Temperature["Température_en_janvier"]
X = Temperature[["Latitude","Longitude","Altitude"]]

# Question 1 - Quatrième tiret
print("Question 1 - Quatrième tiret")
# calcul de la régression linéaire liant la variable à expliquer avec les variables explicatives
X_constant = sm.add_constant(X)
Modèle = OLS(Y, X_constant)
Resultat = Modèle.fit()

# Question 1 - Cinquième tiret
print("Question 1 - Cinquième tiret")
# Coefficients de la régression
print(Resultat.params)

# Coefficient de détermination
print(Resultat.rsquared)

# Résumé de la régression
print(Resultat.summary())

# p-value
print(Resultat.pvalues)

# Question 1 - Sixième tiret
print("Question 1 - Sixième tiret")
# Isolation des coefficients
Resultat2 = LinearRegression().fit(X,Y)
print(Resultat2.coef_)
print(Resultat2.intercept_)

# Question 2
print("Question 2")
# Ouverture du ficher csv
Geomarketing = ouvrirUnFichier("data/geomarketing.csv")
print(Geomarketing.head())

# Question 2 - Premier tiret
print("Question 2 - Premier tiret")
# Isolation de ca et isolation des neuf variables significatives.
Y2 = Geomarketing["ca"]
X2 = Geomarketing[["surface_totale","potentiel_Z20","nb_primaire_Z10","nb_primaire_Z20","nb_gsa_Z10","nb_pharmacie_Z5","nb_conc2_Z10","nb_conc2_Z20","P10_POP_Z15","P10_MEN_Z10"]]

# Question 2 - Deuxième tiret
print("Question 2 - Deuxième tiret")
# calcul de la régression linéaire liant la variable à expliquer avec les variables explicatives
X2_constant = sm.add_constant(X2)
Modèle = OLS(Y2, X2_constant)
Resultat2 = Modèle.fit()

# Coefficients de la régression
print(Resultat2.params)

# Coefficient de détermination
print(Resultat2.rsquared)

# Résumé de la régression
print(Resultat2.summary())

# p-value
print(Resultat2.pvalues)

# Question Bonus
print("Question Bonus")
Y3 = Geomarketing["ca"]
X3 = Geomarketing.drop(columns=["ca","enseigne"])

# calcul de la régression linéaire liant la variable à expliquer avec les variables explicatives
X3_constant = sm.add_constant(X3)
Modèle = OLS(Y3, X3_constant)
Resultat3 = Modèle.fit()

# Coefficients de la régression
print(Resultat3.params)

# Coefficient de détermination
print(Resultat3.rsquared)

# Résumé de la régression
print(Resultat3.summary())

# p-value
print(Resultat3.pvalues)


