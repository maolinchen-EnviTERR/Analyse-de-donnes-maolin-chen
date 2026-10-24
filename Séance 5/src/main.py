#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding='utf-8') as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element][1]])
    return classement

# Question 2
print("Question 2")
# Ouverture du ficher csv
island = ouvrirUnFichier("data/island-index.csv")
print(island.head())

# Question 3
print("Question 3")
# Extraction de la colonne "surface" et ajout à cette liste
surface = [85545323.0, 37856841.0, 7768030.0, 7605049.0]
for element in island["Surface (km²)"]:
    surface.append(float(element))

# Question 4
print("Question 4")
# Tri de la liste obtenue avec la fonction locale ordreDecroissant() proposée
surface = ordreDecroissant(surface)

# Question 5
print("Question 5")
# Visualisation de la loi rang-taille
rang = []
for element in range(1, len(surface)+1):
    rang = rang + [element]

x = rang
y = surface
fig, plot = plt.subplots()
plot.bar(x, y)
plot.set_title("Visualisation de la loi rang-taille")
plot.set_xlabel("Rang")
plot.set_ylabel("Surface")
plt.savefig("Visualisation de la loi rang-taille-1.png")

# Question 6
print("Question 6")
# Visualisation de la loi rang-taille en logarithme
surface = conversionLog(surface)
rang = conversionLog(rang)
x = rang
y = surface
fig, plot = plt.subplots()
plot.bar(x, y)
plot.set_title("Visualisation de la loi rang-taille en logarithme")
plot.set_xlabel("log(Rang)")
plot.set_ylabel("log(Surface)")
plt.savefig("Visualisation de la loi rang-taille-2.png")

# Question 7
print("Question 7")
# Mettre ici votre réponse argumentée
print("No, le test de rang a besoin au moins de 2 différents classemnts, mais on a seulement un classement selon la surface.")

# Question 9
print("Question 9")
# Ouverture du ficher csv
monde = ouvrirUnFichier("data/Le-Monde-HS-Etats-du-monde-2007-2026.csv") 
print(monde.head())

# Question 10
print("Question 10")
# Extraction des colonnes
État = list(monde["État"])
Pop_2007 = list(pd.to_numeric(monde["Pop 2007"], errors="coerce"))
Pop_2026 = list(pd.to_numeric(monde["Pop 2026"], errors="coerce"))
Densité_2007 = list(pd.to_numeric(monde["Densité 2007"], errors="coerce"))
Densité_2026 = list(pd.to_numeric(monde["Densité 2026"], errors="coerce"))

# Question 11
print("Question 11")
# Tri des listes de manière décroissante avec la fonction locale ordrePopulation()
ordrePopulation_2007 = ordrePopulation(Pop_2007, État)
ordrePopulation_2026 = ordrePopulation(Pop_2026, État)
ordreDensité_2007 = ordrePopulation(Densité_2007, État)
ordreDensité_2026 = ordrePopulation(Densité_2026, État)

# Question 12
print("Question 12")
# Préparation de la comparaison des listes sur la population et la densité
classementPays_2007 = classementPays(ordrePopulation_2007, ordreDensité_2007)
classementPays_2026 = classementPays(ordrePopulation_2026, ordreDensité_2026)
classementPays_2007.sort()
classementPays_2026.sort()

# Question 13 
print("Question 13")
# Extraction des deux colonnes sous la forme de liste différentes en utilisant une boucle
OrdrePopulation_2007 = []
OrdreDensité_2007 = []
for element in classementPays_2007:
    OrdrePopulation_2007.append(element[0])
    OrdreDensité_2007.append(element[1])
print(OrdrePopulation_2007[:5])
print(OrdreDensité_2007[:5])

# Question 14
print("Question 14")
# Coefficient de corrélation et Concordance des rangs
from scipy.stats import spearmanr,kendalltau
spearmanr_2007 = spearmanr(OrdrePopulation_2007,OrdreDensité_2007)
print(spearmanr_2007)
kendalltau_2007 = kendalltau(OrdrePopulation_2007,OrdreDensité_2007)
print(kendalltau_2007)

# Bonus
print("Bonus")
# Extraction de la colonne des traits de côte
Traits = []
Surface = []
for element in island["Trait de côte (km)"]:
    Traits.append(float(element))
for element in island["Surface (km²)"]:
    Surface.append(float(element))

# Tri des listes de manière décroissante avec la fonction locale ordrePopulation()
Identifiant = list(range(1,len(island)+1))
ordre_surface = ordrePopulation(Surface, Identifiant)
ordre_Traits = ordrePopulation(Traits, Identifiant)

# Préparation la comparaison des listes sur la surface et les traits de côte
def classementsimple(ordre1, ordre2):
    classementsimple = []
    Rang = {}
    for element in ordre1:
        Rang[element[1]] = element[0]
    for element in ordre2:
        if element[1] in Rang:
         classementsimple.append([Rang[element[1]],element[0]])
    return classementsimple

classement_ST = classementsimple(ordre_Traits, ordre_surface)
classement_ST.sort()

# Extraction des deux colonnes sous la forme de liste différentes en utilisant une boucle
Ordresurface = []
OrdreTraits = []
for element in classement_ST:
    OrdreTraits.append(element[0])
    Ordresurface.append(element[1])

# Coefficient de corrélation et Concordance des rangs
spearmanr_ST = spearmanr(OrdreTraits,Ordresurface)
print(spearmanr_ST)
kendalltau_ST = kendalltau(OrdreTraits,Ordresurface)
print(kendalltau_ST)

# Factorisation du code d’analyse des rangs
def rang (ordre1,ordre2):
    spearmanr(ordre1,ordre2)
    kendalltau(ordre1,ordre2)
    return spearmanr,kendalltau
