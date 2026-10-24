#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

# C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
# Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
# Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

# Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1 : Théorie de l'échantillonnage (intervalles de fluctuation)
# L'échantillonnage se base sur la répétitivité.
print("Question 1")

# Ouverture du fichier csv
données = ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv")
print(données.head())

# Calcul de la moyenne Pour chaque colonne et arrondissement du calcul sans décimale
moyennes_Pour = round(données["Pour"].mean())
moyennes_Contre = round(données["Contre"].mean())
moyennes_Sans_opinion = round(données["Sans opinion"].mean())
print(moyennes_Pour, moyennes_Contre, moyennes_Sans_opinion)

# Somme des trois moyennes obtenues
somme_moyennes = (moyennes_Pour + moyennes_Contre +  moyennes_Sans_opinion)
print(somme_moyennes)

# Fréquences de l'échantillon
frequencesEchantillon_Pour = round( moyennes_Pour / somme_moyennes,2)
frequencesEchantillon_Contre = round( moyennes_Contre / somme_moyennes,2)
frequencesEchantillon_Sans_opinion = round( moyennes_Sans_opinion / somme_moyennes,2)
print(frequencesEchantillon_Pour, frequencesEchantillon_Contre,  frequencesEchantillon_Sans_opinion)

# Fréquences de la population mère
frequences_mère_Pour = round(852/2185,2)
frequences_mère_Contre = round(911/2185,2)
frequences_mère_Sans_opinion = round(422/2185,2)
print(frequences_mère_Pour, frequences_mère_Contre, frequences_mère_Sans_opinion )

# Intervalle de fluctuation de chacune des fréquences
p1 = frequences_mère_Pour
p2 = frequences_mère_Contre
p3 = frequences_mère_Sans_opinion
from scipy.stats import norm
erreur1 = math.sqrt(p1 * (1 - p1) / 1000)
erreur2 = math.sqrt(p2 * (1 - p2) / 1000)
erreur3 = math.sqrt(p3 * (1 - p3) / 1000)
intervalle_Pour = norm.interval(0.95, loc=p1, scale=erreur1)
intervalle_Contre = norm.interval(0.95, loc=p2, scale=erreur2)
intervalle_Sans_opinion = norm.interval(0.95, loc=p3, scale=erreur3)

print("Résultat sur le calcul d'un intervalle de fluctuation")
print(intervalle_Pour)
print(intervalle_Contre)
print(intervalle_Sans_opinion)

# Question 2 : Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Question 2")

# Extraction du premier échantillon de la liste
df = pd.read_csv("data/Echantillonnage-100-Echantillons.csv")
échantillon_premier = list(df.iloc[0])
print(échantillon_premier)

# Somme de la ligne
Somme_échantillon = sum(échantillon_premier)
print(Somme_échantillon)

# Fréquences de du premier échantillon
Fréquence_premier = [n/Somme_échantillon for n in échantillon_premier]
print(Fréquence_premier)

# Calcul de l’intervalle de confiance
print("Résultat sur le calcul d'un intervalle de confiance")
n = 1000
marge = 1/math.sqrt (n)
for f in Fréquence_premier:
  intervalle_confiance = [f-marge, f+marge]
  print(intervalle_confiance)


# Question 3 : Théorie de la décision (tests d'hypothèse)
# La décision se base sur la notion de risques alpha et bêta.
# Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Question 3")
print("Théorie de la décision")
# Ouverture du fichier csv
données_test1 = ouvrirUnFichier("data/Loi-normale-Test-1.csv")
données_test2 = ouvrirUnFichier("data/Loi-normale-Test-2.csv")
print(données_test1.head())
print(données_test2.head())

# Test de Shapiro-Wilks
from scipy.stats import shapiro
test1 = shapiro(données_test1["Test"].dropna())
test2 = shapiro(données_test2["Test"].dropna())

# W-Value
print(test1[0])
print(test2[0])

# P-Value
print(test1[1])
print(test2[1])

# Résultat sur la théorie de la décision
print("La loi du test 1 est une distribution normale")

# Question bonus 
print("Question bonus")
import matplotlib.pyplot as plt
donnees = données_test2["Test"]
print(donnees.head())

# Comme le montre l’histogramme, les données présentent une distribution statistique des variables discrètes
# De plus, les valeurs étant fortement concentrées sur la valeur minimale de 1 à gauche et s'accompagnant d'une longue queue décroissante à droite, elle présente une forte asymétrie à droite
plot = donnees.plot(
    kind="hist",
    bins=15,
    color="lightgray",
    figsize=(12, 7),
    title="Distribution de Test 2",
    alpha=0.5,
    edgecolor="black"
)
plot.set_xlabel("Valeurs de Test")
plot.set_ylabel("Effectif")
plt.savefig("distribution_test2.png")
plt.show()

# Formulation de deux hypothèses quant à sa loi de probabilité : la loi géométrique et la loi de Poisson.
# Calcul de la moyenne et de la variance
print("Moyenne :",donnees.mean())
print("Variance :",donnees.var())

# La variance (0.58) n'est pas égale à la moyenne (1.19). Par conséquent, la loi de Poisson est écartée.
print("La distribution du test 2 est la loi géométrique")