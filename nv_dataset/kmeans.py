# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster, metrics

# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = './dataset-rapport/'
dataset_name='y1.txt'

with open(path+dataset_name, 'r') as file:
    # Lire chaque ligne du fichier
    dataset = file.readlines()

f0  = [float(x.split()[0]) for x in dataset]
f1  = [float(x.split()[1]) for x in dataset]
data  = [[float(x.split()[0]), float(x.split()[1])] for x in dataset]

# Les données sont dans datanp (2 dimensions)
# f0: valeurs sur la première dimension
# f1: valeur sur la deuxième dimension

S=[]
DB=[]
CH=[]

def kmeans_evaluation(data, k):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    tps2 = time.time()
    labels = model.labels_
    iteration = model.n_iter_

    # Mesures d'évaluation
    silhouette_score = metrics.silhouette_score(data, labels)
    S.append(silhouette_score)
    davies_bouldin_index = metrics.davies_bouldin_score(data, labels)
    DB.append(davies_bouldin_index)
    calinski_harabasz_index = metrics.calinski_harabasz_score(data, labels)
    CH.append(calinski_harabasz_index)
    
    # Visualisation des résultats
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"Données après clustering K-Means (k={k})")
    plt.show()

    print(f"\nNombre de clusters = {k}, Nombre d'itérations = {iteration}, Temps d'exécution = {round((tps2 - tps1) * 1000, 2)} ms")
    print(f"silhouette_score Score = {silhouette_score}, \nDavies-Bouldin Index = {davies_bouldin_index}, \nCalinski-Harabasz Index = {calinski_harabasz_index}")

# Choisissez le nombre maximal de clusters que vous souhaitez tester
max_clusters = 10

# Appliquez itérativement la méthode pour différents nombres de clusters
for k in range(2, max_clusters + 1):
    kmeans_evaluation(data, k)

rg = np.arange(2,11)
plt.subplot(3,1,1)
plt.plot(rg, S)
plt.title("silhouette_score")
plt.subplot(3,1,2)
plt.plot(rg, DB)
plt.title("davies_bouldin_index")
plt.subplot(3,1,3)
plt.plot(rg, CH)
plt.title("calinski_harabasz_index")
plt.show()