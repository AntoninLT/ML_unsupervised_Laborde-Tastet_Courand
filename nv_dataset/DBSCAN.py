# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster, metrics
from scipy . io import arff
import scipy.cluster.hierarchy as shc

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


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
path = './artificial/'  
databrut = arff.loadarff(open(path + "zelnik1.arff" , 'r') )
#databrut = arff.loadarff(open(path + "xclara.arff" , 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = [x[0] for x in datanp] # tous les elements de la premiere colonne
f1 = [ x[1] for x in datanp] # tous les elements de la deuxieme colonne

# Initial random assignment of parameters
initial_eps = 0.5
initial_min_samples = 5

# Function to apply DBSCAN and evaluate clustering
def apply_dbscan_and_evaluate(eps, min_samples):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(datanp)

    # Evaluate clustering using silhouette score
    silhouette = silhouette_score(datanp, labels)

    return labels, silhouette

# Distances k plus proches voisins
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)

# Remove the "origin" point
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])
sorted_distances = np.sort(newDistances)

# Plot k nearest neighbors distances
plt.title("Plus proches voisins (5)")
plt.plot(sorted_distances)
plt.show()


#datanp = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])
datanp = np.array(datanp)
clustering = DBSCAN(eps=3, min_samples=2).fit(datanp)
labels = clustering.labels_
silhouette = silhouette_score(datanp, labels)


