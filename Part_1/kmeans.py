# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
from scipy.io import arff

# Charger les données depuis le fichier ARFF
path = './artificial/'
databrut = arff.loadarff(open(path + "zelnik1.arff" , 'r') )
#databrut = arff.loadarff(open(path + "twodiamonds.arff" , 'r') )
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

f0 = [x[0] for x in datanp]
f1 = [x[1] for x in datanp]

# Afficher le dataset d'origine
plt.scatter(f0, f1, s=8)
plt.title("Données d'origine")
plt.show()

# Trouver le K optimal pour le coefficient de silhouette
max_clusters = 10
silhouette_scores = []

for k in range(2, max_clusters + 1):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    labels = model.fit_predict(datanp)
    silhouette_score = metrics.silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_score)

# Trouver la valeur optimale de K
optimal_k_silhouette = np.argmax(silhouette_scores) + 2

# Afficher l'évolution du coefficient de silhouette
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.axvline(x=optimal_k_silhouette, color='r', linestyle='--', label=f'K optimal = {optimal_k_silhouette}')
plt.title("Évolution du coefficient de silhouette en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Coefficient de silhouette")
plt.legend()
plt.show()

# Afficher le clustering optimal avec le nombre de clusters correspondant
model_optimal_silhouette = cluster.KMeans(n_clusters=optimal_k_silhouette, init='k-means++')
labels_optimal_silhouette = model_optimal_silhouette.fit_predict(datanp)

# Afficher le clustering optimal
plt.scatter(f0, f1, c=labels_optimal_silhouette, s=8)
plt.title(f"Clustering K-Means optimal avec K={optimal_k_silhouette} (Silhouette)")
plt.show()

print(f"Le nombre optimal de clusters (K) avec le coefficient de silhouette est : {optimal_k_silhouette}")

# Trouver le K optimal pour l'indice de Davies-Bouldin
davies_bouldin_indices = []

for k in range(2, max_clusters + 1):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    labels = model.fit_predict(datanp)
    davies_bouldin_index = metrics.davies_bouldin_score(datanp, labels)
    davies_bouldin_indices.append(davies_bouldin_index)

# Trouver la valeur optimale de K
optimal_k_davies_bouldin = np.argmin(davies_bouldin_indices) + 2

# Afficher l'évolution de l'indice de Davies-Bouldin
plt.plot(range(2, max_clusters + 1), davies_bouldin_indices, marker='o')
plt.axvline(x=optimal_k_davies_bouldin, color='r', linestyle='--', label=f'K optimal = {optimal_k_davies_bouldin}')
plt.title("Évolution de l'indice de Davies-Bouldin en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Indice de Davies-Bouldin")
plt.legend()
plt.show()

# Afficher le clustering optimal avec le nombre de clusters correspondant
model_optimal_davies_bouldin = cluster.KMeans(n_clusters=optimal_k_davies_bouldin, init='k-means++')
labels_optimal_davies_bouldin = model_optimal_davies_bouldin.fit_predict(datanp)

# Afficher le clustering optimal
plt.scatter(f0, f1, c=labels_optimal_davies_bouldin, s=8)
plt.title(f"Clustering K-Means optimal avec K={optimal_k_davies_bouldin} (Davies-Bouldin)")
plt.show()

print(f"Le nombre optimal de clusters (K) avec l'indice de Davies-Bouldin est : {optimal_k_davies_bouldin}")

# Trouver le K optimal pour l'indice de Calinski-Harabasz
calinski_harabasz_indices = []

for k in range(2, max_clusters + 1):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    labels = model.fit_predict(datanp)
    calinski_harabasz_index = metrics.calinski_harabasz_score(datanp, labels)
    calinski_harabasz_indices.append(calinski_harabasz_index)

# Trouver la valeur optimale de K
optimal_k_calinski_harabasz = np.argmax(calinski_harabasz_indices) + 2

# Afficher l'évolution de l'indice de Calinski-Harabasz
plt.plot(range(2, max_clusters + 1), calinski_harabasz_indices, marker='o')
plt.axvline(x=optimal_k_calinski_harabasz, color='r', linestyle='--', label=f'K optimal = {optimal_k_calinski_harabasz}')
plt.title("Évolution de l'indice de Calinski-Harabasz en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Indice de Calinski-Harabasz")
plt.legend()
plt.show()

# Afficher le clustering optimal avec le nombre de clusters correspondant
model_optimal_calinski_harabasz = cluster.KMeans(n_clusters=optimal_k_calinski_harabasz, init='k-means++')
labels_optimal_calinski_harabasz = model_optimal_calinski_harabasz.fit_predict(datanp)

# Afficher le clustering optimal
plt.scatter(f0, f1, c=labels_optimal_calinski_harabasz, s=8)
plt.title(f"Clustering K-Means optimal avec K={optimal_k_calinski_harabasz} (Calinski-Harabasz)")
plt.show()

print(f"Le nombre optimal de clusters (K) avec l'indice de Calinski-Harabasz est : {optimal_k_calinski_harabasz}")

