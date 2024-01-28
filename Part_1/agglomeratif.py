# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
from scipy.io import arff

# Charger les données depuis le fichier ARFF
path = './artificial/'
databrut = arff.loadarff(open(path + "xclara.arff", 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

f0 = [x[0] for x in datanp]
f1 = [x[1] for x in datanp]

# Afficher le dataset d'origine
plt.scatter(f0, f1, s=8)
plt.title("Données d'origine")
plt.show()

# Nombre de clusters à évaluer
cluster_range = range(2, 11)

# Métriques d'évaluation
silhouette_scores = []
davies_bouldin_indices = []
calinski_harabasz_indices = []

# Pour chaque nombre de clusters
for n_clusters in cluster_range:
    # Créer un modèle de clustering agglomératif
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    
    # Obtenir les labels du clustering
    labels = model.fit_predict(datanp)
    
    # Évaluation avec le coefficient de silhouette
    silhouette_score = metrics.silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_score)
    
    # Évaluation avec l'indice de Davies-Bouldin
    davies_bouldin_index = metrics.davies_bouldin_score(datanp, labels)
    davies_bouldin_indices.append(davies_bouldin_index)
    
    # Évaluation avec l'indice de Calinski-Harabasz
    calinski_harabasz_index = metrics.calinski_harabasz_score(datanp, labels)
    calinski_harabasz_indices.append(calinski_harabasz_index)

# Trouver les valeurs optimales de K pour chaque métrique
optimal_k_silhouette = np.argmax(silhouette_scores) + 2
optimal_k_davies_bouldin = np.argmin(davies_bouldin_indices) + 2
optimal_k_calinski_harabasz = np.argmax(calinski_harabasz_indices) + 2

# Afficher les évolutions des métriques en fonction du nombre de clusters (K)
plt.figure(figsize=(10, 6))

# Graphique pour le coefficient de silhouette
plt.plot(cluster_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.scatter(optimal_k_silhouette, max(silhouette_scores), color='r', label=f'Optimal K (Silhouette) = {optimal_k_silhouette}')
plt.title("Évolution du coefficient de silhouette en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Coefficient de silhouette")
plt.legend()
plt.show()

# Graphique pour l'indice de Davies-Bouldin
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, davies_bouldin_indices, marker='o', label='Davies-Bouldin Index')
plt.scatter(optimal_k_davies_bouldin, min(davies_bouldin_indices), color='r', label=f'Optimal K (Davies-Bouldin) = {optimal_k_davies_bouldin}')
plt.title("Évolution de l'indice de Davies-Bouldin en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Indice de Davies-Bouldin")
plt.legend()
plt.show()

# Graphique pour l'indice de Calinski-Harabasz
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, calinski_harabasz_indices, marker='o', label='Calinski-Harabasz Index')
plt.scatter(optimal_k_calinski_harabasz, max(calinski_harabasz_indices), color='r', label=f'Optimal K (Calinski-Harabasz) = {optimal_k_calinski_harabasz}')
plt.title("Évolution de l'indice de Calinski-Harabasz en fonction du nombre de clusters (K)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Indice de Calinski-Harabasz")
plt.legend()
plt.show()

# Afficher le clustering optimal avec le nombre de clusters correspondant pour chaque métrique
model_optimal_silhouette = cluster.AgglomerativeClustering(n_clusters=optimal_k_silhouette)
labels_optimal_silhouette = model_optimal_silhouette.fit_predict(datanp)
plt.scatter(f0, f1, c=labels_optimal_silhouette, s=8)
plt.title(f"Clustering Agglomératif optimal avec K={optimal_k_silhouette} (Silhouette)")
plt.show()

model_optimal_davies_bouldin = cluster.AgglomerativeClustering(n_clusters=optimal_k_davies_bouldin)
labels_optimal_davies_bouldin = model_optimal_davies_bouldin.fit_predict(datanp)
plt.scatter(f0, f1, c=labels_optimal_davies_bouldin, s=8)
plt.title(f"Clustering Agglomératif optimal avec K={optimal_k_davies_bouldin} (Davies-Bouldin)")
plt.show()

model_optimal_calinski_harabasz = cluster.AgglomerativeClustering(n_clusters=optimal_k_calinski_harabasz)
labels_optimal_calinski_harabasz = model_optimal_calinski_harabasz.fit_predict(datanp)
plt.scatter(f0, f1, c=labels_optimal_calinski_harabasz, s=8)
plt.title(f"Clustering Agglomératif optimal avec K={optimal_k_calinski_harabasz} (Calinski-Harabasz)")
plt.show()

print(f"Le nombre optimal de clusters (K) avec le coefficient de silhouette est : {optimal_k_silhouette}")
print(f"Le nombre optimal de clusters (K) avec l'indice de Davies-Bouldin est : {optimal_k_davies_bouldin}")
print(f"Le nombre optimal de clusters (K) avec l'indice de Calinski-Harabasz est : {optimal_k_calinski_harabasz}")

