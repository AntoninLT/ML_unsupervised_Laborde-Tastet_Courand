# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster, metrics
import scipy.cluster.hierarchy as shc

path = './dataset-rapport/'
dataset_name='x1.txt'
dataset_list=['x1.txt','x2.txt','x3.txt','x4.txt','y1.txt','zz1.txt','zz2.txt']

for name in dataset_list:
    with open(path+name, 'r') as file:
        # Lire chaque ligne du fichier
        dataset = file.readlines()
    
    f0  = [float(x.split()[0]) for x in dataset]
    f1  = [float(x.split()[1]) for x in dataset]
    data  = [[float(x.split()[0]), float(x.split()[1])] for x in dataset]
    
    # Les données sont dans datanp (2 dimensions)
    # f0: valeurs sur la première dimension
    # f1: valeur sur la deuxième dimension
    
    # Choisissez le nombre maximal de clusters que vous souhaitez tester
    max_clusters = 16
    
    # Appliquez itérativement la méthode pour différents seuils de distance
    for linkage_method in ['single', 'average', 'complete', 'ward']:
        print(f"\nLinkage Method: {linkage_method}\n")
    
        S=[]
        DB=[]
        CH=[]
        
        for k in range(2, max_clusters + 1):
            print(f"\nNombre de clusters k = {k}:")
    
            # Mesurez le temps de calcul
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(linkage=linkage_method, n_clusters=k)
            model.fit(data)
            tps2 = time.time()
    
            labels = model.labels_
            leaves = model.n_leaves_
    
            # Visualisation des résultats du clustering
            plt.scatter(f0, f1, c=labels, s=8)
            plt.title(f"Resultat du clustering (dataset: {name}, Linkage: {linkage_method}, k={k})")
            plt.show()
    
            # Mesures d'évaluation
            silhouette_score = metrics.silhouette_score(data, labels)
            S.append(silhouette_score)
            davies_bouldin_index = metrics.davies_bouldin_score(data, labels)
            DB.append(davies_bouldin_index)
            calinski_harabasz_index = metrics.calinski_harabasz_score(data, labels)
            CH.append(calinski_harabasz_index)
    
    
            print(f"Nombre d'itérations = 0, Temps d'exécution = {round((tps2 - tps1) * 1000, 2)} ms")
            print(f"Silhouette Score = {silhouette_score}, Davies-Bouldin Index = {davies_bouldin_index}, Calinski-Harabasz Index = {calinski_harabasz_index}")
            
        rg = np.arange(2,17)
        plt.subplot(3,1,1)
        plt.plot(rg, S)
        plt.title(f"silhouette_score - dataset: {name}")
        plt.subplot(3,1,2)
        plt.plot(rg, DB)
        plt.title(f"davies_bouldin_index - dataset: {name}")
        plt.subplot(3,1,3)
        plt.plot(rg, CH)
        plt.title(f"calinski_harabasz_index - dataset: {name}")
        plt.show()
        
# Donnees dans datanp
print("Dendrogramme 'single' donnees initiales")
linked_mat = shc.linkage(data, 'ward')
plt.figure(figsize=(12, 12))
shc.dendrogram(linked_mat,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False)
plt.show()