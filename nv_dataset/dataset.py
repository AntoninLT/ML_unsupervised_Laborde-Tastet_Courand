# -*- coding: utf-8 -*-

import numpy as np
import matplotlib . pyplot as plt
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
dataset_name='zz2.txt'

with open(path+dataset_name, 'r') as file:
    # Lire chaque ligne du fichier
    dataset = file.readlines()

f0  = [float(x.split()[0]) for x in dataset]
f1  = [float(x.split()[1]) for x in dataset]

plt.scatter( f0 , f1 , s = 8 )
plt.title( " Donnees dataset : "+str(dataset_name) )
plt.show()

