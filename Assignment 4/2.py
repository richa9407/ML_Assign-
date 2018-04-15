#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 01:10:06 2018

@author: richa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

dataset = pd.read_csv('/Users/savita/desktop/Sales_Transactions_Dataset_Weekly.csv')
x= dataset.iloc[: , 55 :].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
	classifier= KMeans(n_clusters = i, init= 'k-means++', max_iter= 300, n_init=10, random_state=0)
	classifier.fit(x)
	wcss.append(classifier.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

kmeans= KMeans(n_clusters = 2, init='k-means++', max_iter = 300 , n_init=10 , random_state=0)
y_kmeans= kmeans.fit_predict(x)

kmeans= KMeans(n_clusters = 10, init='k-means++', max_iter = 300 , n_init=10 , random_state=0)
y_kmeans1= kmeans.fit_predict(x)

