
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns; sns.set()

dataset = pd.read_csv('/Users/savita/desktop/Facebook_metrics/dataset_Facebook.csv',delimiter= ';')
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset["Type_code"] = lb_make.fit_transform(dataset["Type"])
dataset[["Type", "Type_code"]].head(1)

del dataset['Type']

data= dataset.iloc[:,9:11].values
plt.scatter(data[9],data[10])

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
data= sc_x.fit_transform(data)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
	classifier= KMeans(n_clusters = i, init= 'k-means++', max_iter= 300, n_init=10, random_state=0)
	classifier.fit(data)
	wcss.append(classifier.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

kmeans= KMeans(n_clusters = 2, init='k-means++', max_iter = 300 , n_init=10 , random_state=0)
y_kmeans= kmeans.fit_predict(data)
plt.scatter(data[y_kmeans==0,0],data[y_kmeans==0,1], s=100 , c='green', label = 'Cluster 1')
plt.scatter(data[y_kmeans==1,0],data[y_kmeans==1,1], s=100 , c='blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 100 , c='yellow', label='Centroids')
plt.title('Clusters of classifiers')
plt.xlabel('life time post consumers')
plt.ylabel('Life time post Consumption')
plt.legend()
plt.show()
    
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidian distances')
plt.show()

#fitting heirarchial clustering to the dataset 
from sklearn.cluster import AgglomerativeClustering 
clustering= AgglomerativeClustering(n_clusters=2 , affinity= 'euclidean', linkage='ward')
y_hc= clustering.fit_predict(data)
plt.scatter(data[y_hc==0,0],data[y_hc==0,1], s=100 , c='green', label = 'Cluster 1')
plt.scatter(data[y_hc==1,0],data[y_hc==1,1], s=100 , c='blue', label = 'Cluster 2')
plt.title('Clusters of classifiers')
plt.xlabel('life time post consumers')
plt.ylabel('Life time post Consumption')
plt.legend()
plt.show()

#fitting for gausian mixture model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(data)
labels = gmm.predict(data)
plt.scatter(data[labels==0,0],data[labels==0,1], s=100 , c='green', label = 'Cluster 1')
plt.scatter(data[labels==1,0],data[labels==1,1], s=100 , c='blue', label = 'Cluster 2')
plt.title('Clusters of classifiers')
plt.xlabel('life time post consumers')
plt.ylabel('Life time post Consumption')
plt.legend()
plt.show()


