from tkinter import Label
from typing import final

from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features.copy()
y = car_evaluation.data.targets 
  
# metadata 
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 

# first 5 rows of the data
print(X.head())

# size and shape of the data
print(X.shape)

# missing values and attributes in the data
print(X.info())
print(X.isnull().sum())

# column names and data types
buying_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
maint_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
doors_mapping = {'2': 2, '3': 3, '4': 4, '5more': 5}
persons_mapping = {'2': 2, '4': 4, 'more': 5}
lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
safety_mapping = {'low': 1, 'med': 2, 'high': 3}

# mapping the categorical variables to numerical values
X['buying'] = X['buying'].map(buying_mapping)
X['maint'] = X['maint'].map(maint_mapping)
X['doors'] = X['doors'].map(doors_mapping)
X['persons'] = X['persons'].map(persons_mapping)
X['lug_boot'] = X['lug_boot'].map(lug_boot_mapping)
X['safety'] = X['safety'].map(safety_mapping)

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("explained variance ratio:", pca.explained_variance_ratio_)

# empty list to store inertia values for different k
inertia = []

# range of k values to try
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# plotting the elbow curve
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# 2d scatter plot of the PCA components colored by cluster labels
final_kmeans = KMeans(n_clusters=6, random_state=42)
final_kmeans.fit(X_pca)
Labels = final_kmeans.labels_
print("Cluster sizes: ", np.bincount(Labels))

# slicing the data for the scatter plot
pca_1 = X_pca[:, 0]
pca_2 = X_pca[:, 1]

# build the scatter plot
plt.scatter(pca_1, pca_2, c=Labels, cmap='viridis', alpha=0.6)

# plotting the centroids
centroids = final_kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Car Evaluation Dataset')
plt.legend()
plt.show()