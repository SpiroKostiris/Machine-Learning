# Covers Andrew Ng's machine learning Coursera: Exercise 7 K-Means Clustering and PCA
# Implements python version from https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-7/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

def initialcentroids(X, k):
    len, cols = X.shape
    centroids = np.zeros((k, cols))
    idx = np.random.randint(0, len, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids

def closestcentroids(X, centroids):
    len = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(len)
    
    for i in range(len):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if (dist < min_dist):
                min_dist = dist
                idx[i] = j
    
    return idx

def findcentroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

def pca(X, centroidsstart, iter):
    len, cols = X.shape
    k = centroidsstart.shape[0]
    idx = np.zeros(len)
    centroids = centroidsstart
    
    for i in range(iter):
        idx = closestcentroids(X, centroids)
        centroids = findcentroids(X, idx, k)
    
    return idx, centroids


data = loadmat('ex7data2.mat')
X = data['X']
firstcentroids = initialcentroids(X, 3)

idx = closestcentroids(X, firstcentroids)
print(idx[0:3])

findcentroids(X, idx, 3)

idx, centroids = pca(X, firstcentroids, 100)

cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()



