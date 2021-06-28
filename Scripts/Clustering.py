# Importing required libraries
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

# Class definition
class Clustering_Vectors():

    # Hierarchical Clustering:
    def hierarchical_clustering(self,M):
        Vectors = M.copy()
        
        cluster_id = fclusterdata(X=Vectors, t=3, criterion='maxclust', method='ward')
        mean_vectors = np.zeros((3,3))
        std_vectors = np.zeros((3,3))
        
        for i in np.arange(3) :
            mean_vectors[i, :] = np.mean(Vectors[cluster_id==i+1], axis=0)
            std_vectors[i, :] = np.std(Vectors[cluster_id==i+1])

        print(mean_vectors)

        pass

    # K-means clustering
    def Kmeans_clustering(self, M):
        
        X = M.copy()
        kmeans = KMeans(n_clusters=3).fit(X)
        
        ids = kmeans.labels_
        centers = kmeans.cluster_centers_

        return centers