import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import distance

class Clustering:
	def __init__(self, data):
		self.data = data
		self.random_seed = 12345 # set to None for random
		self.h = None
    
	#---------------------------------------
	def dist_to_centers(self):
		centers = self.h.cluster_centers_ # FIXME: if the clustering has no centers, compute them based on clusters
		dists = []
		for center in centers:
			dists.append( [ distance.euclidean(x, center) for x in self.data.X ] )
			
		return dists
	
	#---------------------------------------
	def kmeans(self, n_clusters = 2):
		k_means = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, random_state = self.random_seed).fit( self.data.X )
		self.h = k_means
		
		# centers = k_means.cluster_centers_
		labels = k_means.labels_
		
		unique_labels = np.unique(labels)
		clusters = { ul:[] for ul in unique_labels }
		for i in range( len(self.data.X) ):
			clusters[ labels[i] ].append( self.data.X[i] )
		
		return clusters
		
	#---------------------------------------
	def dbscan(self, eps = 0.5, min_samples = 5):
		db = DBSCAN(eps = eps, min_samples = min_samples).fit( self.data.X )
		self.h = db
		
		labels = db.labels_
		
		unique_labels = np.unique(labels)
		clusters = { ul:[] for ul in unique_labels }
		for i in range( len(self.data.X) ):
			clusters[ labels[i] ].append( self.data.X[i] )
		
		return clusters
		
	#---------------------------------------
	
	