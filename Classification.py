'''
| This is to do Clustering on data
'''

import numpy as np
import random
import operator
import math

from sklearn import svm, neighbors
from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import *

class Classification:
	def __init__(self, data, method = "svm"):
		self.data = data
		
		self.method = method
		self.random_seed = 12345 # set to None for random
		
		self.h = None
		
		if method == "svm":
			self.GAMMA, self.C = self.svm_best_params()
			
		elif method == "knn":
			self.K = self.knn_best_params()
		
	#---------------------------------------
	def svm_best_params(self, data_limit = 500):
		indexes = range(len(self.data.X))
		random.shuffle(indexes)
		X = [ self.data.X[i] for i in indexes ][:data_limit]
		Y = [ self.data.Y[i] for i in indexes ][:data_limit]
		
		param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]} ]
		clf = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid)
		clf.fit( np.array( X ), np.array( Y ) )
		return clf.best_estimator_.gamma, clf.best_estimator_.C
	
	def knn_best_params(self, data_limit = 500):
		indexes = range(len(self.data.X))
		random.shuffle(indexes)
		X = [ self.data.X[i] for i in indexes ][:data_limit]
		Y = [ self.data.Y[i] for i in indexes ][:data_limit]
		
		param_grid = [ { 'n_neighbors': [5, 10, 15, 20, 25, 30] } ]
		clf = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=param_grid)
		clf.fit( np.array( X ), np.array( Y ) )
		return clf.best_estimator_.n_neighbors

	#---------------------------------------
	def train(self, data_limit = 1000): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		indexes = range(len(self.data.X))
		random.shuffle(indexes)
		X = [ self.data.X[i] for i in indexes ][:data_limit]
		Y = [ self.data.Y[i] for i in indexes ][:data_limit]
	
		if self.method == "svm":
			self.h = svm.SVC(gamma=self.GAMMA, C=self.C, random_state = self.random_seed, probability=True).fit(X, Y)
			
		elif self.method == "knn":
			print "TODO"
			
		else:
			print "TODO"
			
	#---------------------------------------
	def predict(self, x, all = None):
		YP = zip( self.h.classes_, self.h.predict_proba( x )[0] )
		YP.sort(key=operator.itemgetter(1), reverse=True)
		
		if all is None:
			return YP[0] # this is a tuple (y1, p1)
		else:
			return YP # this is a list of tuples [ (y1,p1), (y2,p2), (y3,p3) ]
		
	#---------------------------------------
	def predict_classes(self, xs):
		unique_labels = np.unique( self.data.Y )
		classes = { ul:[] for ul in unique_labels }
		for x in xs:
			y, p = self.predict(x)
			classes[y].append( x )

		return classes
		
	#---------------------------------------
	def uncertainty_margin(self, x):
		YP = self.predict(x, all = True)
		y1, p1 = YP[0]
		y2, p2 = YP[1]
		return 1. - (p1 - p2)
		
	#---------------------------------------
	def uncertainty_prediction(self, x):
		YP = self.predict(x, all = True)
		y1, p1 = YP[0]
		return 1. - p1
	
	#---------------------------------------
	def uncertainty_entropy(self, x):
		YP = self.predict(x, all = True)
		P = [ p for (y,p) in YP ]
		
		entropy = -1.0 * sum( [ p * math.log(p, len(P)) for p in P if p > 0 ] )
	
		return entropy
	#---------------------------------------
	#---------------------------------------
	
	