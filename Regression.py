'''
| This is to do Clustering on data
'''

import numpy as np
import random
import operator

from sklearn import svm
from sklearn.grid_search import GridSearchCV

class Regression:
	def __init__(self, data, method = "svm"):
		self.data = data
		
		self.method = method
		self.random_seed = 12345 # set to None for random
		
		self.h = None
		
		if method == "svm":
			self.GAMMA, self.C = self.svm_best_params()
			
		elif method == "knn":
			# self.K = self.knn_best_params()
			print "TODO"
		
	#---------------------------------------
	def svm_best_params(self, data_limit = 500): # FIXME data_limit
		indexes = range(len(self.data.X))
		random.shuffle(indexes)
		X = [ self.data.X[i] for i in indexes ][:data_limit]
		Y = [ self.data.Y[i] for i in indexes ][:data_limit]
		
		param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]} ]
		clf = GridSearchCV(estimator=svm.SVR(), param_grid=param_grid)
		clf.fit( np.array( X ), np.array( Y ) )
		return clf.best_estimator_.gamma, clf.best_estimator_.C
		
	#---------------------------------------
	def train(self, data_limit = 1000): # TODO make method to shuffle and return sublist with data_limit
		indexes = range(len(self.data.X))
		random.shuffle(indexes)
		X = [ self.data.X[i] for i in indexes ][:data_limit]
		Y = [ self.data.Y[i] for i in indexes ][:data_limit]
		
		if self.method == "svm":
			self.h = svm.SVR(kernel='rbf', C=self.C, gamma=self.GAMMA).fit(X, Y)
			
		elif self.method == "knn":
			print "TODO"
			
		else:
			print "TODO"
			
	#---------------------------------------
	def predict(self, x):
		y = self.h.predict(x)[0]
		return y
		
	#---------------------------------------
	
	