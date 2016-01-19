import numpy as np
import random
import operator
import math

from sklearn import svm, neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *

class Classification:
	def __init__(self, X, Y, method = "svm", Vx = None, Vy = None, tuning = True):
		self.X = X
		self.Y = Y
		
		self.Vx = Vx
		self.Vy = Vy
		
		self.method = method
		self.random_seed = 12345 # set to None for random
		
		self.h = None
		
		if method == "svm":
			if tuning: self.GAMMA, self.C = self.svm_best_params()
			
		elif method == "knn":
			if tuning: self.K = self.knn_best_params()
		
	#---------------------------------------
	def svm_best_params(self, data_limit = 1500):
		if self.Vx == None:
			Vx = self.X
			Vy = self.Y
		else:
			Vx = self.Vx
			Vy = self.Vy
		
		indexes = range(len(Vx))
		random.shuffle(indexes)
		X = [ Vx[i] for i in indexes ][:data_limit]
		Y = [ Vy[i] for i in indexes ][:data_limit]
		
		param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]} ]
		clf = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid)
		clf.fit( np.array( X ), np.array( Y ) )
		return clf.best_estimator_.gamma, clf.best_estimator_.C
	
	def knn_best_params(self, data_limit = 1500):
		if self.Vx == None:
			Vx = self.X
			Vy = self.Y
		else:
			Vx = self.Vx
			Vy = self.Vy
			
		indexes = range(len(Vx))
		random.shuffle(indexes)
		X = [ Vx[i] for i in indexes ][:data_limit]
		Y = [ Vy[i] for i in indexes ][:data_limit]
		
		param_grid = [ { 'n_neighbors': [5, 10, 15, 20, 25, 30] } ]
		clf = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=param_grid)
		clf.fit( np.array( X ), np.array( Y ) )
		return clf.best_estimator_.n_neighbors

	#---------------------------------------
	def train(self, W = None): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		if self.method == "svm":
			W = W if W is not None else [1.]*len(self.X)
			self.h = svm.SVC(gamma=self.GAMMA, C=self.C, random_state = self.random_seed, probability=True).fit(self.X, self.Y, sample_weight = W)
			
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
		unique_labels = np.unique( self.Y )
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
	def uncertainty_weight(self, x, Lx, Ly):	
		y1, y2, p1, p2 = self.getMarginInfo(x)

		Lxx = Lx[:] + [x]
		Lyy = Ly[:] + [y2]
		
		w = 1.; y1_new = y1;
		step = 0.001; lower = 0.; upper = 10.
		
		while (upper - lower > step):
			w = (upper + lower) / 2.
			Lww = [1.]*len(Ly) + [w]
			
			# temp_clf = self.__class__(Lxx, Lyy, method = self.method, Vx = self.Vx, Vy = self.Vy)
			# temp_clf.train(Lww)
			# y1_new = self.predict_label(x)
			
			hh = svm.SVC(gamma=self.GAMMA, C=self.C, random_state = self.random_seed, probability=True).fit(Lxx, Lyy, sample_weight = Lww) #TODO
			y1_new = hh.predict([x])[0]
			
			if y1_new == y2: upper = w # if y1_new != y1: upper = w
			else: lower = w
		
		info = 1. - w
		
		return info

	#---------------------------------------
	def predict_label(self, x):
		return self.h.predict([x])[0]
	
	#---------------------------------------
	def getMarginInfo(self, x):
		y1 = self.h.predict([x])[0]
		l2 = self.getLabelOf(2, x)
		l1 = self.getLabelOf(1, x)
		y2 = l2 if y1 == l1 else l1
		p1 = self.getPredictProba(1, x)
		p2 = self.getPredictProba(2, x)
		return y1, y2, p1, p2
	
	#---------------------------------------
	def getLabelOf(self, i, x):
		YP = zip(self.h.classes_, self.h.predict_proba([x])[0])
		YP.sort(key=operator.itemgetter(1), reverse=True)
		for pos, li in enumerate(YP):
			label, proba = li
			if pos == i-1:
				return label
		return 0.

	#---------------------------------------
	def getProbaOf(self, y, x):
		YP = zip(self.h.classes_, self.h.predict_proba([x])[0])
		YP.sort(key=operator.itemgetter(1), reverse=True)
		for li in YP:
			label, proba = li
			if label == y:
				return proba
		
		return 0.
		
	#---------------------------------------
	def getPredictProba(self, i, x):
		if i < 1: i = 1
		P = list(self.h.predict_proba(x)[0])
		P.sort(reverse=True)
		return P[i-1]
		
	#---------------------------------------
	def getTestAccuracy(self, Tx, Ty):
		return 1. * accuracy_score( Ty, self.h.predict(Tx) )
	#---------------------------------------
	
	