import numpy as np
import random
from sklearn import svm, neighbors
from Classification import Classification

class ActiveLearning:
	def __init__(self, Lx, Ly, Ux, Uy, Tx, Ty, method = "svm", budget = 500):
		self.Lx = Lx
		self.Ly = Ly
		self.Ux = Ux
		self.Uy = Uy # TODO should not be here
		self.Tx = Tx # TODO should not be here
		self.Ty = Ty # TODO should not be here
		
		self.optimization_limit = 10
		self.optimization_method = "margin"
		
		self.budget = budget
		self.accuracys = []
		
		self.clf = Classification( self.Lx, self.Ly, method = method, Vx = Lx+Ux, Vy = Ly+Uy )
		self.clf.train()
		
	#---------------------------------------
	def train(self, mtd = "margin"): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		for i in range(self.budget):
			if len(self.Ux) <= 1: break
			
			ids, scores = self.sortForInformativeness(mtd)
			id = ids[0]
			
			qx = self.Ux[id]
			qy = self.Uy[id]
			
			self.Lx.append(qx)
			self.Ly.append(qy)
			self.Ux.pop(id)
			self.Uy.pop(id)

			self.clf.X = self.Lx; self.clf.Y = self.Ly
			self.clf.train()
			
			test_accuracy = self.clf.getTestAccuracy( self.Tx, self.Ty )
			self.accuracys.append( test_accuracy )
			
			print "i=", i, "; acc=%.4f"%test_accuracy, "%.4f"%np.mean(self.accuracys), "%.4f"%np.average(self.accuracys, weights = range(1,1+len(self.accuracys)))
		
	#---------------------------------------
	def sortForInformativeness(self, mtd):
		if mtd == "etc" or mtd == "expectedErrorReduction" or mtd == "weight" or mtd == "optimal":
			ids, scores = self.sortForInformativeness(self.optimization_method)
		
		scores = []
		for ix, x in enumerate(self.Ux):
			y1, y2, p1, p2 = self.clf.getMarginInfo(x)
			
			if mtd == "margin":
				informativeness = self.clf.uncertainty_margin(x)
			
			#---------------------------------------------------------
			elif mtd == "proba":
				informativeness = self.clf.uncertainty_prediction(x)
			
			#---------------------------------------------------------
			elif mtd == "entropy":
				informativeness = self.clf.uncertainty_entropy(x)
			
			#---------------------------------------------------------
			elif mtd == "random":
				informativeness = random.uniform(0., 1.)
			
			#---------------------------------------------------------
			elif mtd == "weight":
				if ix in ids[:self.optimization_limit]:
					informativeness = self.clf.uncertainty_weight(x, self.Lx, self.Ly)
				else: informativeness = 0.
			
			#---------------------------------------------------------
			elif mtd == "expectedErrorReduction":
				if ix in ids[:self.optimization_limit]:
					sums = 0.
					YP = self.clf.predict(x, all = True)
					for (yy, proba) in YP:
						temp_clf = Classification(self.Lx + [x], self.Ly + [yy], method = self.clf.method); temp_clf.GAMMA, temp_clf.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do ir general not only svm specific
						temp_clf.train()
						e_h1 = sum( [ temp_clf.uncertainty_entropy(dp) for dp in self.Ux if dp != x ] )
						
						sums += (proba) * e_h1
					informativeness = 1. / sums
				else:
					informativeness = 0.
			
			#---------------------------------------------------------
			elif mtd == "etc":
				if ix in ids[:self.optimization_limit]:
					temp_clf1 = Classification(self.Lx + [x], self.Ly + [y1], method = self.clf.method); temp_clf1.GAMMA, temp_clf1.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do ir general not only svm specific
					temp_clf1.train()
					diff1 = sum( [ abs(temp_clf1.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) if temp_clf1.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					temp_clf2 = Classification(self.Lx + [x], self.Ly + [y2], method = self.clf.method); temp_clf2.GAMMA, temp_clf2.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do ir general not only svm specific
					temp_clf2.train()
					diff2 = sum( [ abs(temp_clf2.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) if temp_clf2.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					informativeness = diff1 # this one is particularly good for rejection (to be confirmed)
					informativeness = p1*diff1 + p2*diff2 + 1.
				else:
					informativeness = 0.
				
			scores.append( informativeness )
		
		ids = (-np.array(scores)).argsort()
		sorted_scores = [ 1.*scores[id] / sum(scores) for id in ids ]	
		
		return ids, sorted_scores
		
	#---------------------------------------
	#---------------------------------------
	
