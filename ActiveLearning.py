import numpy as np
import random
import operator
from sklearn import svm, neighbors
from Classification import Classification
from A2ING import A2ING

class ActiveLearning:
	def __init__(self, Lx, Ly, Ux, Uy, Tx, Ty, method = "svm", budget = 300):
		self.Lx = Lx
		self.Ly = Ly
		self.Ux = Ux
		self.Uy = Uy # TODO should not be here
		self.Tx = Tx # TODO should not be here
		self.Ty = Ty # TODO should not be here
		
		self.optimization_limit = 30
		self.optimization_method = "margin" # margin proba entropy random weight expectedErrorReduction etc
		
		self.budget = budget
		self.accuracys = []
		
		self.clf = Classification( self.Lx, self.Ly, method = method, Vx = Lx+Ux, Vy = Ly+Uy )
		self.clf.train()
		
		self.a2ing = A2ING()
		for i,x in enumerate(self.Lx): self.a2ing.learn(x, self.Ly[i])
		self.outsiders_rate = sum([ 1. if self.a2ing.isOutsider(dp) else 0. for dp in self.Ux]) / ( len(self.Ux) )
		
	#---------------------------------------
	def train(self, mtd = "margin"): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		for i in range(self.budget):
			if len(self.Ux) <= 1: break
			
			ids, scores = self.sortForInformativeness(mtd)
			id = ids[0]
			
			qx = self.Ux[id]
			qy = self.Uy[id]
			
			# print len(self.a2ing.graph.nodes), self.a2ing.isOutsider(qx), self.a2ing.r
			self.a2ing.learn(qx, qy)
			
			self.Lx.append(qx)
			self.Ly.append(qy)
			self.Ux.pop(id)
			self.Uy.pop(id)

			self.clf.X = self.Lx; self.clf.Y = self.Ly
			self.clf.train()
			
			test_accuracy = self.clf.getTestAccuracy( self.Tx, self.Ty )
			self.accuracys.append( test_accuracy )
			
			print "i=", i, "; acc=%.4f"%test_accuracy, "%.4f"%np.mean(self.accuracys), "%.4f"%np.average(self.accuracys, weights = range(1,1+len(self.accuracys))), len(self.a2ing.graph.nodes), self.a2ing.r, scores[0], self.outsiders_rate
			
			self.outsiders_rate = sum([ 1. if self.a2ing.isOutsider(dp) else 0. for dp in self.Ux]) / ( len(self.Ux) )
			# TODO self.outsiders_rate defined independently of the radius (outsiders/insiders), but acording to proba of outsider using pvalue
		
	#---------------------------------------
	def sortForInformativeness(self, mtd):
		if mtd in ["etc", "etc_", "expectedErrorReduction", "weight", "optimal"] :
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
					YP.sort(key=operator.itemgetter(1), reverse=True)
					for ir, (yy, proba) in enumerate(YP):
						if ir == 5: break
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
				
			#---------------------------------------------------------
			elif mtd == "etc_":
				if ix in ids[:self.optimization_limit]:
					temp_clf1 = Classification(self.Lx + [x], self.Ly + [y1], method = self.clf.method); temp_clf1.GAMMA, temp_clf1.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do ir general not only svm specific
					temp_clf1.train()
					diff1 = sum( [ 1. if temp_clf1.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					temp_clf2 = Classification(self.Lx + [x], self.Ly + [y2], method = self.clf.method); temp_clf2.GAMMA, temp_clf2.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do ir general not only svm specific
					temp_clf2.train()
					diff2 = sum( [ 1. if temp_clf2.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					informativeness = diff1 # this one is particularly good for rejection (to be confirmed)
					informativeness = p1*diff1 + p2*diff2 + 1.
				else:
					informativeness = 0.
				
			#---------------------------------------------------------
			elif mtd == "test":
				informativeness = self.clf.uncertainty_margin(x) * (1. + self.outsiders_rate if self.a2ing.isOutsider(x) else 1.)
				
			scores.append( informativeness )
		
		ids = (-np.array(scores)).argsort()
		sorted_scores = [ scores[id] for id in ids ]	
		# sorted_scores = [ 1.*scores[id] / sum(scores) for id in ids ]	
		
		return ids, sorted_scores
		
	#---------------------------------------
	#---------------------------------------
	
