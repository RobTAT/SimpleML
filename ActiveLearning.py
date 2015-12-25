import numpy as np
import random
from Classification import Classification

class ActiveLearning:
	def __init__(self, Lx, Ly, Ux, Uy, Tx, Ty, method = "svm", budget = 100):
		self.Lx = Lx
		self.Ly = Ly
		self.Ux = Ux
		self.Uy = Uy # TODO should not be here
		self.Tx = Tx # TODO should not be here
		self.Ty = Ty # TODO should not be here
		
		self.budget = budget
		self.accuracys = []
		
		self.clf = Classification( self.Lx, self.Ly, method = method )
		self.clf.train()
		
	#---------------------------------------
	def train(self, mtd = "margin"): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		for i in range(self.budget):
			ids, scores = self.sortForInformativeness(mtd)
			id = ids[0]
			
			qx = self.Ux[id]
			qy = self.Uy[id]
			
			self.Lx.append(qx)
			self.Ly.append(qy)
			self.Ux.pop(id)
			self.Uy.pop(id)

			self.clf = Classification( self.Lx, self.Ly, method = self.clf.method )
			self.clf.train()
			
			test_accuracy = self.clf.getTestAccuracy( self.Tx, self.Ty )
			self.accuracys.append( test_accuracy )
			
			print "i=", i, "; acc=%.4f"%test_accuracy, "%.4f"%np.mean(self.accuracys), "%.4f"%np.average(self.accuracys, weights = range(1,1+len(self.accuracys)))
		
	#---------------------------------------
	def sortForInformativeness(self, mtd):
		if mtd == "etc" or mtd == "expectedErrorReduction" or mtd == "weight" or mtd == "optimal":
			ids, scores = self.sortForInformativeness("margin")
		
		scores = []
		for ix, x in enumerate(self.Ux):
			y1, y2, p1, p2 = self.clf.getMarginInfo(x)
			
			if mtd == "margin": informativeness = self.clf.uncertainty_margin(x)
			elif mtd == "proba": informativeness = self.clf.uncertainty_prediction(x)
			elif mtd == "entropy": informativeness = self.clf.uncertainty_entropy(x)
			elif mtd == "random": informativeness = random.uniform(0., 1.)
			
			'''
			elif mtd == "weight":
				if ix in ids[:10]:
					informativeness = getInformativeness(h, x, X_train, Y_train)
				else: informativeness = 0.
			
			elif mtd == "expectedErrorReduction":
				if ix in ids[:10]:
					sums = 0.
					YP = zip(h.classes_, h.predict_proba([x])[0])
					for (yy, proba) in YP:
						h1 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x], Y_train + [yy])
						e_h1 = sum( [ raf.getPredictionEntropy(dp, h1) for dp in data if dp != x ] )
						sums += (proba) * e_h1
					informativeness = 1. / sums
				else:
					informativeness = 0.
			
			elif mtd == "optimal": #=====================[VERY GOOD] the optimal strategy because you directly use the labeled test set
				if ix in ids[:100]:
					h1 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x], Y_train + [y1])
					info1 = 1. * accuracy_score( labels_test, h1.predict(data) )
					
					h2 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x], Y_train + [y2])
					info2 = 1. * accuracy_score( labels_test, h2.predict(data) )
					
					informativeness = info1
					informativeness = max(info1, info2)
				else: informativeness = 0.
				
			elif mtd == "suboptimal": #======================[BAD but INTERESTING] no many labeled in L: should find a way for oversampling and not using leave one out
				Xoo = np.array(X_train); Yoo = np.array(Y_train)
				loo = cross_validation.LeaveOneOut(len(Xoo))
				
				info1 = 0.
				for itra,ites in loo:
					X_tra = [ list(e) for e in Xoo[itra] ]; Y_tra = list( Yoo[itra] )
					h1 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_tra + [x], Y_tra + [y1])
					X_tes = [ list(e) for e in Xoo[ites] ]; Y_tes = list( Yoo[ites] )
					info1 += 1. * accuracy_score( Y_tes, h1.predict(X_tes), normalize = False )
					
				informativeness = info1
				
			elif mtd == "etc": #=====================[GOOD!!] I don't know why it is good ! Test using y2 h2 diff2 max(diff1,diff2)
				if ix in ids[:10]:
					h1 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x], Y_train + [y1])
					diff1 = sum( [ abs(raf.getPredictProba(1,dp,h1)-raf.getPredictProba(1,dp,h)) if h1.predict([dp])[0] != h.predict([dp])[0] else 0. for dp in data if dp != x ] ) / (len(data) - 1.)
					
					# h2 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x], Y_train + [y2])
					# diff2 = sum( [ abs(raf.getPredictProba(1,dp,h2)-raf.getPredictProba(1,dp,h)) if h2.predict([dp])[0] != h.predict([dp])[0] else 0. for dp in data if dp != x ] ) / (len(data) - 1.)
					
					# h12 = svm.SVC(gamma=cf.GAMMA, C=cf.C, random_state = RANDOM_SEED, probability=True).fit(X_train + [x,x], Y_train + [y1,y2])
					# diff12 = sum( [ abs(raf.getPredictProba(1,dp,h12)-raf.getPredictProba(1,dp,h)) if h12.predict([dp])[0] != h.predict([dp])[0] else 0. for dp in data if dp != x ] ) / (len(data) - 1.)

					# diff_12 = sum( [ abs(raf.getPredictProba(1,dp,h1)-raf.getPredictProba(1,dp,h2)) if h1.predict([dp])[0] != h2.predict([dp])[0] else 0. for dp in data if dp != x ] ) / (len(data) - 1.)
					
					
					informativeness = diff1 # this one is particularly good for rejection
					# informativeness = p1*diff1 + p2*diff2 + 1.
					
					# informativeness = diff12
					# informativeness = diff_12
				else: informativeness = 0.
				
			'''
			scores.append( informativeness )
		
		ids = (-np.array(scores)).argsort()
		sorted_scores = [ 1.*scores[id] / sum(scores) for id in ids ]	
		
		return ids, sorted_scores
		
	#---------------------------------------
	#---------------------------------------
	
