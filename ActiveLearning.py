import numpy as np
import random
import operator
from Util import Util
from Visualize import Visualize
from sklearn import svm, neighbors
from Classification import Classification

class ActiveLearning:
	def __init__(self, Lx, Ly, Ux, Uy, Tx, Ty, method = "svm", budget = 1000):
		self.Lx = Lx
		self.Ly = Ly
		self.Ux = Ux
		self.Uy = Uy # TODO should not be here
		self.Tx = Tx # TODO should not be here
		self.Ty = Ty # TODO should not be here
		
		self.optimization_limit = 20
		self.optimization_method = "margin" # margin proba entropy random weight expectedErrorReduction etc
		
		self.budget = budget
		self.accuracys = []
		
		self.clf = Classification( self.Lx, self.Ly, method = method, Vx = Lx+Ux, Vy = Ly+Uy )
		self.clf.train()
		
		self.viz_A = []; self.viz_B = []; self.viz_C = []; self.viz_D = []; self.viz_E = []; self.viz_F = []
		
	#---------------------------------------
	def train(self, mtd = "margin", backupfile = "backupfile"): # TODO implement sample_weight + make method to shuffle and return sublist with data_limit
		backupfile += ".opt-"+str(self.optimization_limit)+"-"+self.optimization_method+".txt"
		for i in range(self.budget):
			if len(self.Ux) <= 1: break
			# self.viz_A = []; self.viz_B = []; self.viz_C = []; self.viz_D = []; self.viz_E = []; self.viz_F = []
			
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
			
			print "i=", i+1, "; acc=%.4f"%(test_accuracy*100), "%.4f"%(np.mean(self.accuracys)*100), "%.4f"%(np.average(self.accuracys, weights = range(1,1+len(self.accuracys)))*100), scores[0]
			
			if (i+1)%10 == 0:
				Util.pickleSave(backupfile, self)
				viz = Visualize(); viz.plot( [range(len(self.accuracys)), self.accuracys], fig = backupfile+".png", color = 'r', marker = '-' )
			
	#---------------------------------------
	def sortForInformativeness(self, mtd):
		if mtd in ["etc", "etc_", "expectedErrorReduction", "weight", "optimal", "test", "intuition"] :
			ids, scores = self.sortForInformativeness(self.optimization_method)
			
		scores = []
		for ix, x in enumerate(self.Ux):
			y1, y2, p1, p2 = self.clf.getMarginInfo(x)
			
			if mtd == "intuitionM":
				if ix in ids[:self.optimization_limit]:
					informativeness = self.clf.uncertainty_margin(x)
				else:
					informativeness = 0.
			#---------------------------------------------------------
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
						temp_clf = Classification(self.Lx + [x], self.Ly + [yy], method = self.clf.method); temp_clf.GAMMA, temp_clf.C = self.clf.GAMMA, self.clf.C # TODO FIXME: do it in general not specifically for svm
						temp_clf.train()
						e_h1 = sum( [ temp_clf.uncertainty_entropy(dp) for dp in self.Ux if dp != x ] )
						
						sums += (proba) * e_h1
					informativeness = 1. / sums
				else:
					informativeness = 0.
			
			#---------------------------------------------------------
			elif mtd == "etc":
				if ix in ids[:self.optimization_limit]:
					temp_clf1 = Classification(self.Lx + [x], self.Ly + [y1], method = self.clf.method); temp_clf1.GAMMA, temp_clf1.C = self.clf.GAMMA, self.clf.C
					temp_clf1.train()
					diff1 = sum( [ abs(temp_clf1.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) if temp_clf1.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					temp_clf2 = Classification(self.Lx + [x], self.Ly + [y2], method = self.clf.method); temp_clf2.GAMMA, temp_clf2.C = self.clf.GAMMA, self.clf.C
					temp_clf2.train()
					diff2 = sum( [ abs(temp_clf2.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) if temp_clf2.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					informativeness = diff1 # this one is particularly good for rejection (to be confirmed)
					informativeness = diff1 if p1/(1+diff1) >= p2/(1+diff2) else diff2
					informativeness = p1*diff1 + p2*diff2 + 1.
				else:
					informativeness = 0.
				
			#---------------------------------------------------------
			elif mtd == "etc_":
				if ix in ids[:self.optimization_limit]:
					temp_clf1 = Classification(self.Lx + [x], self.Ly + [y1], method = self.clf.method); temp_clf1.GAMMA, temp_clf1.C = self.clf.GAMMA, self.clf.C
					temp_clf1.train()
					diff1 = sum( [ 1. if temp_clf1.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					temp_clf2 = Classification(self.Lx + [x], self.Ly + [y2], method = self.clf.method); temp_clf2.GAMMA, temp_clf2.C = self.clf.GAMMA, self.clf.C
					temp_clf2.train()
					diff2 = sum( [ 1. if temp_clf2.predict_label(dp) != self.clf.predict_label(dp) else 0. for dp in self.Ux if dp != x ] ) / (len(self.Ux) - 1.)
					
					informativeness = diff1 # this one is particularly good for rejection (to be confirmed)
					informativeness = diff1 if p1/(1+diff1) >= p2/(1+diff2) else diff2
					informativeness = p1*diff1 + p2*diff2 + 1.
				else:
					informativeness = 0.
				
			#---------------------------------------------------------
			elif mtd == "test":
				if ix in ids[:self.optimization_limit]:
					temp_clf1 = Classification(self.Lx + [x], self.Ly + [y1], method = self.clf.method); temp_clf1.GAMMA, temp_clf1.C = self.clf.GAMMA, self.clf.C
					temp_clf1.train()
					diff1 = np.mean( [0.]+[ abs(temp_clf1.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) for dp in self.Ux if temp_clf1.predict_label(dp) != self.clf.predict_label(dp) and dp != x ] )
					
					temp_clf2 = Classification(self.Lx + [x], self.Ly + [y2], method = self.clf.method); temp_clf2.GAMMA, temp_clf2.C = self.clf.GAMMA, self.clf.C
					temp_clf2.train()
					diff2 = np.mean( [0.]+[ abs(temp_clf2.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp)) for dp in self.Ux if temp_clf2.predict_label(dp) != self.clf.predict_label(dp) and dp != x ] )
					
					informativeness = diff1 # this one is particularly good for rejection (to be confirmed)
					informativeness = diff1 if p1/(1+diff1) >= p2/(1+diff2) else diff2
					informativeness = p1*diff1 + p2*diff2 + 1.
				else:
					informativeness = 0.
				
			#---------------------------------------------------------
			elif mtd == "intuition":
				if ix in ids[:self.optimization_limit]:
					true_y = self.Uy[ self.Ux.index(x) ]
					
					temp_clf = Classification(self.Lx + [x], self.Ly + [true_y], method = self.clf.method)
					temp_clf.GAMMA, temp_clf.C = self.clf.GAMMA, self.clf.C; temp_clf.train()
					
					# ---------------------
					imp_x = [ xdp for xdp in self.Tx if temp_clf.predict_label(xdp) != self.clf.predict_label(xdp) ]
					imp_y_hh = [ temp_clf.predict_label(xdp) for xdp in self.Tx if temp_clf.predict_label(xdp) != self.clf.predict_label(xdp) ]
					
					if len( set(imp_y_hh) ) > 1: 
						# hh = Classification(imp_x, imp_y_hh, method = self.clf.method)
						hh = Classification(imp_x + [x], imp_y_hh + [true_y], method = self.clf.method, tuning = False)
						hh.GAMMA, hh.C = self.clf.GAMMA, self.clf.C; hh.train()
					else:
						hh = self.clf
					# ---------------------
					
					h_inconsistant_truth = 0; hh_inconsistant_truth = 0; hh_inconsistant_h = 0; h_consistency = []; hh_consistency = []
					for ilx, lx in enumerate(self.Lx):
						h_consistency.append( self.clf.getProbaOf( self.Ly[ilx], lx ) )
						# hh_consistency.append( hh.getProbaOf( self.Ly[ilx], lx ) )
						hh_consistency.append( hh.getProbaOf( self.Ly[ilx], lx ) if hh.predict_label(lx) == self.Ly[ilx] else 0. )
						
						if self.clf.predict_label(lx) != self.Ly[ilx]: h_inconsistant_truth += 1.
						if hh.predict_label(lx) != self.Ly[ilx]: hh_inconsistant_truth += 1.
						if hh.predict_label(lx) != self.clf.predict_label(lx): hh_inconsistant_h += 1.
					h_consistency = np.mean(h_consistency)
					hh_consistency = np.mean(hh_consistency) if len( set(imp_y_hh) ) > 1 else 0.
					
					consistency_dif = hh_consistency - h_consistency
					
					# ---------------------
					diff = []; errors = 0.; trues = 0.; impacted = 0; impacted_probs = [];
					for idp, dp in enumerate(self.Tx):
						if temp_clf.predict_label(dp) != self.clf.predict_label(dp): ##################
							impacted += 1.
							impacted_probs.append( abs( temp_clf.getPredictProba(1,dp) - self.clf.getPredictProba(1,dp) ) )
							if self.Ty[idp]!=temp_clf.predict_label(dp): errors += 1.
							else: trues += 1.
						
						# if temp_clf.predict_label(dp) != self.clf.predict_label(dp) and self.Ty[idp]==temp_clf.predict_label(dp): diff.append( 1. )
						# if temp_clf.predict_label(dp) != self.clf.predict_label(dp) and trues - errors > 0: diff.append( 1. )
						# if temp_clf.predict_label(dp) != self.clf.predict_label(dp): diff.append( 1. )
						
						if temp_clf.predict_label(dp) != self.clf.predict_label(dp): diff.append( 1. )
						
						else: diff.append( 0. )
					diff = np.mean( diff )
					
					# diff = diff * np.mean(impacted_probs) # seems to be working ...
					
					# ---------------------
					# self.viz_A.append( consistency_dif )
					self.viz_A.append( hh_consistency )
					self.viz_B.append( errors )
					self.viz_C.append( trues )
					self.viz_D.append( trues - errors ); posI = [inb for inb,nbD in enumerate(self.viz_D) if nbD >= 0.]
					self.viz_E.append( impacted )
					self.viz_F.append( np.mean(impacted_probs) )
					viz = Visualize(); viz.plot( [self.viz_A, self.viz_B], fig = "test_errors.png", color = 'r', marker = 'o' )
					vizu = Visualize(); vizu.plot( [self.viz_A, self.viz_C], fig = "test_trues.png", color = 'r', marker = 'o' )
					vizuu = Visualize(); vizuu.plot( [self.viz_A, self.viz_D], fig = "test_trues_errors.png", color = 'r', marker = 'o' )
					
					vizuuu = Visualize(); vizuuu.do_plot( [self.viz_A, self.viz_E], color = 'r', marker = 'o' )
					vizuuu.do_plot( [[self.viz_A[inb] for inb in posI], [self.viz_E[inb] for inb in posI]], color = 'b', marker = 'o' )
					vizuuu.end_plot(fig = "impacted.png")
					
					print hh_consistency, hh_inconsistant_truth, "---", len(imp_x), len( set(imp_y_hh) ), "============>", impacted, trues - errors
					
					informativeness = diff
				else:
					informativeness = 0.
				
			#---------------------------------------------------------
			
			scores.append( informativeness )
		
		ids = (-np.array(scores)).argsort()
		sorted_scores = [ scores[id] for id in ids ]	
		# sorted_scores = [ 1.*scores[id] / sum(scores) for id in ids ]	
		
		return ids, sorted_scores
		
	#---------------------------------------
	#---------------------------------------
	
