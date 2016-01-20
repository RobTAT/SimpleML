import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import Util
from Parameters import *

from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import math
from scipy.stats import norm
from GNG import GNG
from IGNG import IGNG

#---------------------------------------
class AnomalyModel:
	def __init__(self, trainingSet, anomalyMethod = "KNN", h = None ):
		self.method = anomalyMethod
		
		if self.method == "online":
			self.h = h
		
		if self.method == "IGNG":
			self.h = IGNG( radius = PARAMS["R"] ) # IGNG.estimate_radius( trainingSet )
			self.h.train( trainingSet )
			# print len( self.h.get_nodes_positions() ), len(trainingSet)
			
		if self.method == "GNG":
			self.h = GNG(period = 50)
			self.h.train( trainingSet )
			
		if self.method == "KNN":
			self.h = NearestNeighbors(algorithm='ball_tree', metric='euclidean').fit(trainingSet)
			
		elif self.method == "RNN":
			self.h = NearestNeighbors(algorithm='ball_tree', metric='euclidean').fit(trainingSet)
			
		elif self.method == "SVM":
			self.h = svm.OneClassSVM(nu=PARAMS["NU"], kernel="rbf", gamma=PARAMS["GAMMA"]).fit(trainingSet)
			
	def getAnomalyScore(self, x, inversed = False):
		if self.method == "online":
			# alpha_m = self.h.getNearestDist(x)
			# alpha_m = 1. if self.h.getNearestDist(x) > PARAMS["R"] else 0.
			alpha_m = 1. if self.h.getNearestDistToMature(x) > PARAMS["R"] else 0.
			if inversed == True: alpha_m = 1. / alpha_m
			
		if self.method == "IGNG":
			alpha_m = self.h.getNearestDist(x)
			if inversed == True: alpha_m = 1. / alpha_m
			
		if self.method == "GNG":
			alpha_m = self.h.getNearestDist(x)
			if inversed == True: alpha_m = 1. / alpha_m
			
		if self.method == "KNN":
			distances, indices = self.h.kneighbors( x, n_neighbors = PARAMS["K"] )
			alpha_m = sum( distances[0] )
			if inversed == True: alpha_m = 1. / alpha_m
			
		elif self.method == "RNN":
			distances, indices = self.h.radius_neighbors(x, radius = PARAMS["R"])
			alpha_m = 1. / ( 1. + sum( [ 1./di for di in distances[0] if di != 0 ] ) )
			if inversed == True: alpha_m = 1. / alpha_m
		
		elif self.method == "SVM":
			alpha_m = -1. * self.h.decision_function(x)[0][0]
			if inversed == True: alpha_m = -1. * alpha_m
		
		return alpha_m
	
################################################################################################
def getPValue_V1( all_buses, id_bus, i, alpha_m, anomalyMethod = "KNN", h = None ):
	alphas = []
	for id_bus_j in range( len(all_buses) ):
		if id_bus_j == id_bus: continue;
		
		own = all_buses[id_bus_j]
		fleet = all_buses[:id_bus_j] + all_buses[id_bus_j+1 :]
		
		own_ = Util.shrink(i, own, TH1)
		fleet_ = [ Util.shrink(i, bus, TH1) for bus in fleet ]
		flat_fleet_ = Util.flatList( fleet_ )
		
		model = AnomalyModel(flat_fleet_, anomalyMethod, h)
		for his_j in own_:
			alpha = model.getAnomalyScore(his_j)
			alphas += [alpha]
	
	return len( [d for d in alphas if d >= alpha_m] )*1. / len(alphas)
	# return sum( [d for d in alphas if d >= alpha_m] )*1. / sum(alphas)

################################################################################################
def getPValue_V2( all_buses, id_bus, i, alpha_m, anomalyMethod = "KNN", h = None ):
	alphas = []
	for id_bus_j in range( len(all_buses) ):
		if id_bus_j == id_bus: continue;
		
		own = all_buses[id_bus_j]
		fleet = all_buses[:id_bus_j] + all_buses[id_bus_j+1 :]
		
		own_ = Util.shrink(i, own, TH1)
		fleet_ = [ Util.shrink(i, bus, TH1) for bus in fleet ]
		flat_fleet_ = Util.flatList( fleet_ )
		
		model = AnomalyModel(own_, anomalyMethod, h)
		for his_j in own_:
			alpha = model.getAnomalyScore(his_j, inversed = True)
			alphas += [alpha]
	
	return 1.*len( [d for d in alphas if d >= alpha_m] ) / len(alphas)
	# return 1.*sum( [d for d in alphas if d >= alpha_m] ) / sum(alphas)

################################################################################################
def getZvalues( Z, mean_popu = 0.5, var_popu = 1./12, th = TH2):
	std_popu = math.sqrt(var_popu) # std deviation for the uniform distribution
	
	# Z_means = Util.weightedMovingAverage(Z, th)
	Z_means = Util.movingAverage(Z, th)
	Z_pvalues = [ norm.cdf( ( mean_z - mean_popu ) / ( std_popu/math.sqrt(1.*th) ) ) for mean_z in Z_means ]
	
	zeros = [0] * ( len(Z) - len(Z_means) )
	Z_means = zeros + Z_means
	Z_pvalues = zeros + Z_pvalues
	Z_pvalues = [ 0. if v <= 0 else -math.log(v) for v in Z_pvalues ]
	
	return Z_means, Z_pvalues

################################################################################################
def estimateRadius(data):
	central = [ numpy.mean(elem) for elem in zip(*data) ]
	return numpy.mean( [ Util.dist(elem, central) for elem in data ] )

################################################################################################
def normalityProba_V1( method, flat_fleet_test_, his_test, all_buses, id_bus, i, h = None ):
	model = AnomalyModel(flat_fleet_test_, method, h)
	alpha_m = model.getAnomalyScore(his_test)
	pvalue = getPValue_V1( all_buses, id_bus, i, alpha_m, method, h )
	return pvalue, alpha_m

################################################################################################
def normalityProba_V2( method, flat_fleet_test_, his_test, all_buses, id_bus, i, h = None ):
	model = AnomalyModel(flat_fleet_test_, method, h)
	alpha_m = model.getAnomalyScore(his_test, inversed = True)
	pvalue = getPValue_V2( all_buses, id_bus, i, alpha_m, method, h )
	return pvalue, alpha_m

	

################################################################################################
