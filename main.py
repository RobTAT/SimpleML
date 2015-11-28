import random

from Data import Data
from Explore import Explore
from Visualize import Visualize
from Clustering import Clustering
from Classification import Classification
from Regression import Regression

#-----------------------------------


#-----------------------------------
if __name__ == "__main__":
	random.seed( 1234 )
	
	#-----------------------------------
	# data = Data("data_MSL.mat", "array_slip_ratio")
	data = Data()
	data.loadBusesData("data_buses.mat")
	
	print data.nb_features
	
	#-----------------------------------
	viz = Visualize()
	
	# viz.PCA_Plot(data.X_transpose, fig = "_PCA_.png")
	# viz.plot(data.X_transpose[:3], data.features_name[:3], fig = "vizu.png", color = data.X_transpose[0])
	
	#-----------------------------------
	# clustering = Clustering( data )
	
	# clusters = clustering.kmeans(n_clusters = 3)
	# viz.plot_groups(clusters, fig = "3kmeans.png")
	
	# dists = clustering.dist_to_centers()
	# viz.plot(data.X_transpose, data.features_name, fig = "dists0.png", color = dists[0])
	# viz.plot(data.X_transpose, data.features_name, fig = "dists1.png", color = dists[1])
	# viz.plot(data.X_transpose, data.features_name, fig = "dists2.png", color = dists[2])
	
	#-----------------------------------
	'''
	data.discretize_Y(n_classes = 3)
	classification = Classification( data, method = "svm" )
	
	classification.train()
	print classification.predict( data.X[-1000] )
	
	classes = classification.predict_classes( data.X )
	viz.plot_groups(classes, fig = "svm_classification.png")
	
	data.restore_Y()
	'''
	#-----------------------------------
	'''
	regression = Regression( data, method = "svm" )
	
	regression.train()
	_Y_ = [ regression.predict(x) for x in data.X ]
	viz.plot(data.X_transpose, color = _Y_, fig = "svm_regression.png")
	'''
	#-----------------------------------
	explore = Explore(data)
	explore.fire()
	
	#-----------------------------------
	#-----------------------------------
	#-----------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	