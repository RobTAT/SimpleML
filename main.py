import random

from Data import Data
from Explore import Explore
from Visualize import Visualize
from Clustering import Clustering
from Classification import Classification
from Regression import Regression

if __name__ == "__main__":
	random.seed( 1234 )
	
	#-----------------------------------
	data = Data("MIT_dataset.mat", "array_slip_ratio")
	
	print len(data.Y)
	print len(data.X)
	print len(data.X_transpose)
	print data.nb_data
	print data.nb_features
	print data.features_name
	print data.target_name
	
	#-----------------------------------
	viz = Visualize()
	
	viz.plot(data.X_transpose, data.features_name, figure_name = "vizu.png", color = data.Y)
	
	viz.do_plot(data.X_transpose)
	viz.end_plot()
	
	#-----------------------------------
	clustering = Clustering( data )
	
	clusters = clustering.kmeans(n_clusters = 2)
	viz.plot_groups(clusters, figure_name = "2kmeans.png")
	
	clusters = clustering.kmeans(n_clusters = 3)
	viz.plot_groups(clusters, figure_name = "3kmeans.png")
	
	clusters = clustering.dbscan(eps = 0.2, min_samples = 100)
	viz.plot_groups(clusters, figure_name = "1dbscan.png")
	
	#-----------------------------------
	data.discretize_Y(n_classes = 3)
	classification = Classification( data, method = "svm" )
	
	classification.train()
	print classification.predict( data.X[-1000] )
	
	classes = classification.predict_classes( data.X )
	viz.plot_groups(classes, figure_name = "svm_classification.png")
	
	data.restore_Y()
	
	#-----------------------------------
	regression = Regression( data, method = "svm" )
	
	regression.train()
	_Y_ = [ regression.predict(x) for x in data.X ]
	viz.plot(data.X_transpose, color = _Y_, figure_name = "svm_regression.png")
	
	#-----------------------------------
	explore = Explore(data)
	explore.fire()
	
	
	#-----------------------------------
	#-----------------------------------
	#-----------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	