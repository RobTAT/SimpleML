import random
from Data import Data
from ActiveLearning import ActiveLearning

#-----------------------------------
if __name__ == "__main__":
	random.seed( 12345 )
	
	#-----------------------------------
	data = Data( source_file = "weka" )
	print "nb data points:", len(data.X)
	print "nb features in data:", data.nb_features
	
	#-----------------------------------
	al = ActiveLearning( data.X[:50], data.Y[:50], data.X[50:], data.Y[50:], data.Tx, data.Ty )
	# al.train( mtd = "margin" )
	# al.train( mtd = "proba" )
	# al.train( mtd = "entropy" )
	# al.train( mtd = "random" )
	al.train( mtd = "weight" )
	
	#-----------------------------------
	# classification = Classification( data.X, data.Y, method = "svm" )
	
	# classification.train()
	# print classification.predict( data.X[-511] )
	# print classification.getLabelOf( 1, data.X[-511] )
	# print classification.getProbaOf( 8, data.X[-511] )
	# print classification.getPredictProba( 1, data.X[-511] )
	
	# print classification.uncertainty_margin( data.X[-511] )
	# print classification.uncertainty_prediction( data.X[-511] )
	# print classification.uncertainty_entropy( data.X[-511] )
	
	# print classification.getMarginInfo( data.X[-511] )
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	