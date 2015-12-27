import random
from Data import Data
from ActiveLearning import ActiveLearning
from Util import Util
from Visualize import Visualize

#-----------------------------------
if __name__ == "__main__":
	random.seed( 12345 )
	
	#-----------------------------------
	data = Data( source_file = "weka" )
	print "nb data points:", len(data.X)
	print "nb features in data:", data.nb_features
	
	#-----------------------------------
	AL_Method = "etc" # margin proba entropy random weight expectedErrorReduction etc
	AL_Init = 50 # margin proba entropy random weight expectedErrorReduction etc
	
	al = ActiveLearning( data.X[:AL_Init], data.Y[:AL_Init], data.X[AL_Init:], data.Y[AL_Init:], data.Tx, data.Ty )
	al.train( mtd = AL_Method )
	
	filename = "_optdigits."+AL_Method+"."+str(AL_Init)+".opt-"+str(al.optimization_limit)+"-"+al.optimization_method+".txt"
	Util.pickleSave(filename, al)
	al = Util.pickleLoad(filename)
	
	viz = Visualize()
	viz.plot( [range(len(al.accuracys)), al.accuracys], fig = filename+".png", color = 'r', marker = '-' )
	
	#-----------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	