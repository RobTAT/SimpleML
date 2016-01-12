import random
import numpy as np
from Data import Data
from ActiveLearning import ActiveLearning
from Util import Util
from Visualize import Visualize

#-----------------------------------
if __name__ == "__main__":
	random.seed( 12345 )
	
	#-----------------------------------
	viz = Visualize()
	colors = ['y','c','m','b','g','k','r']
	
	datasetname = "iris"
	data = Data( source_file = datasetname )
	print "nb data points:", len(data.X), "nb features in data:", data.nb_features
	
	#-----------------------------------
	'''
	opt = {}
	opt["random"] = Util.pickleLoad('___AL Results\\optdigits\\random\\_optdigits.random.50.opt-10-margin.txt')
	# opt["entropy"] = Util.pickleLoad('___AL Results\\optdigits\\entropy\\_optdigits.entropy.50.opt-10-margin.txt')
	# opt["margin"] = Util.pickleLoad('___AL Results\\optdigits\\margin\\_optdigits.margin.50.opt-10-margin.txt')
	opt["proba"] = Util.pickleLoad('___AL Results\\optdigits\\proba\\_optdigits.proba.50.opt-10-margin.txt')
	# opt["weight"] = Util.pickleLoad('___AL Results\\optdigits\\weight\\_optdigits.weight.50.opt-20-entropy.txt')
	opt["etc_"] = Util.pickleLoad('___AL Results\\optdigits\\etc\\_optdigits.etc_.50.opt-20-margin.txt')
	opt["etc"] = Util.pickleLoad('___AL Results\\optdigits\\etc\\_optdigits.etc.50.opt-20-margin.txt')
	
	for ic,key in enumerate(opt):
		al = opt[key]
		print key, "==> color=", colors[ic], "==>", np.mean(al.accuracys[:100]), np.mean(al.accuracys[:200]), np.mean(al.accuracys[:300])
		viz.do_plot( [range(300), al.accuracys[:300]], color = colors[ic], marker = '-' )
	viz.end_plot( fig = "AL_opt.png" )
	print ""
	
	pen = {}
	pen["random"] = Util.pickleLoad('___AL Results\\pendigit\\random\\_pendigits.random.50.opt-10-margin.txt')
	# pen["entropy"] = Util.pickleLoad('___AL Results\\pendigit\\entropy\\_pendigits.entropy.50.opt-10-margin.txt')
	# pen["margin"] = Util.pickleLoad('___AL Results\\pendigit\\margin\\_pendigits.margin.50.opt-10-margin.txt')
	pen["proba"] = Util.pickleLoad('___AL Results\\pendigit\\proba\\_pendigits.proba.50.opt-10-margin.txt')
	# pen["weight"] = Util.pickleLoad('___AL Results\\pendigit\\weight\\_pendigits.weight.50.opt-20-margin.txt')
	pen["etc_"] = Util.pickleLoad('___AL Results\\pendigit\\etc\\_pendigits.etc_.50.opt-20-margin.txt')
	pen["etc"] = Util.pickleLoad('___AL Results\\pendigit\\etc\\_pendigits.etc.50.opt-20-margin.txt')
	
	for ic,key in enumerate(pen):
		al = pen[key]
		print key, "==> color=", colors[ic], "==>", np.mean(al.accuracys[:100]), np.mean(al.accuracys[:200]), np.mean(al.accuracys[:300])
		viz.do_plot( [range(300), al.accuracys[:300]], color = colors[ic], marker = '-' )
	viz.end_plot( fig = "AL_pen.png" )
	print ""
	
	'''
	#-----------------------------------
	# '''
	AL_Method = "etc_" # margin proba entropy random etc, etc_ weight  //  expectedErrorReduction test
	AL_Init = 50
	
	al = ActiveLearning( data.X[:AL_Init], data.Y[:AL_Init], data.X[AL_Init:], data.Y[AL_Init:], data.Tx, data.Ty )
	al.train( mtd = AL_Method )
	
	filename = datasetname+"."+AL_Method+"."+str(AL_Init)+".opt-"+str(al.optimization_limit)+"-"+al.optimization_method+".txt"
	Util.pickleSave(filename, al)
	al = Util.pickleLoad(filename)
	
	viz = Visualize()
	viz.plot( [range(len(al.accuracys)), al.accuracys], fig = filename+".png", color = 'r', marker = '-' )
	# '''
	#-----------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	