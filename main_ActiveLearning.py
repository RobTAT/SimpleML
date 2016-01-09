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
	
	data = Data( source_file = "weka" )
	print "nb data points:", len(data.X)
	print "nb features in data:", data.nb_features
	
	#-----------------------------------
	
	al_opt_random = Util.pickleLoad('___AL Results\\optdigits\\random\\_optdigits.random.50.opt-10-margin.txt')
	al_opt_entropy = Util.pickleLoad('___AL Results\\optdigits\\entropy\\_optdigits.entropy.50.opt-10-margin.txt')
	al_opt_margin = Util.pickleLoad('___AL Results\\optdigits\\margin\\_optdigits.margin.50.opt-10-margin.txt')
	al_opt_proba = Util.pickleLoad('___AL Results\\optdigits\\proba\\_optdigits.proba.50.opt-10-margin.txt')
	al_opt_weight_20_entropy = Util.pickleLoad('___AL Results\\optdigits\\weight\\_optdigits.weight.50.opt-20-entropy.txt')
	al_opt_etccc_20_margin = Util.pickleLoad('___AL Results\\optdigits\\etc\\_optdigits.etc_.50.opt-20-margin.txt')
	al_opt_etc_20_margin = Util.pickleLoad('___AL Results\\optdigits\\etc\\_optdigits.etc.50.opt-20-margin.txt')
	
	al_pen_random = Util.pickleLoad('___AL Results\\pendigit\\random\\_pendigits.random.50.opt-10-margin.txt')
	al_pen_entropy = Util.pickleLoad('___AL Results\\pendigit\\entropy\\_pendigits.entropy.50.opt-10-margin.txt')
	al_pen_margin = Util.pickleLoad('___AL Results\\pendigit\\margin\\_pendigits.margin.50.opt-10-margin.txt')
	al_pen_proba = Util.pickleLoad('___AL Results\\pendigit\\proba\\_pendigits.proba.50.opt-10-margin.txt')
	al_pen_weight_20_margin = Util.pickleLoad('___AL Results\\pendigit\\weight\\_pendigits.weight.50.opt-20-margin.txt')
	al_pen_etccc_20_margin = Util.pickleLoad('___AL Results\\pendigit\\etc\\_pendigits.etc_.50.opt-20-margin.txt')
	al_pen_etc_20_margin = Util.pickleLoad('___AL Results\\pendigit\\etc\\_pendigits.etc.50.opt-20-margin.txt')
	
	viz = Visualize()
	colors = ['y','c','m','b','g','k','r']
	
	# for ic,al in enumerate([al_opt_random, al_opt_entropy, al_opt_margin, al_opt_proba, al_opt_weight_20_entropy, al_opt_etccc_20_margin, al_opt_etc_20_margin]):
	for ic,al in enumerate([al_opt_random, al_opt_random, al_opt_margin, al_opt_proba, al_opt_random, al_opt_etccc_20_margin, al_opt_etc_20_margin]):
		print colors[ic], "==>", np.mean(al.accuracys[:100]), np.mean(al.accuracys[:200]), np.mean(al.accuracys[:300])
		viz.do_plot( [range(300), al.accuracys[:300]], color = colors[ic], marker = '-' )
	viz.end_plot( fig = "AL_opt.png" )
	print "------------"
	# for ic,al in enumerate([al_pen_random, al_pen_entropy, al_pen_margin, al_pen_proba, al_pen_weight_20_margin, al_pen_etccc_20_margin, al_pen_etc_20_margin]):
	for ic,al in enumerate([al_pen_random, al_pen_random, al_pen_margin, al_pen_proba, al_pen_random, al_pen_etccc_20_margin, al_pen_etc_20_margin]):
		print colors[ic], "==>", np.mean(al.accuracys[:100]), np.mean(al.accuracys[:200]), np.mean(al.accuracys[:300])
		viz.do_plot( [range(300), al.accuracys[:300]], color = colors[ic], marker = '-' )
	viz.end_plot( fig = "AL_pen.png" )
	
	#-----------------------------------
	'''
	AL_Method = "test" # margin proba entropy random etc expectedErrorReduction weight test
	AL_Init = 50
	
	al = ActiveLearning( data.X[:AL_Init], data.Y[:AL_Init], data.X[AL_Init:], data.Y[AL_Init:], data.Tx, data.Ty )
	al.train( mtd = AL_Method )
	
	filename = "_opt."+AL_Method+"."+str(AL_Init)+".opt-"+str(al.optimization_limit)+"-"+al.optimization_method+".txt"
	# filename = "_pendigits."+AL_Method+"."+str(AL_Init)+".opt-"+str(al.optimization_limit)+"-"+al.optimization_method+".txt"
	Util.pickleSave(filename, al)
	al = Util.pickleLoad(filename)
	
	viz = Visualize()
	viz.plot( [range(len(al.accuracys)), al.accuracys], fig = filename+".png", color = 'r', marker = '-' )
	'''
	#-----------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	