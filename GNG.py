import numpy as np
import time
import os
from scipy.spatial import distance
import mdp

from Visualize import Visualize

class GNG:
	def __init__(self, data = None, max_nodes=2147483647, eps_b=0.2, eps_n=0.006, max_age = 50, period=100, d=0.995, alpha=0.5):
		self.data = data # FIXME
		self.gng = mdp.nodes.GrowingNeuralGasNode(max_nodes = max_nodes, eps_b=eps_b, eps_n=eps_n, max_age=max_age, lambda_=period, d=d, alpha=alpha)
		
		self.nb_nodes = 0
		
	#---------------------------------------
	def train(self, X, step = None, directory = "graph_plots\\"):
		if step is None:
			self.gng.train( np.array( X ) )
			# self.plot_graph()
		else:
			for i in range( 0, len(X), step ):
				self.gng.train( np.array( X[i:i+step] ) )
				self.plot_graph(data = X, iter = i+step, directory = directory)
				# self.plot_graph(iter = i+step, directory = directory)
		
		# self.gng.stop_training() # FIXME
		self.nb_nodes = len( self.gng.get_nodes_position() )
		
	#---------------------------------------
	def get_ccn(self):
		return self.gng.graph.connected_components()
	
	#---------------------------------------
	def get_ccn_pos(self):
		cnn = self.get_ccn()
		return [ [list(n.data.pos) for n in sub_g] for sub_g in cnn ]
	
	#---------------------------------------
	def get_nodes_positions(self):
		return self.gng.get_nodes_position()
		
	#---------------------------------------
	def getNearestDist(self, x):
		dists = np.array([ distance.euclidean(pos, x) for pos in self.get_nodes_positions() ])
		ids = dists.argsort()
		dists = dists.take(ids)
		return dists[0]
		
	#---------------------------------------
	def plot_graph(self, data = None, iter = None, directory = "graph_plots\\"): # TODO: this should be generalized and added to Vizualize.py
		viz = Visualize()
		
		if data is not None:	
			viz.do_plot( zip( *data[:iter] )[:3], color = 'y', marker = '.')
			# viz.do_plot( zip( *data[:iter] )[:3], color = self.data.Y[:iter], marker = '.')
		
		viz.do_plot( zip( *self.get_nodes_positions() )[:3], color = 'r', marker = 'o')
		
		for e in self.gng.graph.edges:
			pos_head = e.head.data.pos
			pos_tail = e.tail.data.pos
			
			viz.do_plot( zip(* [pos_head, pos_tail] )[:3], color = 'r', marker='-')
		
		
		if not os.path.exists(directory): os.makedirs(directory)
		
		filename = str(time.time()) + '.png'
		
		
		if iter is None: viz.end_plot(fig = directory+'_'+filename)
		else: viz.end_plot(fig = directory+filename)
		
	#---------------------------------------
