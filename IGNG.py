import numpy as np
import time
import math
from scipy.spatial import distance
import os
from sklearn.neighbors import NearestNeighbors
import mdp
from mdp import graph, Node

from Visualize import Visualize

#======================================================================
class NodeData(object): # Define a node (neuron) of the graph
	def __init__(self, pos, age = 0):
		self.pos = pos # Its coordinates in space (position)
		self.age = age # age of the node
		
#======================================================================

class EdgeData(object): # Define an edge linking two nodes of the graph
	def __init__(self):
		self.age = 0 # age of the edge

#======================================================================
class IGNG(Node):  # Define a graph topology of nodes
	
	@staticmethod
	def estimate_radius(X):
		central = [ np.mean(x) for x in zip(*X) ]
		estimated_radius = np.mean( [ distance.euclidean(x, central) for x in X ] )
		
		print "estimated_radius = ", estimated_radius
		
		return estimated_radius
		
	#---------------------------------------
	def __init__(self, radius, data = None, eps_b=0.2, eps_n=0.006, max_age = 50, mature_age = 5):
		self.data = data # FIXME
		
		self.graph = graph.Graph()
		self.r = radius
		super(IGNG, self).__init__()
		
		self.eps_b = eps_b
		self.eps_n = eps_n
		self.max_age = max_age
		self.mature_age = mature_age
		
		self.nb_nodes = 0
		
	#---------------------------------------
	# Returns the n nearest nodes (in the graph) from the input point x, and their distances to that point
	def getNearestNodes(self, x, n=2, fast = True):
		if fast: return self.getNearestNodes_fast(x,n)
		else: return self.getNearestNodes_slow(x,n)
		
	def getNearestNodes_fast(self, x, n=2):
		positions = [node.data.pos for node in self.graph.nodes]
		h = NearestNeighbors(algorithm='ball_tree', metric='euclidean').fit( positions )
		dists, ids = h.kneighbors(x, n_neighbors=n)
		dists, ids = dists[0], ids[0]
		
		nodes = [self.graph.nodes[i] for i in ids]
		if n < 2: nodes, dists = nodes[0], dists[0] # if n=1 then return one node and one distance
		
		return nodes, dists
		
	def getNearestNodes_slow(self, x, n=2):
		dists = np.array([ distance.euclidean(node.data.pos, x) for node in self.graph.nodes ])
		ids = dists.argsort()[:n]
		
		nodes = [self.graph.nodes[i] for i in ids]
		dists = dists.take(ids)
		
		if n < 2: return nodes[0], dists[0] # if n=1 then return one node and one distance
		else: return nodes, dists # if n>1 then return n nodes and n distances
	
	#---------------------------------------
	# Returns the n nearest mature nodes (in the graph) from the input point x, and their distances to that point
	def getNearestMatureNodes(self, x, n=2, fast = True):
		if fast: return self.getNearestMatureNodes_fast(x,n)
		else: return self.getNearestMatureNodes_slow(x,n)
		
	def getNearestMatureNodes_fast(self, x, n=2):
		positions = [node.data.pos for node in self.graph.nodes if node.data.age > self.mature_age]
		h = NearestNeighbors(algorithm='ball_tree', metric='euclidean').fit( positions )
		dists, ids = h.kneighbors(x, n_neighbors=n)
		dists, ids = dists[0], ids[0]
		
		nodes = [self.graph.nodes[i] for i in ids]
		if n < 2: nodes, dists = nodes[0], dists[0] # if n=1 then return one node and one distance
		
		return nodes, dists
		
	def getNearestMatureNodes_slow(self, x, n=2):
		dists = np.array([ distance.euclidean(node.data.pos, x) for node in self.graph.nodes if node.data.age > self.mature_age ])
		ids = dists.argsort()[:n]
		
		nodes = [self.graph.nodes[i] for i in ids]
		dists = dists.take(ids)
		
		if n < 2: return nodes[0], dists[0] # if n=1 then return one node and one distance
		else: return nodes, dists # if n>1 then return n nodes and n distances
		
	#---------------------------------------
	def get_ccn(self):
		return self.graph.connected_components()
	
	#---------------------------------------
	def get_ccn_pos(self):
		cnn = self.get_ccn()
		return [ [list(n.data.pos) for n in sub_g] for sub_g in cnn ]
		
	#---------------------------------------
	def get_nodes_positions(self):
		return [node.data.pos for node in self.graph.nodes]
	
	#---------------------------------------
	def removeOldEdgesAndIsolatedMatures(self):
		for edge in self.graph.edges:
			if edge.data.age > self.max_age:
				self.graph.remove_edge(edge)
				for n in [edge.head, edge.tail]:
					if n.degree() == 0:
						if n.data.age > self.mature_age:
							self.graph.remove_node(n)
							self.nb_nodes -= 1
	
	#---------------------------------------
	# update the graph using the point x and its associated label y
	def learn(self, x):
		if len(self.graph.nodes) < 2:
			# self.graph.add_node( NodeData(x) )
			self.graph.add_node( NodeData(x, age = 10) )
			self.nb_nodes += 1
			return 
		#------------------------------------------------------
		nodes, dists = self.getNearestNodes(x, 2)
		n1, n2 = nodes[0], nodes[1]
		d1, d2 = dists[0], dists[1]
		#------------------------------------------------------
		if d1 > self.r:
			self.graph.add_node( NodeData(x) )
			self.nb_nodes += 1
		#------------------------------------------------------
		else:
			if d2 > self.r:
				newnode = self.graph.add_node( NodeData(x) )
				self.nb_nodes += 1
				self.graph.add_edge(n1, newnode, EdgeData())
			#------------------------------------------------------
			else:
				for e in n1.get_edges(): e.data.age += 1 # increase age of n1's edges
				
				
				n1.data.pos += self.eps_b*(np.array(x) - np.array(n1.data.pos)) # move n1 and its neighbours
				for n in n1.neighbors(): n.data.pos += self.eps_n*(np.array(x) - np.array(n.data.pos))
				
				if n2 in n1.neighbors(): self.graph.remove_edge( n1.get_edges(n2)[0] ) # link n1 to n2
				self.graph.add_edge(n1, n2, EdgeData())

				for n in n1.neighbors(): n.data.age += 1 # increase age of n1's neighbours
				
				self.removeOldEdgesAndIsolatedMatures()

	#---------------------------------------
	def train( self, X, step = 1, directory = "graph_plots\\"):
		for i, x in enumerate(X):
			self.learn(x)
			# if i%step == 0: self.plot_graph(data = X, iter = i+1, directory = directory)
	
	#---------------------------------------
	def getNearestDist(self, x):
		node, dist = self.getNearestNodes(x, 1)
		return dist

	#---------------------------------------
	def getNearestDistToMature(self, x):
		node, dist = self.getNearestMatureNodes(x, 1)
		return dist

	#---------------------------------------
	def plot_graph(self, data = None, iter = None, directory = "graph_plots\\"): # TODO: this should be generalized and added to Vizualize.py
		viz = Visualize()
		
		if data is not None:	
			viz.do_plot( zip( *data[:iter] ), color = 'y', marker = '.')
			# viz.do_plot( zip( *data[:iter] ), color = self.data.Y[:iter], marker = '.')
		
		matures = [node.data.pos for node in self.graph.nodes if node.data.age > self.mature_age ]
		embryon = [node.data.pos for node in self.graph.nodes if node.data.age <= self.mature_age ]
		if len(matures) > 0: viz.do_plot( zip( *matures ), color = 'r', marker = 'o')
		if len(embryon) > 0: viz.do_plot( zip( *embryon ), color = 'y', marker = 'o')
		
		for e in self.graph.edges:
			pos_head = e.head.data.pos
			pos_tail = e.tail.data.pos
			
			viz.do_plot( zip(* [pos_head, pos_tail] ) , color = 'r', marker='-')
		
		
		if not os.path.exists(directory): os.makedirs(directory)
		
		filename = str(time.time()) + '.png'
		
		
		if iter is None: viz.end_plot(fig = directory+'_'+filename)
		else: viz.end_plot(fig = directory+filename)
		
	#---------------------------------------
