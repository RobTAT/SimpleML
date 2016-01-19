'''
| This is to explore your data and then decide what operations you need to perform on that data
'''

from Visualize import Visualize

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Explore:
	def __init__(self, data):
		self.data = data
		self.viz = Visualize()

	#---------------------------------------
	def fire(self):
		range_features = range( len(self.data.X_transpose) )
		
		for i in range_features:
			axs = [ self.data.X_transpose[i] ]
			axs_labels = [ self.data.features_name[i] ]
			self.viz.plot(axs, axs_labels = axs_labels, color = self.data.Y, marker = '.', fig = "explore_1D_"+str(i)+".png")
		
		pairs = [ (i,j) for i in range_features for j in range_features ]
		for pair in pairs:
			if pair[0] != pair[1]:
				axs = [self.data.X_transpose[id] for id in pair]
				axs_labels = [self.data.features_name[id] for id in pair]
				self.viz.plot(axs, axs_labels = axs_labels, color = self.data.Y, marker = '.', fig = "explore_2D_"+str(pair)+".png")
			
		triplets = [ (i,j,k) for i in range_features for j in range_features for k in range_features ]
		for triplet in triplets:
			if triplet[0] != triplet[1] and triplet[1] != triplet[2] and triplet[0] != triplet[2]:
				axs = [self.data.X_transpose[id] for id in triplet]
				axs_labels = [self.data.features_name[id] for id in triplet]
				self.viz.plot(axs, axs_labels = axs_labels, color = self.data.Y, marker = '.', fig = "explore_3D_"+str(triplet)+".png")
			
	#---------------------------------------
	
	