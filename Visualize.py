import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualize:
	def __init__(self):
		self.w = 15 # width of the plots
		self.h = 10 # hight of the plots
		
		self.cmap = plt.copper()
		self.lw = 0
		self.s = 20
		
		self.plots = None
		self.xyz_range = { 'x':[float("inf"), float("-inf")], 'y':[float("inf"), float("-inf")], 'z':[float("inf"), float("-inf")] }
	
	#---------------------------------------
	def start_plot( self, axs_labels ):
		if len(axs_labels) < 3:
			fig, self.plots = plt.subplots( 1, 1, sharex=False )
			fig.set_size_inches(self.w, self.h)
			
			self.plots.set_xlabel(axs_labels[0])
			self.plots.set_ylabel(axs_labels[1])
		else:
			fig = plt.figure()
			self.plots = fig.add_subplot(111, projection='3d')
			fig.set_size_inches(self.w, self.h)
			
			self.plots.set_xlabel(axs_labels[0])
			self.plots.set_ylabel(axs_labels[1])
			self.plots.set_zlabel(axs_labels[2])

			
	#---------------------------------------
	def do_plot(self, axs, axs_labels = None, color = 'r', marker = '.'): # FIXME what is len(axs) is > 3 ? Use Multidim Scaling or Feature selection
		if axs_labels is None:
			axs_labels = [ "Axis "+str(i+1) for i in range( len(axs) ) ]
		
		if len(axs) == 1:
			axs = [ range( len(axs[0]) ) ] + axs
			axs_labels = [ "Samples" ] + axs_labels
		
		if self.plots is None:
			self.start_plot( axs_labels )
		
		# print axs_labels
		self.plots.scatter( *axs, c = color, marker = marker, lw = self.lw, s = self.s, cmap = self.cmap )
		
		# Re adjusting the xrange, yrange and zrange limits
		min_x, max_x = self.xyz_range['x']; self.xyz_range['x'] = [ min( min_x, min(axs[0]) ), max( max_x, max(axs[0]) ) ]
		min_y, max_y = self.xyz_range['y']; self.xyz_range['y'] = [ min( min_y, min(axs[1]) ), max( max_y, max(axs[1]) ) ]
		self.plots.set_xlim( self.xyz_range['x'] )
		self.plots.set_ylim( self.xyz_range['y'] )
		if len(axs) >= 3:
			min_z, max_z = self.xyz_range['z']; self.xyz_range['z'] = [ min( min_z, min(axs[2]) ), max( max_z, max(axs[2]) ) ]
			self.plots.set_zlim( self.xyz_range['z'] )
		
	#---------------------------------------
	def end_plot(self, figure_name = None):
		if figure_name is None: plt.show()
		else: plt.savefig(figure_name)
		
		# plt.grid(True) # FIXME
		plt.close()
		
		self.plots = None
		self.xyz_range = { 'x':[float("inf"), float("-inf")], 'y':[float("inf"), float("-inf")], 'z':[float("inf"), float("-inf")] }
		
	#---------------------------------------
	def plot(self, axs, axs_labels = None, color = 'r', marker = '.', figure_name = None):
		self.do_plot( axs, axs_labels, color, marker )
		self.end_plot( figure_name )
		
	#---------------------------------------
	def plot_groups(self, groups, figure_name = None):
		colors = ['r', 'b', 'g','m', 'y', 'k']
		keys = groups.keys()
		
		if len(keys) > len(colors):
			print "Warning: the number of groups to plot is ", len(keys), " > ", len(colors),". Some groups may be colored similarly."
			
		for i, label in enumerate( keys ):
			cl = colors[i % len(colors)]
			self.do_plot( zip(* groups[label] ), color = cl )
		
		self.end_plot(figure_name)
		
	#---------------------------------------
	
	