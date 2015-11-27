'''
| This is to format and read your data 
'''

import scipy.io
import os

class Data:
	def __init__(self, source_file = "", target_name = ""):
		if source_file == "":
			self.Y = [ 1, 2, 3, 4, 5 ]
			self.YY = self.Y[:]
			self.X = [ [1, 8], [9, 9], [0, 4], [0, 6], [3, 9] ]
			self.X_transpose = [ list(v) for v in zip(*self.X) ] #FIXME define external function to transpose and call it
			
			self.nb_data = len(self.X)
			self.nb_features = len(self.X[0])
			
			self.features_name = ["feature 1", "feature 2"]
			self.target_name = "target"
			
		else:
			file_name, file_extension = os.path.splitext( source_file )
			
			if file_extension == ".mat":
				self.readFromMatlab( source_file, target_name )
			else:
				print "TODO"
	
	#---------------------------------------
	def readFromCSV(self, source_file, target_name = ""):
		print "TODO"
		
	#---------------------------------------
	# Init the class attributes from a .mat file
	def readFromMatlab(self, source_file, target_name = ""):
		mat = scipy.io.loadmat(source_file)
		
		ignored_keys = ['__header__', '__globals__', '__version__', target_name]
		keys = mat.keys()
		self.features_name = [ k for k in keys if k not in ignored_keys ]
		self.target_name = target_name
		
		self.Y = [] if target_name == "" else [ v[0] for v in mat[self.target_name] ]
		self.YY = self.Y[:]
		self.X_transpose = [ [ v[0] for v in mat[fname] ] for fname in self.features_name ]
		self.X = [ list(v) for v in zip(*self.X_transpose) ]
		
		self.nb_data = len(self.X)
		self.nb_features = len(self.X[0])
		
	#---------------------------------------
	def discretize_Y(self, n_classes = 2):
		# FIXME not sure if correct
		min_Y = min(self.Y); max_Y = max(self.Y)
		lim_inf = [ int ( min_Y + (max_Y-min_Y) *(i*1.)/n_classes ) for i in range(n_classes) ]
		lim_sup = [ int ( min_Y + (max_Y-min_Y) *(i+1.)/n_classes ) for i in range(n_classes) ]
		lims = [ (lim_inf[i],lim_sup[i]) for i in range(n_classes) ]

		temp_list = []
		for y in self.Y:
			for i, (inf, sup) in enumerate(lims):
				if inf <= y  and y <= sup+1:
					temp_list.append( i )
					break
			
		self.Y = temp_list
		
	def restore_Y(self, n_classes = 2):
		self.Y = self.YY[:]
		
	#---------------------------------------
	# Scale only one data point (a list L) to make its values in [0, 1]
	def scale(self, L):
		maxL = max(L)*1.
		minL = min(L)*1.
		return [0.5]*len(L) if minL == maxL else [ (v - minL) / (maxL -minL) for v in L ] 
