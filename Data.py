import scipy.io
import os
import numpy as np
from DatasetLoader import *

from sklearn.datasets import load_digits

class Data:
	def __init__(self, source_file = "", target_name = ""):
		if source_file == "":
			self.loadFromSklearn()
			
		else:
			file_name, file_extension = os.path.splitext( source_file )
			
			if file_extension == ".mat":
				self.readFromMatlab( source_file, target_name )
			else:
				print "TODO"
	
	#---------------------------------------
	def loadFromSklearn(self): # TODO add other datasets etc.
		digits = load_digits()
		
		self.Y = [y for y in digits["target"]]
		self.YY = self.Y[:]
		self.X = [list(x) for x in digits["data"]]
		self.X_transpose = [ list(v) for v in zip(*self.X) ]
		
		self.nb_data = len(self.X)
		self.nb_features = len(self.X[0])
		
		self.features_name = [ "feature "+str(i) for i in range(self.nb_features) ]
		self.target_name = "target"
		
	#---------------------------------------
	def loadBusesData(self, source_file):
		mat = scipy.io.loadmat(source_file)
		
		data_0 = mat.values()[0]
		
		data_0 = [ [ v if not np.isnan(v) else 0. for v in x ] for x in data_0 ]
		data_0[2] = [ v if v < 200 else 0. for v in data_0[2] ]
		data_0[4] = [ v if v < 200 else 0. for v in data_0[4] ]
		data_0[5] = [ v if v < 3000 else 0. for v in data_0[5] ]
		data_0[6] = [ v if v < 500 else 0. for v in data_0[6] ]
		data_0[10] = [ v if v < 20 else 0. for v in data_0[10] ]
		data_0[11] = [ v if v < 500000 else 0. for v in data_0[11] ]
		data_0[12] = [ v if v < 200 else 0. for v in data_0[12] ]
		
		self.features_name = ["Timestamp","AcceleratorPedalPos","AmbientAirTemperature","BrakePedalPos","EngineCoolantTemperature","EngineSpeed","Fuel Rate", "RelSpdFrontLeft","RelSpdFrontRight","Selected Gear","SteeringWheelAngle","Total Vehicle Distance","VehicleSpeed"]
		self.target_name = ""
		
		self.X = [ list(v) for v in data_0 ]
		self.X_transpose = [ list(v) for v in zip(*self.X) ]
		
		self.Y = self.X_transpose[0]
		self.YY = self.Y[:]
		
		self.nb_data = len(self.X)
		self.nb_features = len(self.X_transpose)

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
	def rescale(self):
		X_transpose = []
		for fe in self.X_transpose:
			min_fe = min(fe)*1.
			max_fe = max(fe)*1.
			X_transpose.append( [ (v - min_fe) / (max_fe - min_fe) for v in fe ] )
		
		self.X_transpose = X_transpose
		self.X = [ list(v) for v in zip(*self.X_transpose) ]
		
	#---------------------------------------
	def standardize(self):
		X_transpose = []
		for fe in self.X_transpose:
			mean_fe = np.mean(fe)*1.
			std_fe = np.std(fe)*1.
			X_transpose.append( [ (v - mean_fe) / std_fe for v in fe ] )
		
		self.X_transpose = X_transpose
		self.X = [ list(v) for v in zip(*self.X_transpose) ]
		
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
