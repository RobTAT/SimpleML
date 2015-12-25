class DatasetLoader:
	#--------------------------------------------------------------------------------------------------------
	def __init__(self, datasetname, p = 0.0, noise_type = 1):
		self.data_train = np.genfromtxt(datasetname+'.tra', delimiter=',')[:,:-1]
		self.labels_train = np.genfromtxt(datasetname+'.tra', delimiter=',', usecols=-1, dtype=str)
		
		self.data_test = np.genfromtxt(datasetname+'.tes', delimiter=',')[:,:-1]
		self.labels_test = np.genfromtxt(datasetname+'.tes', delimiter=',', usecols=-1, dtype=str)
		
		self.data_train = [list(x) for x in self.data_train]
		self.data_test = [list(x) for x in self.data_test]
		
		self.unic_labels = set(self.labels_train)
		self.get_labels = dict((tuple(x), y) for x,y in zip(self.data_train, self.labels_train))
		self.get_indexes = dict((tuple(x), y) for x,y in zip(self.data_train, range(len(self.labels_train))))
		
		self.labels_train = self.labels_train.tolist()
		self.labels_train_noisy = copy.deepcopy(self.labels_train)
		self.nbr_noisy_labels = 0
		self.createNoisyLabels( p, noise_type )
		self.get_noisy_labels = dict((tuple(x), y) for x,y in zip(self.data_train, self.labels_train_noisy))
		
		# print "noise", self.nbr_noisy_labels, "total", len(self.labels_train_noisy), "==>", 1. * self.nbr_noisy_labels / len(self.labels_train_noisy)
	
	#--------------------------------------------------------------------------------------------------------
	def getTrueLabel(self, x):
		return self.get_labels[tuple(x)]
		
	def getIndex(self, x):
		return self.get_indexes[tuple(x)]
	
	def getTrueNoisyLabel(self, x):
		return self.get_noisy_labels[tuple(x)]
		
	#--------------------------------------------------------------------------------------------------------
	def queryNoisyLabel1(self, x, p = 0.3): # random label error
		random_label = self.getTrueLabel(x)
		
		if random.random() <= p:
			random_label = random.sample([y for y in self.unic_labels if y != self.getTrueLabel(x)], 1)[0]
			
			self.labels_train_noisy[ self.data_train.index(x) ] = random_label
			self.nbr_noisy_labels += 1
			print "Perturbed = ", self.nbr_noisy_labels
			
		return random_label

	def queryNoisyLabel2(self, x, p = 0.3): # most probable label error
		random_label = self.getTrueLabel(x)
		if random.random() <= p:
			GAMMA, C, K = 0.1, 100, 10
			hh = svm.SVC(gamma=GAMMA, C=C, probability=True).fit(self.data_test, self.labels_test)
			for id_p in range( len( list(self.unic_labels) ) ):
				random_label = raf.getLabelOf(id_p+1, x, hh)
				if random_label != self.getTrueLabel(x): break

			self.labels_train_noisy[ self.data_train.index(x) ] = random_label
			self.nbr_noisy_labels += 1
			print "Perturbed = ", self.nbr_noisy_labels
			
		return random_label
	
	#--------------------------------------------------------------------------------------------------------
	def createNoisyLabels(self, p = 0.3, noise_type = 1):
		if noise_type == 1: self.createNoisyLabels_NCAR( p )
		if noise_type == 2: self.createNoisyLabels_NAR( p )
		
		if noise_type == 311: self.createNoisyLabels_NNAR1( p , uncertainty = 1 )
		if noise_type == 312: self.createNoisyLabels_NNAR1( p , uncertainty = 2 )
		if noise_type == 313: self.createNoisyLabels_NNAR1( p , uncertainty = 3 )
		
		if noise_type == 321: self.createNoisyLabels_NNAR2( p , uncertainty = 1 )
		if noise_type == 322: self.createNoisyLabels_NNAR2( p , uncertainty = 2 )
		if noise_type == 323: self.createNoisyLabels_NNAR2( p , uncertainty = 3 )
		
		if noise_type == 331: self.createNoisyLabels_NNAR3( p , uncertainty = 1 )
		if noise_type == 332: self.createNoisyLabels_NNAR3( p , uncertainty = 2 )
		if noise_type == 333: self.createNoisyLabels_NNAR3( p , uncertainty = 3 )
		
		if noise_type == 341: self.createNoisyLabels_NNAR4( p , uncertainty = 1 )
		if noise_type == 342: self.createNoisyLabels_NNAR4( p , uncertainty = 2 )
		if noise_type == 343: self.createNoisyLabels_NNAR4( p , uncertainty = 3 )
	
	def createNoisyLabels_NCAR(self, p=0.3): # Noisy Completely At Random (depends on nothing)
		index_shuf = range(len(self.labels_train_noisy))
		random.shuffle(index_shuf)
		
		for i in index_shuf[:int(p*len(self.labels_train_noisy))]:
			random_label = random.sample([y for y in self.unic_labels if y != self.labels_train_noisy[i]], 1)[0]
			self.labels_train_noisy[i] = random_label
			self.nbr_noisy_labels += 1

	def createNoisyLabels_NAR(self, p=0.3): # Noisy At Random (depends on Y)
		nbr_classes_to_perturb = 10#int(len(list(self.unic_labels)) / 2.)
		labels_to_perturb = random.sample(list(self.unic_labels), nbr_classes_to_perturb)
		
		index_shuf = range(len(self.labels_train_noisy))
		random.shuffle(index_shuf)
		
		for i in index_shuf:
			if self.labels_train_noisy[i] in labels_to_perturb:
				random_label = random.sample([y for y in labels_to_perturb if y != self.labels_train_noisy[i]], 1)[0]
				self.labels_train_noisy[i] = random_label
				self.nbr_noisy_labels += 1
				
			if self.nbr_noisy_labels >= int(p*len(self.labels_train_noisy)):
				break
	
	def createNoisyLabels_NNAR1(self, p=0.3, uncertainty = 1): # Noisy Not At Random (depends on X)
		GAMMA, C, K = 0.1, 100, 10
		hh = svm.SVC(gamma=GAMMA, C=C, probability=True).fit(self.data_test, self.labels_test)
		if uncertainty == 1: scores = [ raf.getPredictProba(1, x, hh) for x in self.data_train ]
		if uncertainty == 2: scores = [ raf.getPredictProba(1, x, hh) - raf.getPredictProba(2, x, hh) for x in self.data_train ]
		if uncertainty == 3: scores = [ 1.-raf.getPredictionEntropy(x, hh) for x in self.data_train ]
		ids = (np.array(scores)).argsort()
		
		for i in ids:
			random_label = random.sample([y for y in self.unic_labels if y != self.labels_train_noisy[i]], 1)[0]
			self.labels_train_noisy[i] = random_label
			self.nbr_noisy_labels += 1
				
			if self.nbr_noisy_labels >= int(p*len(self.labels_train_noisy)):
				break
	
	def createNoisyLabels_NNAR2(self, p=0.3, uncertainty = 1): # Noisy Not At Random (depends on Y and X)
		nbr_classes_to_perturb = 10#int(len(list(self.unic_labels)) / 2.)
		labels_to_perturb = random.sample(list(self.unic_labels), nbr_classes_to_perturb)
		
		GAMMA, C, K = 0.1, 100, 10
		hh = svm.SVC(gamma=GAMMA, C=C, probability=True).fit(self.data_test, self.labels_test)
		if uncertainty == 1: scores = [ raf.getPredictProba(1, x, hh) for x in self.data_train ]
		if uncertainty == 2: scores = [ raf.getPredictProba(1, x, hh) - raf.getPredictProba(2, x, hh) for x in self.data_train ]
		if uncertainty == 3: scores = [ 1.-raf.getPredictionEntropy(x, hh) for x in self.data_train ]
		ids = (np.array(scores)).argsort()
		
		for i in ids:
			if self.labels_train_noisy[i] in labels_to_perturb:
				random_label = random.sample([y for y in labels_to_perturb if y != self.labels_train_noisy[i]], 1)[0]
				self.labels_train_noisy[i] = random_label
				self.nbr_noisy_labels += 1
				
			if self.nbr_noisy_labels >= int(p*len(self.labels_train_noisy)):
				break
	
	def createNoisyLabels_NNAR3(self, p=0.3, uncertainty = 1): # Noisy Not At Random (depends on X and uncertainty)
		GAMMA, C, K = 0.1, 100, 10
		hh = svm.SVC(gamma=GAMMA, C=C, probability=True).fit(self.data_test, self.labels_test)
		
		if uncertainty == 1: scores = [ raf.getPredictProba(1, x, hh) for x in self.data_train ]
		if uncertainty == 2: scores = [ raf.getPredictProba(1, x, hh) - raf.getPredictProba(2, x, hh) for x in self.data_train ]
		if uncertainty == 3: scores = [ 1.-raf.getPredictionEntropy(x, hh) for x in self.data_train ]
		ids = (np.array(scores)).argsort()
		
		for i in ids:
			for id_p in range( len( list(self.unic_labels) ) ):
				random_label = raf.getLabelOf(id_p+1, x, hh)
				if random_label != self.getTrueLabel(x):
					break
			
			self.labels_train_noisy[i] = random_label
			self.nbr_noisy_labels += 1
			
			if self.nbr_noisy_labels >= int(p*len(self.labels_train_noisy)):
				break
				
	def createNoisyLabels_NNAR4(self, p=0.3, uncertainty = 1): # Noisy Not At Random (depends on Y and X and uncertainty)
		nbr_classes_to_perturb = 10#int(len(list(self.unic_labels)) / 2.)
		labels_to_perturb = random.sample(list(self.unic_labels), nbr_classes_to_perturb)
		
		GAMMA, C, K = 0.1, 100, 10
		hh = svm.SVC(gamma=GAMMA, C=C, probability=True).fit(self.data_test, self.labels_test)
		
		if uncertainty == 1: scores = [ raf.getPredictProba(1, x, hh) for x in self.data_train ]
		if uncertainty == 2: scores = [ raf.getPredictProba(1, x, hh) - raf.getPredictProba(2, x, hh) for x in self.data_train ]
		if uncertainty == 3: scores = [ 1.-raf.getPredictionEntropy(x, hh) for x in self.data_train ]
		ids = (np.array(scores)).argsort()
		
		for i in ids:
			if self.labels_train_noisy[i] in labels_to_perturb:
				for id_p in range( len( list(self.unic_labels) ) ):
					random_label = raf.getLabelOf(id_p+1, x, hh)
					if random_label != self.getTrueLabel(x):
						break
				
				self.labels_train_noisy[i] = random_label
				self.nbr_noisy_labels += 1
				
			if self.nbr_noisy_labels >= int(p*len(self.labels_train_noisy)):
				break
				
