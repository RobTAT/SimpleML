import numpy as np
import math
import random
from rafik_utilities import weighted_choice

class Bandit:
	def __init__(self, algos, method = "UCB", alpha = 1, epsilon = 0.2):
		self.method = method
		self.algos = algos
		self.K = len(algos)
		
		self.alpha = alpha
		self.epsilon = epsilon
		
		self.rewards = [ [] for a in algos ]
		self.WE = [1./self.K]*self.K
		
		self.counter = 0
		self.r_expected = 0.
		
		# Just some statistics
		self.nb_choices = [0.] * len(algos)
		self.current_nb_choices = [ [] for a in algos ]
		
		
	#==========================================
	def choose(self):
		if self.counter < self.alpha * self.K:
			id_a = self.counter % self.K
		else:
			if self.method == "UCB": id_a = self.choose_UCB()
			elif self.method == "greedy": id_a = self.choose_greedy()
			elif self.method == "boltzmann": id_a = self.choose_boltzmann()
			elif self.method == "pursuit": id_a = self.choose_pursuit()
			elif self.method == "EXP3": id_a = self.choose_EXP3()
			elif self.method == "reinforcement": id_a = self.choose_reinforcement()
			
		self.counter += 1
		self.nb_choices[id_a] += 1
		self.current_nb_choices = [ L + [1] if ia == id_a else L + [0] for ia,L in enumerate(self.current_nb_choices) ]
		
		return id_a

	#------------------------------------------
	def update(self, id_a, reward):
		if self.method == "UCB": self.update_UCB(id_a, reward)
		elif self.method == "greedy": self.update_greedy(id_a, reward)
		elif self.method == "boltzmann": self.update_boltzmann(id_a, reward)
		elif self.method == "pursuit": self.update_pursuit(id_a, reward)
		elif self.method == "EXP3": self.update_EXP3(id_a, reward)
		elif self.method == "reinforcement": self.update_reinforcement(id_a, reward)
	
	#==========================================
	def choose_UCB(self):
		means = [ np.mean(L) for L in self.rewards ]
		nbs = [ len(L) for L in self.rewards ]
		sco = [ m + math.sqrt( 2. * np.log( self.counter+1. ) / nbs[i] ) for i,m in enumerate(means) ]
		id_a = sco.index( max(sco) )
		return id_a
	
	def update_UCB(self, id_a, reward):
		self.rewards[id_a].append( reward )
		
	#==========================================
	def choose_greedy(self):
		if random.uniform(0., 1.) <= self.epsilon:
			id_a = random.randint( 0, self.K-1 )
		else:
			means = [ np.mean(L) for L in self.rewards ]
			id_a = means.index( max(means) )
			
		return id_a
		
	def update_greedy(self, id_a, reward):
		self.rewards[id_a].append( reward )
		
	#==========================================
	def choose_boltzmann(self):
		means = [ np.mean(L) for L in self.rewards ]
		sco = [ math.exp(m / self.epsilon) for m in means ]
		sco = [ s / sum(sco) for s in sco ]
		id_a = weighted_choice( range(self.K), sco )
		return id_a
		
	def update_boltzmann(self, id_a, reward):
		self.rewards[id_a].append( reward )
		
	#==========================================
	def choose_pursuit(self):
		means = [ np.mean(L) for L in self.rewards ]
		id_max = means.index( max(means) )
		self.WE = [ pi + self.epsilon*(1.-pi) if i == id_max else pi + self.epsilon*(0.-pi) for i,pi in enumerate( self.WE )  ]
		id_a = weighted_choice( range(self.K), self.WE )
		
		return id_a
		
	def update_pursuit(self, id_a, reward):
		self.rewards[id_a].append( reward )
		
	#==========================================
	def choose_EXP3(self):
		P = [ (1. - self.epsilon) * w/sum(self.WE) + self.epsilon * 1./self.K for w in self.WE ]
		id_a = weighted_choice( range(self.K), P )
		return id_a
		
	def update_EXP3(self, id_a, reward):
		P_id_a = (1. - self.epsilon) * self.WE[id_a]/sum(self.WE) + self.epsilon * 1./self.K
		rew = reward / P_id_a
		self.WE[id_a] = self.WE[id_a] * math.exp( self.epsilon * rew / self.K )
		
	#==========================================
	def choose_reinforcement(self):
		sco = [ math.exp(pi) for pi in self.WE ]
		sco = [ s / sum(sco) for s in sco ]
		id_a = weighted_choice( range(self.K), sco )
		return id_a
		
	def update_reinforcement(self, id_a, reward):
		eps2 = self.epsilon
		eps1 = 1. - eps2
		self.WE[id_a] = self.WE[id_a] + eps1 * ( reward - self.r_expected )
		self.r_expected = (1 - eps2) * self.r_expected + eps2 * reward
		
	#==========================================
	
	