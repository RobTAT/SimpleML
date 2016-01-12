import cPickle
import json
import random

class Util:
	#---------------------------------------
	@staticmethod
	def shuffle_related_lists(L1, L2):
		indexes = range(len(L1))
		random.shuffle(indexes)
		L1 = [ L1[i] for i in indexes ]
		L2 = [ L2[i] for i in indexes ]
		return L1, L2

	#---------------------------------------
	@staticmethod
	def weighted_choice(items, weights):
		choices = zip(items, weights)
		total = sum(w for c,w in choices)
		r = random.uniform(0, total)
		upto = 0
		for c, w in choices:
			if upto + w >= r:
				return c
			upto += w
		assert False, "Shouldn't get here"

	#---------------------------------------
	@staticmethod
	def jsonSave(filename, ob):
		with open(filename, 'w') as savefile: json.dump(ob, savefile)
	
	@staticmethod
	def jsonLoad(filename):
		with open(filename) as savefile: ob = json.load(savefile)
		return ob
		
	#---------------------------------------
	@staticmethod
	def pickleSave(filename, ob):
		with open(filename, 'w') as savefile: cPickle.dump(ob, savefile)
	
	@staticmethod
	def pickleLoad(filename):
		with open(filename) as savefile: ob = cPickle.load(savefile)
		return ob
		
	#---------------------------------------

