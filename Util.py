import cPickle
import json

class Util:
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

