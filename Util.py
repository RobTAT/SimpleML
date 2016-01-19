import cPickle
import json
import random
import numpy as np
import sqlite3
import datetime

class Util:
	#---------------------------------------
	def shrink(i, L, nb): # get the list of the nb elements preceding i in L
		j = len(L) if i > len(L) else i
		return L[j-nb:j] if len(L[j-nb:j]) >= nb else L[j+1:j+nb]

	#---------------------------------------
	@staticmethod
	def getDate(timestamp_value):
		INITIAL_DATE = datetime.datetime(year=2011, month=6, day=16, hour=5, minute=23, second=0)
		return INITIAL_DATE + datetime.timedelta(milliseconds=int(timestamp_value))

	@staticmethod
	def str2date(strDate, format = "%Y, %m, %d"):
		return datetime.datetime.strptime(strDate, format)
	
	@staticmethod
	def date2str(theDate, format = "%Y, %m, %d"):
		return theDate.strftime(format)
	
	@staticmethod
	def strs2dates(strs, format = "%Y, %m, %d"):
		return [ str2date(date_str, format) for date_str in strDate ]
	
	@staticmethod
	def dates2strs(dates, format = "%Y, %m, %d"):
		return [ date2str(dt, format) for dt in dates ]

	#---------------------------------------
	@staticmethod
	def connectDB(dbfile):
		conn = sqlite3.connect(dbfile)
		conn.row_factory = lambda cursor, row: row[0]
		cur = conn.cursor()
		return conn, cur
	#---------------------------------------
	@staticmethod
	def signalToHistogram( sig, bins=60, ran=(0,12) ):
		his, bin_edges = np.histogram( sig, bins=bins, range=ran )
		s = sum(his)*1.0
		if s > 0.0: his = list([1.0*h/s for h in his])
		else: his = list([h for h in his])
		return his

	#=====================================================================================================
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

	#=====================================================================================================
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

