import cPickle
import json
import random
import os
import numpy as np
import sqlite3
import datetime
import math
from itertools import tee, izip

#---------------------------------------
def mkdir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
	
#---------------------------------------
def slidingWindow(iterable, size):
	iters = tee(iterable, size)
	for i in xrange(1, size):
		for each in iters[i:]:
			next(each, None)
	return izip(*iters)

def weightedMean(L, power = 2):
	weights = [ (w+1.)**power for w in range(len(L)) ]
	wL = 	  [ weights[i]*e for i,e in enumerate(L) ]
	return sum(wL) / sum(weights)

def movingAverage(Z, size): # return list( np.convolve(Z, np.ones(size)/size) )
	Z_means = []
	for each in slidingWindow(Z, size):
		Z_means += [ np.mean( list(each) ) ]
	return Z_means

def movingMedian(Z, size, rem_outliers = False):
	Z_means = []
	for each in slidingWindow(Z, size):
		if rem_outliers:
			each = [ v for v in each if abs(v - np.median(each)) < 1. * np.std(each) ]
			
		Z_means += [ np.median( list(each) ) ]
	return Z_means

def weightedMovingAverage(Z, size, power = 2):
	Z_means = []
	for each in slidingWindow(Z, size):
		Z_means += [ weightedMean( list(each), power ) ]
	return Z_means

#---------------------------------------
def flatList(L):
	return [item for sublist in L for item in sublist]

#---------------------------------------
def entropy(L):
	return -1.0 * sum([ p * math.log(p, len(L)) for p in L if p > 0 ])
	
#---------------------------------------
def dist(x1, x2):
	return math.sqrt( sum((np.array(x1) - np.array(x2))**2) )

#---------------------------------------
def normalize(L): # Normalize values of L to be in [0, 1]
	return [0.5]*len(L) if min(L) == max(L) else [ (v - min(L)) / (max(L) - min(L)) for v in L ] 

#---------------------------------------
def shrink(i, L, nb): # get the list of the nb elements preceding i in L
	j = len(L) if i > len(L) else i
	return L[j-nb:j] if len(L[j-nb:j]) >= nb else L[j+1:j+nb]

#---------------------------------------
def getDate(timestamp_value):
	INITIAL_DATE = datetime.datetime(year=2011, month=6, day=16, hour=5, minute=23, second=0)
	return INITIAL_DATE + datetime.timedelta(milliseconds=int(timestamp_value))

def str2date(strDate, format = "%Y, %m, %d"):
	return datetime.datetime.strptime(strDate, format)

def date2str(theDate, format = "%Y, %m, %d"):
	return theDate.strftime(format)

def strs2dates(strs, format = "%Y, %m, %d"):
	return [ str2date(date_str, format) for date_str in strDate ]

def dates2strs(dates, format = "%Y, %m, %d"):
	return [ date2str(dt, format) for dt in dates ]

#---------------------------------------
def connectDB(dbfile):
	conn = sqlite3.connect(dbfile)
	conn.row_factory = lambda cursor, row: row[0]
	cur = conn.cursor()
	return conn, cur

#=====================================================================================================
def shuffle_related_lists(L1, L2):
	indexes = range(len(L1))
	random.shuffle(indexes)
	L1 = [ L1[i] for i in indexes ]
	L2 = [ L2[i] for i in indexes ]
	return L1, L2

#---------------------------------------
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
def centroid(X):
	return [ np.mean(x) for x in zip(*X) ]
	
#=====================================================================================================
def jsonSave(filename, ob):
	with open(filename, 'w') as savefile: json.dump(ob, savefile)

def jsonLoad(filename):
	with open(filename) as savefile: ob = json.load(savefile)
	return ob
	
#---------------------------------------
def pickleSave(filename, ob):
	with open(filename, 'w') as savefile: cPickle.dump(ob, savefile)

def pickleLoad(filename):
	with open(filename) as savefile: ob = cPickle.load(savefile)
	return ob
	
#---------------------------------------

