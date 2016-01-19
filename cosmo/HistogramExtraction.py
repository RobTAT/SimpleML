import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import Util
from Parameters import *
import sqlite3
import datetime
import numpy as np

################################################################################################
def signalToHistogram(Sig): # ooo
	his, bin_edges = np.histogram(Sig, bins=HIST_BINS, range=HIST_BRANGE)
	s = sum(his)*1.0
	if s > 0.0: his = list([1.0*h/s for h in his])
	else: his = list([h for h in his])
	return his

################################################################################################
def computeHistogramsOneBus(dbfile):
	conn = sqlite3.connect(dbfile)
	conn.row_factory = lambda cursor, row: row[0]
	cur = conn.cursor() #; cur.arraysize = 10000
	
	cur.execute('SELECT timestamp FROM Data')
	firstT = cur.fetchone(); firstDate = Util.getDate(firstT)
	deltaMS = (firstDate - datetime.datetime(year=firstDate.year, month=firstDate.month, day=firstDate.day)).seconds * 1000
	firstT -= deltaMS; firstDate = Util.getDate(firstT) # to start from midnight
	
	hists_one_bus = []
	periods_one_bus = []
	
	while True:
		cur.execute('SELECT value FROM Data WHERE timestamp >= ? AND timestamp < ?', (firstT, firstT + TIME_INTERVAL_MS))
		result_value = cur.fetchall()
		cur.execute('SELECT timestamp FROM Data WHERE timestamp >= ? AND timestamp < ?', (firstT, firstT + TIME_INTERVAL_MS))
		result_timestamp = cur.fetchall()
		
		if result_value == []:
			cur.execute('SELECT value FROM Data WHERE timestamp >= ?', (firstT + TIME_INTERVAL_MS,))
			if cur.fetchone() == None: break
		
		if len(result_value) > TIME_OPERATION: # filter some useless signals
			histo = signalToHistogram(result_value)
			hists_one_bus += [histo]
			periods_one_bus += [ result_timestamp[0] ]
			
			sys.stdout.write("\r%s" % str(Util.getDate( result_timestamp[0] ))); sys.stdout.flush()
			
		firstT = firstT + TIME_INTERVAL_MS
		
	conn.close()
	
	return hists_one_bus, periods_one_bus

################################################################################################
def computeHistogramsAllBuses():
	DBFILES = [ f for f in os.listdir(DB_PATH) if os.path.isfile(os.path.join(DB_PATH,f)) and SIGNAL_CODE in f ]
	hists_all_buses = []; periods_all_buses = []
	
	for i_f, f in enumerate(DBFILES):
		print "\n========================> ", f, 100.*i_f / len(DBFILES)
		hists_one_bus, periods_one_bus = computeHistogramsOneBus(dbfile = DB_PATH+f)
		hists_all_buses += [hists_one_bus]
		periods_all_buses += [periods_one_bus]
		# if i_f >= 1: break
		
	Util.pickleSave("./"+DATA_FILE_NAME+"_"+SIGNAL_CODE+".txt", (hists_all_buses, periods_all_buses))
	print "Extracted data saved."

################################################################################################






