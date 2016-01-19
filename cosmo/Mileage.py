import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import Util
from Parameters import *

import datetime

################################################################################################
def getSomeMileageFromSignal(cur):
	
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
			# histo = signalToHistogram(result_value)
			# hists_one_bus += [histo]
			histo = result_value[-1]
			hists_one_bus += [histo]
			periods_one_bus += [ result_timestamp[0] ]
			
			sys.stdout.write("\r%s --- Mileage" % str(Util.getDate( result_timestamp[0] ))); sys.stdout.flush()
			
		firstT = firstT + TIME_INTERVAL_MS
		
	return hists_one_bus, periods_one_bus

################################################################################################
def getMileage(cur, busname, fromVsr = False):	
	
	if fromVsr:
		cur.execute( "SELECT Visits.Date FROM Visits,Operations WHERE Visits.VisitID = Operations.Visit AND Visits.Bus LIKE '%"+busname+"%'" )
		result_dates_str = cur.fetchall()
		result_dates = [ Util.str2date(d_str, format = "%Y-%m-%d") for d_str in result_dates_str ]
		
		cur.execute( "SELECT ROUND(Visits.Mileage) FROM Visits,Operations WHERE Visits.VisitID = Operations.Visit AND Visits.Bus LIKE '%"+busname+"%'" )
		result_mileage = cur.fetchall()
		
	else:
		result_mileage, result_timestamp = getSomeMileageFromSignal( cur )
		result_dates = [ Util.getDate(ts) for ts in result_timestamp ]
		
	return result_dates, result_mileage

################################################################################################
