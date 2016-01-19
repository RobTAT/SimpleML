import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Util import str2date


# time interval (in ms) where histograms are calculated (1mn = 60000 ms, 1day = 86400000 ms)
TIME_INTERVAL_MS = 86400000

# the required time of operation per day to compute a histogram in seconds
TIME_OPERATION = 1*60*60

# Number of bins and range of histograms
HIST_BINS = 60
HIST_BRANGE = (0,12)

################################################################################################
PARAMS = {'R': 0.142658578471, 'K': 100, 'NU': 0.3, 'GAMMA': 0.1, 'LOF': 3}

################################################################################################

# The path to the database
DB_PATH = "C:/Users/mohbou/Desktop/MachineLearning Toolbox/datasets/kungsbacka/"

# Name (actually subname) of the file containing the extracted histograms
DATA_FILE_NAME = DB_PATH+"kungsbacka"

# The path to the database
SIGNAL_CODE = "16929" # 16929 = WTAP, 16772 = Gearbox, 16644 = Mileage

# The list of files
DBFILES = [ f for f in os.listdir(DB_PATH) if os.path.isfile(os.path.join(DB_PATH,f)) and SIGNAL_CODE in f ]

# Time horizon
TH1 = 7 # period where hists are collected for computing the matrix
TH2 = 30 # period where p-values are computed (the moving average)

################################################################################################
SIGNIFICANCE = 0.05 # repair is significant if its change is higher than SIGNIFICANCE
# SIGNIFICANCE = 0.8 # repair is significant if its change is higher than SIGNIFICANCE
VELOCITY = 5 # distance (in days) between repair date and the changes close to that date (tolerance for the possibly shifted repair dates)

################################################################################################
################################################################################################
################################################################################################
################################################################################################
def getRepairsByCategory(busname):
	dico_repairs = {}
	
	# (1) CRwT = compressor replacement with towing
	# (2) CRiT = compressor replacement without towing
	# (3) PHRD = congested air pipes/hoses, malfunctioning regulator/dryer
	# (4) GBAB = gearbox and air brakes
	# (5) ALKS = air leaks
	
	if busname == '369' or busname == '396':
		dico_repairs['CRwT'] = [ str2date("2012, 7, 2"), str2date("2012, 7, 9") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2014, 3, 14"), str2date("2014, 3, 17") ]
		dico_repairs['GBAB'] = [ str2date("2011, 8, 1"), str2date("2011, 8, 19"), str2date("2012, 5, 1"), str2date("2012, 5, 11"), str2date("2014, 5, 05"), str2date("2014, 5, 6") ]
		dico_repairs['ALKS'] = [ str2date("2011, 8, 1"), str2date("2011, 8, 19") ]
		
	elif busname == '370':
		dico_repairs['CRwT'] = [ str2date("2014, 3, 13"), str2date("2014, 3, 17") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2012, 1, 31"), str2date("2012, 2, 3"), str2date("2012, 3, 4"), str2date("2012, 3, 19"), str2date("2013, 12, 12"), str2date("2013, 12, 19") ]
		dico_repairs['GBAB'] = [ str2date("2011, 8, 2"), str2date("2011, 8, 6"), str2date("2012, 3, 4"), str2date("2012, 3, 19"), str2date("2012, 4, 1"), str2date("2012, 4, 12"), str2date("2013, 2, 19"), str2date("2013, 3, 6") ]
		dico_repairs['ALKS'] = [ str2date("2013, 6, 19"), str2date("2013, 6, 28") ]
		
	elif busname == '371':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2013, 7, 22"), str2date("2013, 7, 25"), str2date("2013, 1, 18"), str2date("2013, 2, 27") ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2012, 2, 16"), str2date("2012, 2, 23"), str2date("2013, 9, 23"), str2date("2013, 9, 26"), str2date("2014, 3, 11"), str2date("2014, 3, 12") ]
		dico_repairs['ALKS'] = [ str2date("2013, 7, 12"), str2date("2013, 7, 17") ]
		
	elif busname == '372':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2012, 9, 1"), str2date("2012, 9, 3") ]
		dico_repairs['PHRD'] = [ str2date("2011, 9, 2"), str2date("2011, 9, 8"), str2date("2012, 8, 1"), str2date("2012, 8, 4") ]
		dico_repairs['GBAB'] = [ str2date("2012, 2, 5"), str2date("2012, 2, 9"), str2date("2013, 4, 9"), str2date("2013, 4, 17"), str2date("2014, 4, 4"), str2date("2014, 4, 7") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '373':
		dico_repairs['CRwT'] = [ str2date("2014, 1, 3"),  str2date("2014, 2, 12") ]
		dico_repairs['CRiT'] = [ str2date("2012, 6, 29"), str2date("2012, 7, 2") ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2012, 9, 10"), str2date("2012, 9, 20"), str2date("2013, 5, 8"), str2date("2013, 5, 9"), str2date("2013, 9, 10"), str2date("2013, 9, 11"), str2date("2014, 4, 8"), str2date("2013, 4, 9") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '374':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2012, 5, 15"), str2date("2012, 5, 20"), str2date("2012, 9, 18"), str2date("2012, 9, 20") ]
		dico_repairs['PHRD'] = [ str2date("2013, 1, 28"), str2date("2012, 2, 1") ]
		dico_repairs['GBAB'] = [ str2date("2011, 9, 19"), str2date("2011, 9, 21") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '375':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2013, 8, 1"), str2date("2013, 8, 7") ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2011, 10, 14"), str2date("2011, 10, 17"), str2date("2012, 7, 17"), str2date("2012, 8, 10"), str2date("2013, 1, 17"),  str2date("2013, 2, 1"), str2date("2013, 8, 1"), str2date("2013, 9, 17") ]
		dico_repairs['ALKS'] = [ str2date("2012, 2, 12"), str2date("2012, 4, 1"), str2date("2012, 12, 22"), str2date("2012, 12, 28") ]
		
	elif busname == '376':
		dico_repairs['CRwT'] = [ str2date("2012, 9, 17"), str2date("2012, 9, 20") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2012, 10, 24"), str2date("2012, 10, 26"), str2date("2012, 11, 30"), str2date("2012, 12, 1"), str2date("2013, 2, 12"), str2date("2013, 2, 14") ]
		dico_repairs['GBAB'] = [ str2date("2011, 11, 15"), str2date("2011, 11, 17"), str2date("2011, 12, 1"),  str2date("2011, 12, 12"), str2date("2014, 1, 17"), str2date("2014, 3, 4"), str2date("2014, 7, 1"), str2date("2014, 7, 3") ]
		dico_repairs['ALKS'] = [ str2date("2012, 8, 22"), str2date("2012, 8, 24"), str2date("2013, 10, 8"), str2date("2013, 10, 24") ]
		
	elif busname == '377':
		dico_repairs['CRwT'] = [ str2date("2012, 6, 25"), str2date("2012, 6, 27") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2013, 3, 22"), str2date("2013, 3, 27") ]
		dico_repairs['GBAB'] = [ str2date("2012, 1, 27"), str2date("2012, 1, 31"), str2date("2012, 4, 12"), str2date("2012, 4, 23"), str2date("2012, 10, 3"), str2date("2012, 10, 12"), str2date("2012, 11, 15"), str2date("2012, 11, 21"), str2date("2013, 7, 8"), str2date("2013, 7, 18"), str2date("2013, 9, 2"), str2date("2013, 9, 9") ]
		dico_repairs['ALKS'] = [ str2date("2012, 6, 27"), str2date("2012, 6, 28") ]
		
	elif busname == '378':
		dico_repairs['CRwT'] = [ str2date("2013, 7, 1"), str2date("2013, 7, 5") ]
		dico_repairs['CRiT'] = [ str2date("2013, 2, 28"), str2date("2013, 2, 28") ]
		dico_repairs['PHRD'] = [ str2date("2013, 12, 20"), str2date("2014, 1, 1") ]
		dico_repairs['GBAB'] = [ str2date("2011, 9, 26"), str2date("2011, 9, 29"), str2date("2011, 10, 19"), str2date("2011, 11, 15"),str2date("2013, 7, 11"), str2date("2013, 7, 15"), str2date("2012, 10, 8"), str2date("2012, 10, 22") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '379':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2012, 4, 23"), str2date("2012, 4, 25") ]
		dico_repairs['PHRD'] = [ str2date("2012, 10, 1"), str2date("2012, 10, 2") ]
		dico_repairs['GBAB'] = [ str2date("2011, 10, 31"), str2date("2011, 11, 9"), str2date("2013, 12, 9"), str2date("2013, 12, 11") ]
		dico_repairs['ALKS'] = [ str2date("2012, 5, 14"), str2date("2012, 6, 17") ]
		
	elif busname == '380':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2012, 10, 19"), str2date("2012, 10, 22") ]
		dico_repairs['PHRD'] = [ str2date("2011, 10, 24"), str2date("2011, 11, 8"), str2date("2012, 6, 27"), str2date("2012, 7, 2"), str2date("2013, 6, 20"), str2date("2013, 6, 24"), str2date("2014, 3, 26"), str2date("2014, 4, 20") ]
		dico_repairs['GBAB'] = [ str2date("2011, 10, 24"), str2date("2011, 11, 8"), str2date("2012, 1, 31"), str2date("2012, 2, 2"), str2date("2012, 4, 16"), str2date("2012, 4, 18"), str2date("2012, 4, 20"), str2date("2012, 4, 24"), str2date("2012, 6, 27"), str2date("2012, 7, 2"), str2date("2013, 12, 16"), str2date("2013, 12, 17"), str2date("2014, 3, 26"), str2date("2014, 4, 20") ]
		dico_repairs['ALKS'] = [ str2date("2011, 11, 23"), str2date("2011, 11, 24") ]
		
	elif busname == '381':
		dico_repairs['CRwT'] = [ str2date("2012, 2, 15"), str2date("2012, 2, 16") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2012, 9, 27"), str2date("2012, 10, 2"), str2date("2013, 1, 22"), str2date("2013, 2, 13"), str2date("2013, 7, 3"), str2date("2013, 7, 4"), str2date("2013, 8, 20"), str2date("2013, 8, 24"), str2date("2014, 5, 3"), str2date("2014, 5, 5") ]
		dico_repairs['ALKS'] = [ str2date("2013, 7, 3"), str2date("2013, 7, 4") ]
		
	elif busname == '382':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2012, 3, 23"), str2date("2012, 3, 26") ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2013, 7, 8"), str2date("2013, 7, 9"), str2date("2013, 6, 24"), str2date("2013, 7, 2") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '383':
		dico_repairs['CRwT'] = [ str2date("2012, 10, 24"), str2date("2012, 10, 25") ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2012, 3, 5"), str2date("2012, 3, 8"), str2date("2013, 5, 28"), str2date("2012, 6, 5") ]
		dico_repairs['GBAB'] = [ str2date("2012, 5, 28"), str2date("2012, 6, 5"), str2date("2013, 5, 28"), str2date("2012, 6, 5") ]
		dico_repairs['ALKS'] = [ str2date("2012, 2, 27"), str2date("2012, 2, 28"), str2date("2012, 6, 12"), str2date("2012, 6, 15") ]
		
	elif busname == '452':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2013, 5, 3"), str2date("2013, 6, 12") ]
		dico_repairs['PHRD'] = [ str2date("2014, 4, 17"), str2date("2014, 4, 22") ]
		dico_repairs['GBAB'] = [ str2date("2011, 11, 23"), str2date("2011, 11, 28"), str2date("2012, 11, 23"), str2date("2012, 12, 3") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '453':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2013, 10, 4"), str2date("2013, 10, 5") ]
		dico_repairs['PHRD'] = [ str2date("2014, 5, 27"), str2date("2014, 5, 28") ]
		dico_repairs['GBAB'] = [ str2date("2012, 12, 13"), str2date("2012, 12, 14") ]
		dico_repairs['ALKS'] = [  ]
		
	elif busname == '454':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [  ]
		dico_repairs['PHRD'] = [ str2date("2014, 1, 8"), str2date("2014, 1, 10") ]
		dico_repairs['GBAB'] = [ str2date("2013, 11, 26"), str2date("2013, 12, 5") ]
		dico_repairs['ALKS'] = [ str2date("2012, 11, 6"), str2date("2012, 11, 9"), str2date("2013, 5, 3"), str2date("2013, 5, 14"), str2date("2013, 7, 10"), str2date("2013, 7, 12") ]
		
	elif busname == '455':
		dico_repairs['CRwT'] = [  ]
		dico_repairs['CRiT'] = [ str2date("2014, 2, 16"), str2date("2014, 2, 25") ]
		dico_repairs['PHRD'] = [  ]
		dico_repairs['GBAB'] = [ str2date("2011, 9, 26"), str2date("2011, 9, 27"), str2date("2013, 5, 17"), str2date("2013, 6, 4") ]
		dico_repairs['ALKS'] = [ str2date("2012, 4, 12"), str2date("2012, 4, 20") ]
		
	return dico_repairs








