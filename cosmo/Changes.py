import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import Util
from Parameters import *

import numpy as np

################################################################################################
def getChanges(hists, nb = 1):
	
	changes = []
	
	for i in range( len(hists) ):
		hists_before_repair = hists[i-nb:i] if len(hists[i-nb:i]) > 0 else hists[i+1:i+nb+1] # FIXME
		hists_after_repair  = hists[i+1:i+nb+1] if len(hists[i+1:i+nb+1]) > 0 else hists[i-nb:i] # FIXME
		
		# median_hists_before_repair = [ np.median(elem) for elem in zip(*hists_before_repair) ]
		# median_hists_after_repair = [ np.median(elem) for elem in zip(*hists_after_repair) ]
		# changes += [ Util.dist(median_hists_before_repair, median_hists_after_repair) ]
		
		changes += [ 1. - np.corrcoef(hists_after_repair, hists_before_repair)[0, 1] ]
	
	changes = Util.normalize(changes)
	
	return changes

################################################################################################
def getSignificantRepairs(repair_dates, dates, changes):
	___repair_dates___ = []
	___repair_changes___ = []
	for i_change, change in enumerate(changes):
		if change > SIGNIFICANCE:
			current_date = dates[i_change]
			nearestRepairs = sorted(repair_dates, key=lambda dr:abs(current_date-dr))[:2]
			
			for nearestRepair in nearestRepairs:
				if abs( (current_date-nearestRepair).days ) < VELOCITY: 
					___repair_dates___ += [nearestRepair]
					___repair_changes___ += [change]
	
	return ___repair_dates___, ___repair_changes___

################################################################################################
