import random
import numpy as np
import os
import sys
import datetime

from GNG import GNG
from IGNG import IGNG
import Util
import cosmo.Anomaly as como_anomaly
import cosmo.Ploting as como_ploting
import cosmo.Changes as como_changes
import cosmo.Mileage as como_mileage
import cosmo.HistogramExtraction as como_extract
from cosmo.Parameters import *
from sklearn.neighbors import NearestNeighbors

#-----------------------------------

if __name__ == "__main__":
	random.seed( 12345 )
	
	# como_extract.computeHistogramsAllBuses(); exit(0)
	(all_buses, periods_all_buses) = Util.pickleLoad(DATA_FILE_NAME+"_"+SIGNAL_CODE+".txt")
	dates_all_buses = [ [ Util.getDate(tm) for tm in times ] for times in periods_all_buses ]
	
	# como_ploting.Ploting.vizualize_buses(all_buses, dates_all_buses, m = '.'); exit(0)
	
	#-----------------------------------
	for id_bus in range( len(all_buses) ): # for each test bus
		dir_imgs = "centroid/"
		
		# h = IGNG( radius = PARAMS["R"] ); h.train( [Util.centroid( Util.flatList(all_buses) )] ) 
		# h = GNG(period = 100); h.train( [Util.centroid( Util.flatList(all_buses) )] ) 
		
		own_test = all_buses[id_bus]
		fleet_test = all_buses[:id_bus] + all_buses[id_bus+1 :]
		
		filename = DBFILES[id_bus]
		busname = filename.split("_")[0]
		dates = dates_all_buses[id_bus]
		
		Z1 = []; Z2 = []; S1 = []; S2 = []
		
		#--------------------------
		for i, his_test in enumerate( own_test ): # for each day
			sys.stdout.write( "\r%s" % "---------------------------- progress = " + str(i*100./len(own_test)) + " " + DBFILES[id_bus] + " " ); sys.stdout.flush()
			
			own_test_ = Util.shrink(i, own_test, TH1)
			fleet_test_ = [ Util.shrink(i, bus, TH1) for bus in fleet_test ]
			flat_fleet_test_ = Util.flatList( fleet_test_ )
			
			# ------
			center = Util.centroid( own_test_ ); score1 = Util.dist(his_test, center); pvalue1 = 0.5
			# pvalue1, score1 = como_anomaly.normalityProba_V3( "centroid", own_test_, his_test, all_buses, id_bus, i )
			# pvalue1, score1 = como_anomaly.normalityProba_V3( "medoid", own_test_, his_test, all_buses, id_bus, i )
			# ------
			
			# pvalue1, score1 = como_anomaly.normalityProba_V1( "online", flat_fleet_test_, his_test, all_buses, id_bus, i, h )
			# pvalue1, score1 = como_anomaly.normalityProba_V1( "IGNG", flat_fleet_test_, his_test, all_buses, id_bus, i )
			# pvalue1, score1 = como_anomaly.normalityProba_V1( "GNG", flat_fleet_test_, his_test, all_buses, id_bus, i )
			# pvalue1, score1 = como_anomaly.normalityProba_V1( "KNN", flat_fleet_test_, his_test, all_buses, id_bus, i )
			# pvalue2, score2 = como_anomaly.normalityProba_V2( "RNN", own_test_, his_test, all_buses, id_bus, i )
			pvalue2, score2 = 0.5, 0.5
			
			# h.train( [ bus[0] for bus in fleet_test_ ] )# ; print "nb_nodes", h.nb_nodes
			
			
			Z1.append( pvalue1 ); Z2.append( pvalue2 )
			S1.append( score1 ); S2.append( score2 )
		
		#--------------------------
		connexion_sig, cursor_sig = Util.connectDB(DB_PATH+filename)
		connexion_sig, cursor_sig_mil = Util.connectDB(DB_PATH+busname+"_16644.db")
		connexion_vsr, cursor_vsr = Util.connectDB(DB_PATH+"_vsr.db")
		if busname == '369': busname = '396'
		
		#--------------------------
		image = como_ploting.Ploting( busname, dates )
		
		Z1_means, Dev1 = como_anomaly.getZvalues( Z1 )
		Z2_means, Dev2 = como_anomaly.getZvalues( Z2 )
		image.plotScores(S1, S2); image.plotPValues(Z1, Z2, Z1_means, Z2_means); image.plotDeviations(Dev1, Dev2)
		
		#----------
		repair_dates_str = cursor_vsr.execute( "SELECT DISTINCT Visits.Date FROM Visits,Operations WHERE Visits.VisitID = Operations.Visit AND Visits.Bus LIKE '%"+busname+"%'" )
		repair_dates = [ Util.str2date(date_str, format = "%Y-%m-%d") for date_str in repair_dates_str ]
		
		interesting_repairs = []
		
		#          ['r',    'm',   'b',    'g',      'k',     'c']
		for opr in ["213", "57", "86", "1088",   "FdddD",     "FdddD"] :
			interesting_repairs_str = cursor_vsr.execute( "SELECT DISTINCT Visits.Date, Operations.* FROM Visits,Operations WHERE Visits.VisitID = Operations.Visit AND Operations.Code LIKE '"+opr+"' AND Visits.Bus LIKE '%"+busname+"%'" )
			interesting_repairs.append( [ Util.str2date(date_str, format = "%Y-%m-%d") for date_str in interesting_repairs_str ] )
		
		image.plotRepairs( repair_dates, interesting_repairs )
		
		#----------
		entropies = [ 1. - Util.entropy(hi) for hi in own_test ]
		changes = como_changes.getChanges(own_test, 1)
		repair_dates_significant, repair_significance = como_changes.getSignificantRepairs(repair_dates, dates = dates, changes = changes)
		image.plotChanges(entropies, changes, repair_dates_significant, repair_significance)
		
		#----------
		# dates_mil_vsr, values_mil_vsr = como_mileage.getMileage(cursor_vsr, busname, fromVsr = True)
		# dates_mil_sig, values_mil_sig = como_mileage.getMileage(cursor_sig_mil, busname, fromVsr = False)
		# image.plotMileage( dates_mil_vsr, values_mil_vsr, dates_mil_sig, values_mil_sig )
		
		#----------
		image.savePlot(dir_imgs)
		connexion_sig.close(); connexion_vsr.close()
