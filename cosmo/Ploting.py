import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange, MonthLocator
import datetime
import Util

class Ploting(object):
	def __init__( self, busname, dates, ylabels = ["Scores", "P-Values", "Deviations", "Changes", "Mileage", "Etc"], plot_h = 25, plot_w = 15 ):
		self.dates = dates
		self.busname = busname
		self.fig, self.axs = plt.subplots( len(ylabels), 1, sharex=True )
		for i,yl in enumerate(ylabels): self.axs[i].set_ylabel(yl)
		
		min_time = datetime.datetime(year=2011, month=5, day=1)
		max_time = datetime.datetime(year=2015, month=9, day=1)
		self.axs[0].set_xlim([min_time, max_time]) 
		self.fig.autofmt_xdate()
		self.axs[0].xaxis.set_major_locator(MonthLocator(interval=1))
		self.fig.set_size_inches(plot_h, plot_w)
		
	# -------------------------------------------------------------------
	def savePlot(self):
		plt.savefig('_plt_' + self.busname + '.png')
		plt.close()
		
	# -------------------------------------------------------------------
	def plotScores(self, S1, S2):
		self.axs[0].plot_date(self.dates, Util.normalize(S1), 'b.')
		self.axs[0].plot_date(self.dates, Util.normalize(S2), 'r.')
	
	# -------------------------------------------------------------------
	def plotPValues(self, Z1, Z2, Z1_means, Z2_means):
		self.axs[1].plot_date(self.dates, Z1, 'b.')
		self.axs[1].plot_date(self.dates, Z2, 'r.')
		self.axs[1].plot_date(self.dates, Z1_means, 'b-')
		self.axs[1].plot_date(self.dates, Z2_means, 'r-')
	
	# -------------------------------------------------------------------
	def plotDeviations(self, Dev1, Dev2):
		self.axs[2].plot_date(self.dates, Dev1, 'b.')
		self.axs[2].plot_date(self.dates, Dev2, 'r.')
	
	# -------------------------------------------------------------------
	def plotChanges(self, entropies, changes, repair_dates_significant, repair_significance):
		self.axs[3].plot_date(self.dates, changes, 'm-')
		self.axs[3].plot_date(repair_dates_significant, repair_significance, 'ms', linewidth=3)
		for i,ax in enumerate(self.axs):
			if i != 5:
				for rd in repair_dates_significant: ax.axvline(x=rd, ymax=1, color = 'g', linewidth=1.0, ls='-')
			
		entropies_mean = Util.weightedMovingAverage(entropies, 7, power = 5)
		zeros = [1.] * ( len( entropies ) - len( entropies_mean ) ); entropies_mean = zeros + entropies_mean
		self.axs[3].plot_date(self.dates, entropies, 'ko', markersize=1.5)
		self.axs[3].plot_date(self.dates, entropies_mean, 'k-')

	# -------------------------------------------------------------------
	def plotRepairs( self, repair_dates, interesting_repairs = [] ):
		for ax in self.axs:
			for rd in repair_dates: ax.axvline(x=rd, ymax=1, color = 'y', linewidth=1.0)
	
		'''dico_repairs = getRepairsByCategory( self.busname )
		for rd in dico_repairs["CRwT"]: self.axs[5].axvline(x=rd, ymax=1, color = 'r', linewidth=2, ls='-') # (1) CRwT = compressor replacement with towing
		for rd in dico_repairs["CRiT"]: self.axs[5].axvline(x=rd, ymax=1, color = 'r', linewidth=2, ls='--') # (2) CRiT = compressor replacement without towing
		for rd in dico_repairs["PHRD"]: self.axs[5].axvline(x=rd, ymax=1, color = 'b', linewidth=2, ls='-') # (3) PHRD = congested air pipes/hoses, malfunctioning regulator/dryer
		for rd in dico_repairs["GBAB"]: self.axs[5].axvline(x=rd, ymax=1, color = 'b', linewidth=2, ls='--') # (4) GBAB = gearbox and air brakes
		for rd in dico_repairs["ALKS"]: self.axs[5].axvline(x=rd, ymax=1, color = 'b', linewidth=2, ls=':') # (5) ALKS = air leaks'''
		
		colors = [('r','-'), ('m','--'), ('b','.'), ('g','--'), ('k',':'), ('c','.')]
		for i,repairs in enumerate(interesting_repairs):
			cl, ls = colors[i%len(colors)]
			for rd in repairs: self.axs[5].axvline(x=rd, ymax=1, color = cl, linewidth=2, ls = ls)
		
	# -------------------------------------------------------------------
	def plotMileage( self, dates_mil_vsr, values_mil_vsr, dates_mil_sig, values_mil_sig ):
		max_mileage = values_mil_sig[-1]
		values_mil_vsr = [ v if v < max_mileage else max_mileage for v in values_mil_vsr ]
		
		self.axs[4].plot_date(dates_mil_vsr, values_mil_vsr, 'rs', markersize=4)
		self.axs[4].plot_date(dates_mil_sig, values_mil_sig, 'mo-', markersize=2)
		
		for i_ml, ml in enumerate(values_mil_vsr):
			d_vsr = dates_mil_vsr[i_ml]
			id_nearest_date = min(range(len(dates_mil_sig)), key=lambda ii:abs(d_vsr-dates_mil_sig[ii]))
			id_nearest_value = min(range(len(values_mil_sig)), key=lambda ii:abs(ml-values_mil_sig[ii]))
			
			diff_mi = abs(ml - values_mil_sig[id_nearest_date]) / max(ml, values_mil_sig[id_nearest_date])
			if diff_mi > 0.05 :
				self.axs[4].plot_date( [ dates_mil_sig[id_nearest_value],  dates_mil_vsr[i_ml] ], [ ml, ml ], 'r-', markersize=0.05)
	
	# -------------------------------------------------------------------
	
