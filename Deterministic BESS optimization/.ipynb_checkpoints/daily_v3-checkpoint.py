# --- Arbitrage schedule of a standalone BESS with degradation and price forecast on the Spanish DAM market ---

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# -- Parameters --
Batt_Enom = 34					# [MWh] Battery nominal capacity
Batt_Pnom = 20					# [MW] Battery nominal power
Batt_ChEff = 0.9				# BESS charging efficiency
Batt_DchEff = 0.9			    # BESS discharging efficiency
Batt_Cost= 20000               	# [€] BESS cost
day = '2019-02-21'				# Day for schedule
Batt_Eff = 0.9					# Provisional Battery efficiency
Batt_SOC_init = 0				# Initial SOC
rej_c = [0] * 24            	# Purchasing bids rejected (1) by system operator
rej_d = [0] * 24            	# Selling bids rejected (1) by system operator

#%% -- Real price import function --
def daily_prices(day):
	# Loading dataset
	fields = ["Price", "Hour"]
	prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
	# Setting interval
	init = (day+' 00:00:00')
	init_index = np.where(prices_df["Hour"] == init)[0][0]                 
	# Generating list with prices and hours
	prices = []
	for i in range(init_index, init_index + 24):
		prices.append(prices_df.iloc[i, 0])
	return prices

#%% -- Price forecast generation function--
def daily_forecast(day, train):
	# Loading dataset
	fields = ["Price", "Hour"]
	prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
	# Setting interval
	day = (day +' 00:00:00')
	init_index = np.where(prices_df["Hour"] == day)[0][0] - 24 * (train - 1)
	end_index = np.where(prices_df["Hour"] == day)[0][0] - 24
	# Generating list with prices and hours as training data
	prices_train = []
	for i in range(init_index, end_index):
		prices_train.append(prices_df.iloc[i, 0])
	# Creating SARIMA model
	# model_order = (8, 0, 6)
	# model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order)
	model_order = (2, 0, 0)
	model_seasonal_order = (2, 1, 1, 24)
	model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order, seasonal_order=model_seasonal_order)
	# Fitting model
	model_fit = model.fit(disp=0)
	# Getting prediction
	prices_pred = model_fit.forecast(steps=24)

	return prices_pred

#%% -- Obtaining real and predicted prices
predicted_prices = daily_forecast(day, 50)
real_prices = daily_prices(day)
# predicted_prices = real_prices

#%% -- Arbitrage schedule function --
def arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower, 
              batt_efficiency, cost):
	    # Model initialization
	model = ConcreteModel()
	model.time = range(24)
	model.time2 = range(1, 24)
	model.time3 = range(25)
	model.SOC = Var(model.time3, bounds=(0, batt_capacity), initialize=0)            # Battery SOC
	model.not_charging = Var(model.time, domain=Binary)                # Charge verifier
	model.not_discharging = Var(model.time, domain=Binary)             # Discharge verifier
	model.ESS_C = Var(model.time, bounds=(0, batt_maxpower))           # Energy being charged
	model.ESS_D = Var(model.time, bounds=(0, batt_maxpower))           # Energy being discharged
	model.DOD = Var(bounds=(0,100))
	model.cycles = Var(bounds=(500, 10000))
	model.max_SOC = Var(bounds=(initial_SOC, 100))
	model.min_SOC = Var(bounds=(0, initial_SOC))
	
	# Degradation model
	DOD_index = [0., 10., 10., 20., 20., 30., 30., 40., 40., 
			  50., 50., 60., 60.,70., 70.,80., 80., 90., 90., 100]
	cycles_index = [10000., 10000., 15000., 15000., 7000., 7000., 3300., 3300.,
				  2050., 2050., 1475., 1475., 1150., 1150., 950., 950., 
				  760., 760., 675., 675., 580., 580., 500., 500]
	
	
	model.deg=  Piecewise(model.cycles, model.DOD, # range and domain variables
                      pw_pts=DOD_index ,
                      pw_constr_type='EQ',
                      f_rule=cycles_index,
                      pw_repn='INC')
	
	# Defining the optimization constraints
	def c1_rule(model, t1):  # Checks there is enough room when charging
		return (batt_maxpower * model.not_charging[t1]) >= model.ESS_C[t1]
	model.c1 = Constraint(model.time, rule=c1_rule)
	
	def c2_rule(model, t1):  # Checks there is enough power when discharging
		return (batt_maxpower * model.not_discharging[t1]) >= model.ESS_D[t1]
	model.c2 = Constraint(model.time, rule=c2_rule)
	
	def c3_rule(model, t1):  # Prevents orders of charge and discharge simultaneously
		return (model.not_charging[t1] + model.not_discharging[t1]) <= 1
	model.c3 = Constraint(model.time, rule=c3_rule)
	
	def c4_rule(model, t2):  # The SOC must be the result of (SOC + charge*eff - discharge/eff) on the previous hour
		return model.SOC[t2] == (model.SOC[t2 - 1] + (model.ESS_C[t2 - 1] *
		                                              batt_efficiency - model.ESS_D[t2 - 1] / batt_efficiency))
	model.c4 = Constraint(model.time2, rule=c4_rule)
	
	def c5_rule(model):  # SOC at hour 0 must be the initial SOC
		return model.SOC[0] == initial_SOC
	model.c5 = Constraint(rule=c5_rule)
	   
	def c6_rule(model, t2):
		   return model.max_SOC >= model.SOC[t2] * (100 // Batt_Enom) 
	model.c6 = Constraint(model.time2, rule=c6_rule)

	def c7_rule(model, t2):
		return model.min_SOC <= model.SOC[t2] * (100 // Batt_Enom) 
	model.c7 = Constraint(model.time2, rule=c7_rule)
	
	def c8_rule(model):
		return model.DOD == model.max_SOC - model.min_SOC
	model.c8 = Constraint(rule=c8_rule)
	
	# Objective Function: Maximize profitability
	model.obj = Objective(
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
			 - cost/model.cycles for t1 in model.time)), sense=maximize)
	    
	    
	
	# Applying the solver
	opt = SolverFactory('ipopt')
	opt.solve(model)
	# model.pprint()
	
	# Extracting data from model
	_SOC_E = [model.SOC[t1]() for t1 in model.time]
	_SOC = [i * (100 // Batt_Enom) for i in _SOC_E]
	_SOC.append(0)
	_P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	# Removes last discharge command if SOC at the end of the day is zero
	if _SOC[-1] == 0:
		_P_output[-1] = 0
	return _SOC, _P_output

#%% -- Bid cassation function -- 
def bids(Bids, rej_c, rej_d):
    rejections = ""
    rejected = False
    for bid in Bids:
        h = Bids.index(bid)
        if bid > 0:
            if rej_c[h] > 0:
                rejection = "{}:00h.".format(Bids.index(bid))
                rejections = rejections + rejection
                rejected = True
                bid = 0
        if bid < 0:
            if rej_d[h] > 0:
                rejection = "{}:00h.".format(Bids.index(bid))
                rejections = rejections + rejection 
                rejected = True
                bid = 0



    if rejected == False:
        report = "System and Market operators have accepted all bids"
    else:
        report = "Rejected bids at: " + rejections
    
    Powers = Bids
    return Powers, report
#%% -- Function callings --
SOCs, Bids = arbitrage(Batt_SOC_init, predicted_prices, Batt_Enom, Batt_Pnom, 
                       Batt_Eff, Batt_Cost)
SOCs_real, Powers_real = arbitrage(Batt_SOC_init, real_prices, Batt_Enom, 
                                   Batt_Pnom, Batt_Eff, Batt_Cost)
Powers, market_report = bids(Bids, rej_c, rej_d)
#%% -- Analysis of results --
Ben_real = []
for i in range(1, len(real_prices)):
	if i == 0:
		Benh = 0
	else:
		Benh = -Powers[i] * real_prices[i]
	Ben_real.append(round(Benh,2))

Ben_pred = []
for i in range(1, len(predicted_prices)):
	if i == 0:
		Benh = 0
	else:
		Benh = -Powers[i] * predicted_prices[i]
	Ben_pred.append(round(Benh,2))


print(market_report)
print("Benefits with real prices are {}".format(round(sum(Ben_real), 2)))
print("Expected benefits with price forecast are {}".format(round(sum(Ben_pred), 2)))
#%% -- Visualization engine --
fig = plt.figure()                              # Creating the figure

# Energy price
price_plot = fig.add_subplot(3, 1, 1)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 2.5)            # Thin horizontal grid  spacing
price_plot.set_xticks(ticks_x)
price_plot.set_yticks(ticks_y)
price_plot.set_yticks(minor_ticks_y, minor=True)
price_plot.grid(which='both')
price_plot.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inyecting the data
plt.plot(real_prices, 'b', label='Real price')
plt.plot(predicted_prices, 'r', label='Predicted price')
plt.legend()
# Adding labels
plt.ylabel('Price (€/MWh)')

# SOC
SOC_plot = fig.add_subplot(3, 1, 2)             # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)                   # Vertical grid spacing
ticks_y = np.arange(0, 105, 10)                 # Thick horizontal grid spacing
minor_ticks_y = np.arange(0, 100, 5)            # Thin horizontal grid  spacing
SOC_plot.set_xticks(ticks_x)
SOC_plot.set_yticks(ticks_y)
SOC_plot.set_yticks(minor_ticks_y, minor=True)
SOC_plot.grid(which='both')
SOC_plot.grid(which='minor', alpha=0.2)         # Thin grid thickness
SOC_plot.grid(which='major', alpha=0.7)         # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inyecting the data
plt.plot(SOCs, 'r', label='Predicted price')
plt.plot(SOCs_real, 'b', label='Real price')
plt.legend()
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot = fig.add_subplot(3, 1, 3)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pnom*1.5, Batt_Pnom*1.5, 5)                     	# Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pnom*1.5, Batt_Pnom*1.5, 2.5)               # Thin horizontal grid  spacing
P_output_plot.set_xticks(ticks_x)
P_output_plot.set_yticks(ticks_y)
P_output_plot.set_yticks(minor_ticks_y, minor=True)
P_output_plot.grid(which='both')
P_output_plot.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                                                      # X axis limits
# Inyecting the data
x = np.arange(24)
# plt.bar(x, Powers, color='g', zorder=2)
plt.bar(x-0.25, Powers, width=0.5, color='r', align='center', label='Predicted price')
plt.bar(x+0.25, Powers_real, width=0.5, color='b', align='center', label='Real price')
plt.legend()
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# Launching the plot
plt.show()
