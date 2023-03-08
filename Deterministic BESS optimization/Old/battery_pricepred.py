# --- Arbitration of a standalone battery connected to the grid ---
# Author: Pedro Luis Camuñas
# Date: 02/03/2020

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

# -- Simulation parameters --
Batt_Emax = 4                       # [MW] Rated battery energy
Batt_Pmax = Batt_Emax/4             # [MW] Rated battery power
Batt_Efficiency = 0.9               # [%] Rated battery efficiency
Prediction_error_1 = 0.15			# [%] Prediction error for 1 day (%)
Prediction_error_2 = 0.3            # [%] Prediction error for 2 days (%)

# -- Data initialization --
SOC_i = 0                           # [%] Initial battery SOC

# Initializing the daily market schedule with zeros
# Schedule_DM = []
# for i in range(24):
# Schedule_DM.append(0)

# -- Data importing --
data = pd.read_excel('Prices.xlsx', sheet_name='Prices')

# -- Generating price prediction
Price_0 = list(data['Price'][0:25])
Price_1 =  [i + i*rand.uniform(-Prediction_error_1, Prediction_error_1) for i in Price_0]
Price_2 =  [i + i*rand.uniform(-Prediction_error_1, Prediction_error_2) for i in Price_0]
Price_pred = Price_0 + Price_1 + Price_2
Price_real = Price_0 + Price_0 + Price_0


# -- Daily market --
def daily(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency):
	# Model initialization
	model = ConcreteModel()
	model.time = range(3*24)
	model.time2 = range(1, 3*24)
	model.time3 = range(2*24+25)
	model.SOC = Var(model.time3, bounds=(0, batt_capacity))         # Battery SOC
	model.not_charging = Var(model.time, domain=Binary)             # Charge verifier
	model.not_discharging = Var(model.time, domain=Binary)          # Discharge verifier
	model.ESS_C = Var(model.time, bounds=(0, batt_maxpower))        # Energy being charged
	model.ESS_D = Var(model.time, bounds=(0, batt_maxpower))        # Energy being discharged

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

	# Objective Function: Maximize profitability
	model.obj = Objective(
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
		          for t1 in model.time)), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model)
	model.pprint()

	# Extracting data from model
	# Schedule = [model.ESS_D[t1]() - model.ESS_C[t1]() for t1 in model.time]
	# Charge = [model.ESS_C[t1]() + model.ESS_D[t1]() for t1 in model.time]
	SOC = [model.SOC[t1]() for t1 in model.time]
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	return  SOC, P_output

# Running optimization problem with predicted prices
SOC, P_output = daily(SOC_i, Price_pred, Batt_Emax, Batt_Pmax, Batt_Efficiency)
SOC = [i * (100 // Batt_Emax) for i in SOC]
SOC.append(0)

# Removes last discharge command if SOC at the end of the day is zero in both cases
if SOC[-1] == 0:
	P_output[-1] = 0

# -- Plots --
fig = plt.figure()                              # Creating the figure

# Energy price
price_plot = fig.add_subplot(3, 1, 1)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 2*24+25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 1)            # Thin horizontal grid  spacing
price_plot.set_xticks(ticks_x)
price_plot.set_yticks(ticks_y)
price_plot.set_yticks(minor_ticks_y, minor=True)
price_plot.grid(which='both')
price_plot.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 3*24])                          # X axis limits
# Inyecting the data
plt.plot(Price_pred, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')

# SOC
SOC_plot = fig.add_subplot(3, 1, 2)             # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 3*24, 1)                   # Vertical grid spacing
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
axes.set_xlim([0, 3*24])                          # X axis limits
# Inyecting the data
plt.plot(SOC, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot = fig.add_subplot(3, 1, 3)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 24*2+25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot.set_xticks(ticks_x)
P_output_plot.set_yticks(ticks_y)
P_output_plot.set_yticks(minor_ticks_y, minor=True)
P_output_plot.grid(which='both')
P_output_plot.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24*3])                                                      # X axis limits
# Inyecting the data
x = np.arange(24*3)
plt.bar(x, P_output, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# Launching the plot
plt.show()

