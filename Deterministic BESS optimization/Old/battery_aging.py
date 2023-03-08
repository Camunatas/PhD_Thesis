# --- Arbitration of a standalone battery connected to the grid ---
# Author: Pedro Luis Camuñas
# Date: 19/11/2019

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Simulation parameters --
Batt_Emax = 4                       # Rated battery energy (MWh)
Batt_Pmax = Batt_Emax/4             # Rated battery power (MW)
Batt_Efficiency = 0.9               # Rated battery efficiency
Batt_Price = 100000                  # Battery price (€)

# -- Data initialization --
SOC_i = 0                           # Initial battery SOC

# Initializing the daily market schedule with zeros
# Schedule_DM = []
# for i in range(24):
# 	Schedule_DM.append(0)

# -- Data importing --
data = pd.read_excel('Prices.xlsx', sheet_name='Prices')
Price = list(data['Price'][0:25])


# -- Daily market --
def daily(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency, batt_price):
	# Model initialization
	model = ConcreteModel()
	model.time = range(24)
	model.time2 = range(1, 24)
	model.time3 = range(25)
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
		expr=sum((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]) -
		          (batt_price * 0.01175 * 0.01 * ((model.ESS_D[t1] + model.ESS_C[t1])/batt_capacity)))
		         for t1 in model.time), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model, tee=True, keepfiles=True)
	# model.pprint()

	# Extracting data from model
	# Schedule = [model.ESS_D[t1]() - model.ESS_C[t1]() for t1 in model.time]
	# Charge = [model.ESS_C[t1]() + model.ESS_D[t1]() for t1 in model.time]
	SOC = [model.SOC[t1]() for t1 in model.time]
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	Cycle_cost = [batt_price * 0.01175 * 0.0001 * ((model.ESS_D[t1]() + model.ESS_C[t1]())/batt_capacity) for t1 in model.time]
	Cap_fade = [0.01*0.01175 * 0.01 * (1-(model.ESS_D[t1]() + model.ESS_C[t1]())/batt_capacity) for t1 in model.time]
	for i in range(len(SOC)):
		if SOC[i] == 0 and P_output[i] < 0:
			P_output[i] = 0
			Cycle_cost[i] = 0
	return SOC, P_output, Cycle_cost, Cap_fade


SOC, P_output, Cycle_cost, Cap_fade = daily(SOC_i, Price, Batt_Emax, Batt_Pmax, Batt_Efficiency, Batt_Price)
print(Cycle_cost)
print(Cap_fade)
SOC = [i * (100 // Batt_Emax) for i in SOC]
SOC.append(0)
# -- Plots --
fig = plt.figure()                              # Creating the figure

# Energy price
price_plot = fig.add_subplot(4, 1, 1)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
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
axes.set_xlim([0, 24])                          # X axis limits
# Inyecting the data
plt.plot(Price, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')

# SOC
SOC_plot = fig.add_subplot(4, 1, 2)             # Creating subplot
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
plt.plot(SOC, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot = fig.add_subplot(4, 1, 3)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
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
axes.set_xlim([0, 24])                          # X axis limits
# Inyecting the data
x = np.arange(24)
plt.bar(x, P_output, color='c', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# Cycle cost
P_output_plot = fig.add_subplot(4, 1, 4)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
# ticks_y = np.arange(0, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot.set_xticks(ticks_x)
# P_output_plot.set_yticks(ticks_y)
# P_output_plot.set_yticks(minor_ticks_y, minor=True)
P_output_plot.grid(which='both')
P_output_plot.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inyecting the data
x = np.arange(24)
plt.bar(x, Cycle_cost, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Cycle cost (€)')

# Launching the plot
plt.show()


