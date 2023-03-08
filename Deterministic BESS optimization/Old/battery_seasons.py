# --- Arbitration of a standalone battery connected to the grid with four seasonal different price curves---
# Author: Pedro Luis Camuñas
# Date: 19/11/2019

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Simulation parameters --
Batt_Emax = 5                       # Rated battery energy (MWh)
Batt_Pmax = 3             # Rated battery power (MW)
Batt_Efficiency = 0.9               # Rated battery efficiency

# -- Data initialization --
SOC_i = 0                           # Initial battery SOC

# Initializing the daily market schedule with zeros
# Schedule_DM = []
# for i in range(24):
# 	Schedule_DM.append(0)

# -- Data importing --
data = pd.read_excel('Prices_seasonal.xlsx', sheet_name='Prices', nrows=36)
Price_winter = list(data['Winter'][0:25])
Price_autumn = list(data['Autumn'][0:25])
Price_summer = list(data['Summer'][0:25])
Price_spring = list(data['Spring'][0:25])


# -- Daily market --
def daily(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency):
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
		expr=sum(((energy_price[t1] * (model.ESS_D[t1]*1.2 - model.ESS_C[t1]))
		          for t1 in model.time)), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model)
	#model.pprint()

	# Extracting data from model
	SOC = [model.SOC[t1]() for t1 in model.time]
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	SOC = [i * (100 // Batt_Emax) for i in SOC]
	SOC.append(0)
	if SOC[-1] == 0:
		P_output[-1] = 0
	return SOC, P_output


SOC_winter, P_output_winter = daily(SOC_i, Price_winter, Batt_Emax, Batt_Pmax, Batt_Efficiency)
SOC_spring, P_output_spring = daily(SOC_i, Price_spring, Batt_Emax, Batt_Pmax, Batt_Efficiency)
SOC_summer, P_output_summer = daily(SOC_i, Price_summer, Batt_Emax, Batt_Pmax, Batt_Efficiency)
SOC_autumn, P_output_autumn = daily(SOC_i, Price_autumn, Batt_Emax, Batt_Pmax, Batt_Efficiency)
# Removes last discharge command if SOC at the end of the day is zero

# -- Plots --
fig = plt.figure()                              # Creating the figure

# WINTER
# Energy price
price_plot_winter = fig.add_subplot(3, 4, 1)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 1)            # Thin horizontal grid  spacing
price_plot_winter.set_xticks(ticks_x)
price_plot_winter.set_yticks(ticks_y)
price_plot_winter.set_yticks(minor_ticks_y, minor=True)
price_plot_winter.grid(which='both')
price_plot_winter.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot_winter.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(Price_winter, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')
# Adding title
plt.title('Winter')
# SOC
SOC_plot_winter = fig.add_subplot(3, 4, 5)              # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)                   # Vertical grid spacing
ticks_y = np.arange(0, 105, 10)                 # Thick horizontal grid spacing
minor_ticks_y = np.arange(0, 100, 5)            # Thin horizontal grid  spacing
SOC_plot_winter.set_xticks(ticks_x)
SOC_plot_winter.set_yticks(ticks_y)
SOC_plot_winter.set_yticks(minor_ticks_y, minor=True)
SOC_plot_winter.grid(which='both')
SOC_plot_winter.grid(which='minor', alpha=0.2)         # Thin grid thickness
SOC_plot_winter.grid(which='major', alpha=0.7)         # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(SOC_winter, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot_winter = fig.add_subplot(3, 4, 9)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot_winter.set_xticks(ticks_x)
P_output_plot_winter.set_yticks(ticks_y)
P_output_plot_winter.set_yticks(minor_ticks_y, minor=True)
P_output_plot_winter.grid(which='both')
P_output_plot_winter.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot_winter.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                                                      # X axis limits
# Inserting the data
x = np.arange(24)
plt.bar(x, P_output_winter, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# SPRING
# Energy price
price_plot_spring = fig.add_subplot(3, 4, 2)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 1)            # Thin horizontal grid  spacing
price_plot_spring.set_xticks(ticks_x)
price_plot_spring.set_yticks(ticks_y)
price_plot_spring.set_yticks(minor_ticks_y, minor=True)
price_plot_spring.grid(which='both')
price_plot_spring.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot_spring.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(Price_spring, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')
# Adding title
plt.title('Spring')

#SOC
SOC_plot_spring = fig.add_subplot(3, 4, 6)              # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)                   # Vertical grid spacing
ticks_y = np.arange(0, 105, 10)                 # Thick horizontal grid spacing
minor_ticks_y = np.arange(0, 100, 5)            # Thin horizontal grid  spacing
SOC_plot_spring.set_xticks(ticks_x)
SOC_plot_spring.set_yticks(ticks_y)
SOC_plot_spring.set_yticks(minor_ticks_y, minor=True)
SOC_plot_spring.grid(which='both')
SOC_plot_spring.grid(which='minor', alpha=0.2)         # Thin grid thickness
SOC_plot_spring.grid(which='major', alpha=0.7)         # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(SOC_spring, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot_spring = fig.add_subplot(3, 4, 10)                                    # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot_spring.set_xticks(ticks_x)
P_output_plot_spring.set_yticks(ticks_y)
P_output_plot_spring.set_yticks(minor_ticks_y, minor=True)
P_output_plot_spring.grid(which='both')
P_output_plot_spring.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot_spring.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                                                      # X axis limits
# Inserting the data
x = np.arange(24)
plt.bar(x, P_output_spring, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# SUMMER
# Energy price
price_plot_summer = fig.add_subplot(3, 4, 3)          # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 1)            # Thin horizontal grid  spacing
price_plot_summer.set_xticks(ticks_x)
price_plot_summer.set_yticks(ticks_y)
price_plot_summer.set_yticks(minor_ticks_y, minor=True)
price_plot_summer.grid(which='both')
price_plot_summer.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot_summer.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(Price_summer, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')
# Adding title
plt.title('Summer')
# SOC
SOC_plot_summer = fig.add_subplot(3, 4, 7)              # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)                   # Vertical grid spacing
ticks_y = np.arange(0, 105, 10)                 # Thick horizontal grid spacing
minor_ticks_y = np.arange(0, 100, 5)            # Thin horizontal grid  spacing
SOC_plot_summer.set_xticks(ticks_x)
SOC_plot_summer.set_yticks(ticks_y)
SOC_plot_summer.set_yticks(minor_ticks_y, minor=True)
SOC_plot_summer.grid(which='both')
SOC_plot_summer.grid(which='minor', alpha=0.2)         # Thin grid thickness
SOC_plot_summer.grid(which='major', alpha=0.7)         # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(SOC_summer, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot_summer = fig.add_subplot(3, 4, 11)                                     # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot_summer.set_xticks(ticks_x)
P_output_plot_summer.set_yticks(ticks_y)
P_output_plot_summer.set_yticks(minor_ticks_y, minor=True)
P_output_plot_summer.grid(which='both')
P_output_plot_summer.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot_summer.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                                                      # X axis limits
# Inserting the data
x = np.arange(24)
plt.bar(x, P_output_summer, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# AUTUMN
# Energy price
price_plot_autumn = fig.add_subplot(3, 4, 4)           # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                   # Vertical grid spacing
ticks_y = np.arange(30, 80, 5)                  # Thick horizontal grid spacing
minor_ticks_y = np.arange(30, 80, 1)            # Thin horizontal grid  spacing
price_plot_autumn.set_xticks(ticks_x)
price_plot_autumn.set_yticks(ticks_y)
price_plot_autumn.set_yticks(minor_ticks_y, minor=True)
price_plot_autumn.grid(which='both')
price_plot_autumn.grid(which='minor', alpha=0.2)       # Thin grid thickness
price_plot_autumn.grid(which='major', alpha=0.7)       # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(Price_autumn, 'r', label='Charge')
# Adding labels
plt.ylabel('Price (€/MWh)')
# Adding title
plt.title('Autumn')
# SOC
SOC_plot_autumn = fig.add_subplot(3, 4, 8)              # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)                   # Vertical grid spacing
ticks_y = np.arange(0, 105, 10)                 # Thick horizontal grid spacing
minor_ticks_y = np.arange(0, 100, 5)            # Thin horizontal grid  spacing
SOC_plot_autumn.set_xticks(ticks_x)
SOC_plot_autumn.set_yticks(ticks_y)
SOC_plot_autumn.set_yticks(minor_ticks_y, minor=True)
SOC_plot_autumn.grid(which='both')
SOC_plot_autumn.grid(which='minor', alpha=0.2)         # Thin grid thickness
SOC_plot_autumn.grid(which='major', alpha=0.7)         # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                          # X axis limits
# Inserting the data
plt.plot(SOC_autumn, 'b', label='Charge')
# Adding labels
plt.ylabel('SOC (%)')

# Power
P_output_plot_autumn = fig.add_subplot(3, 4, 12)                                     # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)                                               # Vertical grid spacing
ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.5)                     # Thick horizontal grid spacing
minor_ticks_y = np.arange(-Batt_Pmax*1.5, Batt_Pmax*1.5, 0.1)               # Thin horizontal grid  spacing
P_output_plot_autumn.set_xticks(ticks_x)
P_output_plot_autumn.set_yticks(ticks_y)
P_output_plot_autumn.set_yticks(minor_ticks_y, minor=True)
P_output_plot_autumn.grid(which='both')
P_output_plot_autumn.grid(which='minor', alpha=0.2, zorder=1)                      # Thin grid thickness
P_output_plot_autumn.grid(which='major', alpha=0.7)                                # Thick grid thickness
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])                                                      # X axis limits
# Inserting the data
x = np.arange(24)
plt.bar(x, P_output_autumn, color='g', zorder=2)
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')

# Launching the plot
plt.show()

