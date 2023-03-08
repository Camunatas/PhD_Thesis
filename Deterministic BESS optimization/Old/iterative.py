# --- Arbitrage of a standalone battery connected to the grid which uses price prediction and iterates when gets the real price ---
# Author: Pedro Luis Camuñas
# Date: 25/04/2020

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

# -- Simulation parameters --
Batt_Emax = 4.5                       # Rated battery energy (MWh)
Batt_Pmax = 2             # Rated battery power (MW)
Batt_Efficiency = 0.9               # Rated battery efficiency

# -- Data initialization --
SOC_i = 0
Powers = []
SOCs = []
Prices_real = []
Prices_pred = []

# -- Data importing --
data = pd.read_excel('Prices_week.xlsx', sheet_name='Prices', nrows=200)
Price = list(data['Price'][0:169])
k_pred = 0.1

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
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
		          for t1 in model.time)), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model)
	# model.pprint()

	# Extracting data from model
	# Schedule = [model.ESS_D[t1]() - model.ESS_C[t1]() for t1 in model.time]
	# Charge = [model.ESS_C[t1]() + model.ESS_D[t1]() for t1 in model.time]
	SOC = [model.SOC[t1]() for t1 in model.time]
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	SOC = [i * (100 // Batt_Emax) for i in SOC]

	return SOC, P_output


for n in range(0, 6):
	Price_real = Price[24*n:24*(n+1)]
	Prices_real = Prices_real + Price_real
	Price_pred = [i + i * rand.uniform(-k_pred, k_pred) for i in Price_real]
	Prices_pred = Prices_pred + Price_pred
	SOC, Power = daily(SOC_i, Price_real, Batt_Emax, Batt_Pmax, Batt_Efficiency)
	SOCs[24*n:24*(n+1)] = SOC
	Powers[24*n:24*(n+1)] = Power
	SOC, Power = daily(SOC_i, Price_pred, Batt_Emax, Batt_Pmax, Batt_Efficiency)
	SOCs = SOCs + SOC
	Powers = Powers + Power
	print('Completed day {}'.format(n+1))

print(len(SOCs)/24)
print(len(Powers)/24)
print('SOCs length is {}'.format(len(SOCs)))
print('Powers length is {}'.format(len(Powers)))

# -- Plots --
fig = plt.figure()
# Energy price (real)
price_real_plot = fig.add_subplot(4, 1, 1)
plt.plot(Prices_real, 'r', label='Charge')
plt.ylabel('Price (real) (€/MWh)')
# Energy price (real)
price_pred_plot = fig.add_subplot(4, 1, 2)
plt.plot(Prices_pred, 'r', label='Charge')
plt.ylabel('Price (predicted) (€/MWh)')
# SOC
SOC_plot = fig.add_subplot(4, 1, 3)
plt.plot(SOCs, 'b', label='Charge')
plt.ylabel('SOC (%)')
# Power
P_output_plot = fig.add_subplot(4, 1, 4)
x = np.arange(len(Powers))
plt.bar(x, Powers, color='g', zorder=2)
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')
# Launching the plot
plt.show()