#%% Load python libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#%% Load other project files
from aux_fcns import *
from RHU_model import *
from plt_fcns import *
from sim_fcns import *
#%% RHU parameters
RHU_Parameters = {}
# RHU configuration
RHU_Parameters['Degradation'] = False                                                # Degradation mode enabler
RHU_Parameters['Arbitrage'] = False                                                  # Arbitrage service enabler
RHU_Parameters['Electricity Market'] = True                                         # Electricity Market enabler
RHU_Parameters['Purging'] = False                                                    # AEL purging enabler
RHU_Parameters['AEL Pruchases'] = True                                             # AEL can purchase energy
# AEL parameters
RHU_Parameters['AEL Maximum power'] = 1                                             # [MW]
RHU_Parameters['AEL Minimum power'] = 0.3                                           # [MW]
RHU_Parameters['AEL efficiency'] = 0.75                                             # [p.u.]
RHU_Parameters['Tank capacity'] = 1e100                                             # [kg]
RHU_Parameters['Tank emptying'] = 1                                                 # Number of days between emptyings
RHU_Parameters['Initial hydrogen level']  = 0                                       # [kg]
RHU_Parameters['AEL investment cost']  = 500 * RHU_Parameters['AEL Maximum power']  # [€/kW]
RHU_Parameters['Lifetime hours'] = 100e3                                            # Operating ours before EOL
RHU_Parameters['Lifetime cycles']  = 5000                                           # Start/stop cycles before EOL
RHU_Parameters['Initial state']  = 0                                                # Initial state (0= cold, 5 = hot)
RHU_Parameters['Cold start time']  = 0.2                                            # [h]
RHU_Parameters['Off time']  = 5                                                     # [h]
RHU_Parameters['HHV'] = 0.0394                                                      # [kg*MWh] Hydrogen High Heating Value
RHU_Parameters['Idle state power'] = 0.04                                           # [MW] Idle state power consumption
RHU_Parameters['Off state power'] = 0.007                                           # [MW] Off state power consumption
RHU_Parameters['AEL initial H2 in O2'] = 0                                          # [%] Initial H2 concentration in O2
# BESS paremeters
RHU_Parameters['Batt_SOCi'] = 0                                                     # [p.u.] Initial SOC
RHU_Parameters['Batt_E'] = 10                                                       # [MWh] Default Battery Capacity
RHU_Parameters['Batt_P'] = 2.5                                                      # [MW] Default Battery Power
RHU_Parameters['Batt_Eff'] = 0.9                                                    # [p.u.] Default Battery Efficiency
RHU_Parameters['Batt_Cost'] = 50 * RHU_Parameters['Batt_E']                         # [€/kWh] Default Battery Cost
RHU_Parameters['Batt_EOL'] = 0.8                                                    # [p.u.] BESS capacity at EOL
# Degradation models regulators
RHU_Parameters['K_AEL'] = 1
RHU_Parameters['K_BESS'] = 1
# Hydrogen price
Price_H2 = 4                                                                      # [€/kg] Hydrogen price
#%% Load dataset
Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()

#%% Launching study cases for paper
day_start = '2018-01-01'
# day_end = '2018-01-05'
day_end = '2018-12-31'
simulations = {}
# #%% Case: Electriciy market participation is enabled
# cases = []
# Results_default = []
# for Price_H2 in np.arange(0,3.5,0.5):
#     case = Price_H2
#     Results_case = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
#                                   f'Default {case} ')
#     cases.append(case)
#     Results_default.append(Results_case)
#
# # Case: Electriciy market participation is enabled (AEL purchases disabled)
# RHU_Parameters['AEL Pruchases'] = False
# Results_default_nopurch = []
# for Price_H2 in np.arange(0,3.5,0.5):
#     case = Price_H2
#     Results_case = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
#                                   f'Default, nopurch {case} ')
#     Results_default_nopurch.append(Results_case)
#
#
# # Case: Hydrogen market participation is disabled
# RHU_Parameters['Electricity Market'] = True
# RHU_Parameters['Arbitrage'] = True
# RHU_Parameters['Tank capacity'] = 0
# case = 'No hydrogen'
# Results_onlyel = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
#                                   f'No H2 {case} ')
#
# # Plotting
# # Arranging data for plotting
# x = cases
# y_default = []
# for i in range(len(x)):
#     y_default.append(np.round(sum(Results_default[i]['Generated H2'])/1000,2))
# y_default_nopurch = []
# for i in range(len(x)):
#     y_default_nopurch.append(np.round(sum(Results_default_nopurch[i]['Generated H2'])/1000,2))
#
# # y_onlyel = np.round(sum(Results_onlyel['Generated H2'])/1000,2)
# # Launching plot
# fig = plt.figure('Hydrogen price effect')
# plt.plot(x,y_default_nopurch, color='b', linestyle= '--', label='AEL not connected to grid')
# plt.plot(x,y_default, color='b', label='AEL connected to grid')
# # plt.axhline(y = y_onlyel, color = 'g', label='No H2 market')
# plt.xlabel('Hydrogen price (€/kg)')
# plt.ylabel('Produced hydrogen (tonnes)')
# plt.legend()
# plt.grid()
# plt.show()

# Hydrogen infraestructure


#%% Running infraestructure cases
def run_simulation_structure(Price_H2):
    Results_1day = []
    Results_2day = []
    Results_3day = []
    for tank_cap in np.arange(0,2550,500):
        case = tank_cap
        RHU_Parameters['Tank emptying'] = 1
        RHU_Parameters['Tank capacity'] = tank_cap
        Results_case = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
                                      f'{case}  daily ')
        Results_1day.append(Results_case)
        RHU_Parameters['Tank emptying'] = 3
        Results_case = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
                                      f'{case} every 3 days')
        Results_2day.append(Results_case)
        RHU_Parameters['Tank emptying'] = 5
        Results_case = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
                                      f'{case} every 5 days')
        Results_3day.append(Results_case)

    # Case: Unlimited pipeline
    case = 'Pipeline'
    RHU_Parameters['Tank capacity'] = 1e100
    RHU_Parameters['Tank emptying'] = 1
    Results_pipeline = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset, case)

    return Results_1day, Results_2day, Results_3day, Results_pipeline
cases = np.arange(0,2550,500)
# Simulation with low hydrogen price
Price_H2 = 2
Results_1day_low, Results_2day_low, Results_3day_low, Results_pipeline_low = run_simulation_structure(Price_H2)

# Simulation with medium hydrogen price
# Price_H2 = 4
# Results_1day_medium, Results_2day_medium, Results_3day_medium, Results_pipeline_medium = run_simulation_structure(Price_H2)
# Simulation with high hydrogen price
Price_H2 = 6
Results_1day_high, Results_2day_high, Results_3day_high, Results_pipeline_high = run_simulation_structure(Price_H2)

#%% Plotting
# Hydrogen infraestructure effect on h2 production
x = cases
y_1day_low = []
y_2day_low = []
y_3day_low = []
y_1day_medium = []
y_2day_medium = []
y_3day_medium = []
y_1day_high = []
y_2day_high = []
y_3day_high = []
for i in range(len(x)):
    y_1day_low.append(np.round(Results_1day_low[i]['Hydrogen production benefits'] +
                           Results_1day_low[i]['Energy market accumulated benefits'],2))
    y_2day_low.append(np.round(Results_2day_low[i]['Hydrogen production benefits'] +
                           Results_2day_low[i]['Energy market accumulated benefits'],2))
    y_3day_low.append(np.round(Results_3day_low[i]['Hydrogen production benefits'] +
                           Results_3day_low[i]['Energy market accumulated benefits'],2))
    y_pipeline_low = np.round(Results_pipeline_low['Hydrogen production benefits'] +
                           Results_pipeline_low['Energy market accumulated benefits'],2)
    # y_1day_medium.append(np.round(Results_1day_medium[i]['Hydrogen production benefits'] +
    #                        Results_1day_medium[i]['Energy market accumulated benefits'],2))
    # y_2day_medium.append(np.round(Results_2day_medium[i]['Hydrogen production benefits'] +
    #                        Results_2day_medium[i]['Energy market accumulated benefits'],2))
    # y_3day_medium.append(np.round(Results_3day_medium[i]['Hydrogen production benefits'] +
    #                        Results_3day_medium[i]['Energy market accumulated benefits'],2))
    # y_pipeline_medium = np.round(Results_pipeline_medium['Hydrogen production benefits'] +
    #                        Results_pipeline_medium['Energy market accumulated benefits'],2)
    y_1day_high.append(np.round(Results_1day_high[i]['Hydrogen production benefits'] +
                           Results_1day_high[i]['Energy market accumulated benefits'],2))
    y_2day_high.append(np.round(Results_2day_high[i]['Hydrogen production benefits'] +
                           Results_2day_high[i]['Energy market accumulated benefits'],2))
    y_3day_high.append(np.round(Results_3day_high[i]['Hydrogen production benefits'] +
                           Results_3day_high[i]['Energy market accumulated benefits'],2))
    y_pipeline_high = np.round(Results_pipeline_high['Hydrogen production benefits'] +
                           Results_pipeline_high['Energy market accumulated benefits'],2)

fig = plt.figure('Hydrogen price effect')
low_prices_subplot = fig.add_subplot(2, 1, 1)
plt.title('Hydrogen price: 2€/kg')
plt.plot(x,y_1day_low, linestyle = 'solid', label='Emptying daily')
plt.plot(x,y_2day_low, linestyle = 'solid', label='Emptying each 3 days ')
plt.plot(x,y_3day_low, linestyle = 'solid', label='Emptying each 5 days')
plt.axhline(y = y_pipeline_low, color='r', linestyle = 'solid', label='Pipeline')
plt.ylabel('Accumulated benefits (€)')
plt.legend()
plt.grid()
high_prices_subplot = fig.add_subplot(2, 1, 2)
plt.title('Hydrogen price: 6€/kg')
plt.plot(x,y_1day_high, linestyle = 'solid', label='Emptying daily')
plt.plot(x,y_2day_high, linestyle = 'solid', label='Emptying each 3 days')
plt.plot(x,y_3day_high, linestyle = 'solid', label='Emptying each 5 days')
plt.axhline(y = y_pipeline_high, color='r', linestyle = 'solid', label='Pipeline')
plt.xlabel('Tank capacity (kg)')
plt.ylabel('Accumulated benefits (€)')
plt.legend()
plt.grid()
plt.show()