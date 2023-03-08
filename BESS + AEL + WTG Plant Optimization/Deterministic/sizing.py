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
RHU_Parameters['Degradation'] = True                                                # Degradation mode enabler
RHU_Parameters['Arbitrage'] = True                                                  # Arbitrage service enabler
RHU_Parameters['Purging'] = True                                                    # AEL purging enabler
# AEL parameters
RHU_Parameters['AEL Maximum power'] = 1                                             # [MW]
RHU_Parameters['AEL Minimum power'] = 0.3                                           # [MW]
RHU_Parameters['AEL efficiency'] = 0.75                                             # [p.u.]
RHU_Parameters['Tank capacity'] = 1e100                                             # [kg]
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
# Hydrogen price
Price_H2 = 5                                                                        # [€/kg] Hydrogen price
#%% Load dataset
Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()

#%% Launch K sweep
# Sizing_Results = {}
# # for K_BESS in np.arange(0.1, 2, 0.2):
# #     for K_AEL in np.arange(0.1, 2, 0.2):
# for K_BESS in np.arange(0.2, 2, 0.4):
#     for K_AEL in np.arange(0.2, 2, 0.4):
#         Size_Results = {}
#         # Set parameters for local simulation
#         RHU_Parameters['K_AEL'] = K_AEL
#         RHU_Parameters['K_BESS'] = K_BESS
#         # Launch local simulation
#         day_start = '2018-01-01'
#         day_end = '2018-12-31'
#         # Arbitrage & degradation case
#         Results_local = run_simulation(day_start, day_end, RHU_Parameters, Price_H2, Global_dataset,
#                                          f'K_BESS: {K_BESS}, K_AEL: {K_AEL}')
#         Local_bens = Results_local['Energy market accumulated benefits'] + Results_local['Hydrogen production benefits']
#         # Saving local results
#         Size_Results['K_BESS'] = K_BESS
#         Size_Results['K_AEL'] = K_AEL
#         Size_Results['Benefits'] = np.round(Local_bens, 2)
#         Size_Results['Deg BESS'] = Results_local['BESS accumulated degradation'][-1]
#         Size_Results['Deg AEL'] = Results_local['AEL accumulated degradation'][-1]
#         Sizing_Results[f'K_BESS = {K_BESS}, K_AEL = {K_AEL}'] = Size_Results
#
# #%% Extrapolate results for full project
# import numpy_financial as npf
# for size in Sizing_Results:
#     size_dict = Sizing_Results[f'{size}']
#     daily_ben = size_dict['Benefits']/365
#     daily_CF = []
#     for i in range(3650):
#         daily_CF.append(daily_ben)
#     Sizing_Results[f'{size}']['NPV'] = npf.npv((1+0.07)**(1/365)-1, daily_CF)
#
# # Save global results
# np.save('Sizing_Results.npy', Sizing_Results)

#%% K sweep time calculator
# i = 0
# run_time = 76
# for K_BESS in np.arange(0.2, 2, 0.4):
#     for K_AEL in np.arange(0.2, 2, 0.4):
#         i+=1
# print(f'It will take: {run_time*i/3600} hours, it will have {i} points')
#
#%% Load & plot data
# Load data
Sizing_Results = np.load('sizing_Results.npy', allow_pickle=True).item()
# Generating 2-D arrays
X = []
Y = []
Z = []
for size in Sizing_Results:
    X.append(Sizing_Results[f'{size}']['K_BESS'])
    Y.append(Sizing_Results[f'{size}']['K_AEL'])
    Z.append(Sizing_Results[f'{size}']['NPV'])
mesh_shape = int(np.sqrt(min(len(X), len(Z), len(Y))))
x = np.reshape(X[:mesh_shape**2], (mesh_shape, mesh_shape))
y = np.reshape(Y[:mesh_shape**2], (mesh_shape, mesh_shape))
z = np.reshape(Z[:mesh_shape**2], (mesh_shape, mesh_shape))
# Generating and saving figure
fig = plt.figure('Sizing results')
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_xlabel('K_BESS')
ax.set_ylabel('K_AEL')
ax.set_zlabel('NPV (€)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()