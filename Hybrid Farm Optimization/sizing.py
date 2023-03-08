#%% Importing libraries
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import time
#%% Importing external files
from arb_fcns import *
from aux_fcns import *
from plot_fcns import *
from sim_fcns import *
#%% Importing dataset
Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()
#%% Simulation parameters
starting_day = '2018-01-01'
ending_day = '2018-12-31'
sim_name = 'CF sizing'
Results = {}
#%% Hybrid plant parameters default case definition
HyF_Parameters = {}
# General parameters
HyF_Parameters['ESS Capacity'] = 10                             # [MWh] ESS nominal capacity
HyF_Parameters['ESS Nominal Power'] = 10/4                      # [MW] ESS nominal power
HyF_Parameters['ESS Efficiency'] = 0.9                          # ESS efficiency
HyF_Parameters['Inverter Pnom'] = 7                             # [MW] Inverter nominal power
HyF_Parameters['ESS Replacement Cost'] = 33.77 * 1000 * 10      # [€] ESS estimated replacement costs
HyF_Parameters['ESS Initial SOC'] = 0                           # Initial SOC
HyF_Parameters['ESS EOL'] = 0.8                                 # ESS EOL capacity
HyF_Parameters['ESS dumping SOC'] = 75                          # ESS maximum SOC before dumping
# Configuration parameters for default case
HyF_Parameters['Config'] = {}
HyF_Parameters['Config']['Degradation'] = False                 # Degradation model is enabled
HyF_Parameters['Config']['Ideal'] = False                       # Ideal operation is enabled
HyF_Parameters['Config']['ID Participation'] = True             # ID participation is enabled
HyF_Parameters['Config']['ID Arbitrage'] = False                # Arbitrage on ID is enabled
HyF_Parameters['Config']['ESS DM Participation'] = False         # ESS participates on DM
HyF_Parameters['Config']['SOC Dump'] = False                    # Enabling SOC dump on ID market
HyF_Parameters['Config']['Variable Efficiency'] = False         # Enabling inverter variable efficiency on arbitrage
HyF_Parameters['Config']['Daily plotting'] = False              # Daily plots are enabled
#%% Sizing parameters
Es = np.arange(2,20,1)
Ps = np.arange(1,5,1)
# Simulation time calculation
t = 0
sims = 0
for E in Es:
    for P in Ps:
        t = t + (254/60)
        sims += 1

print(f'{sims} simulations will be carried away')
if t < 60:
    print(f'Sizing time will take aproximately {t} minutes')
else:
    print(f'Sizing time will take aproximately {np.round(t/60,2)} hours')
#%% Launching sizing simulation
sim_time = datetime.now()
sim_timer = time.time()
sim_folder = 'Results/' + sim_time.strftime("%m_%d_%H_%M_") + sim_name
Sizing_Results = {}
for E in Es:
    for P in Ps:
        Size_Results = {}
        HyF_Parameters['ESS Capacity'] = E
        HyF_Parameters['ESS Nominal Power'] = P
        case_name = f'{E}MWh {P}MW'
        Local_Results = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day,
                                         ending_day, Results, sim_time)
        Size_Results['E'] = E
        Size_Results['P'] = P
        Size_Results['Results'] = np.round(local_measure_accumulator(Local_Results, 'Ben_DM_Real_MWh'), 2)
        Sizing_Results[f'{E}MWh {P}MW'] = Size_Results
# Saving results
np.save(sim_folder + '/Sizing_Results.npy', Sizing_Results)
# Stopping timer
if round((time.time() - sim_timer) / 3600) > 1:
    print(f'Global simulation elapsed time: {round((time.time() - sim_timer) / 3600)}h')
else:
    print(f'Global simulation elapsed time: {round((time.time() - sim_timer)) / 60} minutes')
#%% Plotting sizing figure
# Generating 2-D arrays
X = []
Y = []
Z = []
for size in Sizing_Results:
    X.append(Sizing_Results[f'{size}']['E'])
    Y.append(Sizing_Results[f'{size}']['P'])
    Z.append(Sizing_Results[f'{size}']['Results'])
mesh_shape = int(np.sqrt(min(len(X), len(Z), len(Y))))
x = np.reshape(X[:mesh_shape**2], (mesh_shape, mesh_shape))
y = np.reshape(Y[:mesh_shape**2], (mesh_shape, mesh_shape))
z = np.reshape(Z[:mesh_shape**2], (mesh_shape, mesh_shape))
# Generating and saving figure
fig = plt.figure('Sizing results')
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_xlabel('Capacity (MWh)')
ax.set_ylabel('Power (MW)')
ax.set_zlabel('Benefits per MWh (€/MWh)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(sim_folder + '/Sizing Results.png')
plt.show()