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
sim_name = 'Year simulation for paper'
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

#%% Launching simulation
sim_time = datetime.now()
sim_timer = time.time()
Cases = []              # Array of cases
sim_folder = 'Results/' + sim_time.strftime("%m_%d_%H_%M_") + sim_name

# case_name = 'Standalone'
# HyF_Parameters['Config']['ID Participation'] = True
# HyF_Parameters['Config']['ID Arbitrage'] = True
# HyF_Parameters['ESS Capacity'] = 0.0000000001
# HyF_Parameters['Config']['ID Arbitrage'] = False
# Results_Standalone= simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
# Cases.append(case_name)


# case_name = 'CF'
# HyF_Parameters['ESS Capacity'] = 10
# Results_CF= simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
# Cases.append(case_name)


case_name = 'Ideal'    # ESS not doing arbitrage on DM and SOC dump
HyF_Parameters['Config']['Ideal'] = True
HyF_Parameters['Config']['ID Participation'] = False
Results_Ideal= simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
HyF_Parameters['Config']['Ideal'] = False
Cases.append(case_name)

case_name = 'DM + ID'    # ESS arbitrage on DM + plant arbitrage on ID
HyF_Parameters['Config']['ID Arbitrage'] = True
HyF_Parameters['Config']['ID Participation'] = True
Results_DM_ID = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

case_name = 'ID'    # ESS not doing arbitrage on DM + plant arbitrage on ID
HyF_Parameters['Config']['ESS DM Participation'] = False
Results_ID = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

case_name = 'DM'     # ESS arbitrage on DM only
HyF_Parameters['Config']['ESS DM Participation'] = True
HyF_Parameters['Config']['ID Arbitrage'] = False
Results_DM = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

case_name = 'DM + SE'    # ESS arbitrge on DM and SOC dump
HyF_Parameters['Config']['SOC Dump'] = True
Results_DM_SE = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

case_name = 'SE'    # ESS not doing arbitrage on DM and SOC dump
HyF_Parameters['Config']['ESS DM Participation'] = False
Results_SE = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

case_name = 'CF'    # ESS only covers deviations on real-time
HyF_Parameters['Config']['SOC Dump'] = False
Results_none = simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time)
Cases.append(case_name)

# Stopping timer
if round((time.time() - sim_timer) / 3600) > 1:
    print(f'Global simulation elapsed time: {round((time.time() - sim_timer) / 3600)}h')
else:
    print(f'Global simulation elapsed time: {round((time.time() - sim_timer))}s')
#%% Analizing results
# Loading results .npy files
Cases_Results = {}
for case in Cases:
    Cases_Results[f'{case}'] = np.load(sim_folder + f'/{case}/Case_Results.npy', allow_pickle=True).item()
# Generating plots
plot_case_comparison(Cases, Cases_Results, 'Ben_DM_Exp_MWh','Expected benefits (€/MWh)', sim_folder)
plot_case_comparison(Cases, Cases_Results, 'Ben_DM_Real_MWh','Real benefits (€/MWh)', sim_folder)
plot_case_comparison(Cases, Cases_Results, 'ESS_deg','ESS degradation (%)', sim_folder)
plot_case_comparison(Cases, Cases_Results, 'ID_purch_rel','Energy covered with IDs (%)', sim_folder)
plot_case_comparison(Cases, Cases_Results, 'Dev_costs','Deviation costs (€)', sim_folder)
plot_case_comparison(Cases, Cases_Results, 'ESS_E_Real','Energy cycled by ESS (MWh)', sim_folder)

# Writting results
with open(sim_folder + '/Results.txt','w') as f:
    f.writelines(sim_name + ' simulation results:')
    for case in Cases_Results:
        f.writelines(f'\n - {case} case:')
        f.writelines('\n \t' + '- Accumulated expected benefits per MWh: ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'Ben_DM_Exp_MWh'),2)) + '€')
        f.writelines('\n \t' + '- Accumulated real benefits per MWh: ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'Ben_DM_Real_MWh'),2)) + '€')
        f.writelines('\n \t' + '- Accumulated ESS degradation: ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'ESS_deg'),5)) + '%')
        f.writelines('\n \t' + '- Energy cycled by ESS : ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'ESS_E_Real'),5)) + 'MWh')
        f.writelines('\n \t' + '- Mean daily power covered with ID purchases: ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'ID_purch_rel'),2)) + '%')
        f.writelines('\n \t' + '- Accumulated deviation costs: ' +
                     str(np.round(measure_accumulator(Cases_Results, case, 'Dev_costs'),2)) + '€')

