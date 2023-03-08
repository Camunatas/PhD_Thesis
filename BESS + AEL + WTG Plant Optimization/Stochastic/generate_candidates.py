#%% Load python libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time

#%% Load other project files
from aux_fcns import *
from RHU_model import *
from plt_fcns import *
from sim_fcns import *
#%% Get parameters
RHU_Parameters = get_RHU_parameters()
# Hydrogen price
Price_H2 = 4            # [€/kg] Hydrogen price
#%% Load dataset
Global_dataset = np.load('Dataset_10.npy', allow_pickle=True).item()
#%% Generate candidates
data_folder = 'E:\Datos RHU Stochastic/'
starting_day =  '2020-01-01'
ending_day =  '2020-01-01'
day = starting_day
gen_start = time.time()
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    daily_inputs = Global_dataset[pd.Timestamp(day).strftime("%Y-%m-%d")]
    daily_programs_dict = {}
    for windspe_scen_ID in daily_inputs['windspe_scenarios_DM'].keys():
        for price_scen_ID in daily_inputs['price_scenarios'].keys():
            program_start = time.time()
            # Extract scenarios
            windspe_scen = daily_inputs['windspe_scenarios_DM'][windspe_scen_ID]
            Pgen_scen = Siemens_SWT_30_113_curve(windspe_scen)
            price_scen = daily_inputs['price_scenarios'][price_scen_ID]
            # Generate program
            program = RHU_DM(RHU_Parameters, Pgen_scen, price_scen, Price_H2)
            gen_commitments = [a + b for a,b in zip(program['P_WTG_Grid'], program['P_BESS_Grid'])]
            con_commitments = program['P_Grid_AEL']
            # Store program results
            program_dict = {}
            program_id = [windspe_scen_ID, price_scen_ID]
            program_dict['Program ID'] = program_id
            program_dict['Program Commitments'] = gen_commitments
            program_dict['Program Purchases'] = con_commitments
            daily_programs_dict[str(program_id)] = program_dict
            program_end= time.time()
            # Display timer
            # print(f'Generated candidate {program_id},total elapsed time: {round((program_end - gen_start)/60,4)} min,')
    # Save daily candidates
    np.save(data_folder + f'Candidates/{pd.Timestamp(day).strftime("%Y-%m-%d")}.npy', daily_programs_dict)
    print(f'Generated candidates for {pd.Timestamp(day).strftime("%Y-%m-%d")}')
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')
gen_end = time.time()
print(f'Candidates generation completed, elapsed time: {round((gen_end - gen_start) / 3600, 4)} h,')
#%% Plot scenarios
# for key in daily_inputs['price_scenarios']:
#     figure = plt.figure('Price scenarios')
#     plt.plot(daily_inputs['price_scenarios'][key])
#     plt.xlabel('Hour')
#     plt.ylabel('Price (€/MWh)')
# plt.grid()
# plt.show()
#
# for key in daily_inputs['windspe_scenarios_DM']:
#     figure = plt.figure('Wind speed scenarios')
#     plt.plot(daily_inputs['windspe_scenarios_DM'][key])
#     plt.xlabel('Hour')
#     plt.ylabel('Wind speed (m/s)')
# plt.grid()
# plt.show()