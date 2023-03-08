#%% Load python libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import concurrent.futures
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
#%% Evaluate programs
data_folder = 'E:\Datos RHU Stochastic/'
starting_day =  '2020-11-10'
ending_day =  '2020-12-31'
day = starting_day
gen_start = time.time()
if __name__ == '__main__':
    while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
        daily_inputs = Global_dataset[pd.Timestamp(day).strftime("%Y-%m-%d")]
        daily_programs_dict = np.load(data_folder + f'Candidates/{pd.Timestamp(day).strftime("%Y-%m-%d")}.npy', allow_pickle=True).item()
        results = []
        daily_evs_dict = {}
        # Paralelized operation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for program_id in daily_programs_dict.keys():
                results.append(executor.submit(evaluate_candidates, program_id,
                                               daily_programs_dict, daily_inputs, RHU_Parameters, Price_H2))
        for r in results:
            ev_program_dict = r.result()
            daily_evs_dict[str(ev_program_dict['Program ID'])] = ev_program_dict
        # Normal operation
        # for program_id in daily_programs_dict.keys():
        #     ev_program_dict = evaluate_candidates(program_id,daily_programs_dict, daily_inputs,
        #                                           RHU_Parameters, Price_H2)
        #     daily_evs_dict[str(ev_program_dict['Program ID'])] = ev_program_dict
        gen_end = time.time()
        print(f'Candidates samples generated for {pd.Timestamp(day).strftime("%Y-%m-%d")} '
              f', elapsed time: {round((gen_end - gen_start)/3600,4)} h,')
        # Save samples
        np.save(data_folder + f'Samples/ev_{pd.Timestamp(day).strftime("%Y-%m-%d")}.npy', daily_evs_dict)
        # Updating day
        day = pd.Timestamp(day) + pd.Timedelta('1d')

#%% DISABLED: Plot scenarios
# price_scenarios = []
# wind_scenarios = []
# for key in Global_dataset['2021-03-06']['price_scenarios']:
#     price_scenarios.append(Global_dataset['2021-03-06']['price_scenarios'][key])
#     wind_scenarios.append(Global_dataset['2021-03-06']['windspe_scenarios_DM'][key])
# real_wind = Global_dataset['2021-03-06']['windspe_real']
# real_prices = Global_dataset['2021-03-06']['price_real']
#
# figure = plt.figure('Plot scenarios')
# for scenario in wind_scenarios:
#     plt.plot(scenario)
# plt.plot(real_wind, '--', label='Real' )
# plt.legend()
# plt.xlabel('Hour')
# plt.ylabel('Wind speed (m/s)')
# plt.grid()
# plt.show()

#%% DISABLED: Plot histograms of samples
# Analyze programs
# i = 0
# for key in Programs:
#     Program = Programs[key]
#     program_benefits_scens = []
#     for j in range(len(Program['Costs samples'])):
#         program_benefits_scens.append(Program['Costs samples'][j]['Total Cost'])
#     fig = plt.figure(f'Scensbens {i}')
#     plt.hist(program_benefits_scens, bins=100)
#     plt.xlabel('Costs (€)')
#     plt.ylabel('Occurence')
#     plt.savefig(f'Plots/Costs Samples/costs_samples_{i}.png')
#     plt.close()
#     program_ben_real = Program['Real results']['Cash Flow']
#     Bens_reals.append(program_ben_real)
#     i = i+1


