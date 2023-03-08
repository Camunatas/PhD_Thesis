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


#%% Apply algorithm with various approaches
data_folder = 'E:\Datos RHU Stochastic/'
starting_day =  '2020-01-01'
ending_day =  '2020-12-31'
Results = {}
day = starting_day
# measure = 'Electricity Market'
# measure = 'Hydrogen Market'
measure = 'Deviation Costs'
# measure = 'AEL Purchases'
# measure = 'Cash Flow'
results_npy_name = 'final_results_ben_devs'
algorithm_start = time.time()
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    Daily_Results = {}
    # Load samples
    Programs = np.load(data_folder + f'Samples/ev_{day_str}.npy', allow_pickle=True).item()
    Bootstrap_samples= np.load(data_folder + f'Samples/boot_{day_str}.npy', allow_pickle=True).item()
    Benchmarks= np.load(data_folder + f'Benchmarks/{day_str}.npy', allow_pickle=True).item()
    # Apply stochastic approaches
    boot_risk_neutral = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Bootstrapping')
    boot_risk_averse = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Bootstrapping')
    mc_risk_neutral = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Monte Carlo')
    mc_risk_averse = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Monte Carlo')
    # Obtain the best candidate
    best_program_ID = find_best_program(Programs)
    best_program_results = Programs[best_program_ID]['Real results'][measure]
    # Load benchmarks
    direct_results = Benchmarks['Direct approach results'][measure]
    ideal_results = Benchmarks['Ideal results'][measure]
    # Compare algorithm results
    boot_risk_neutral_results = Programs[boot_risk_neutral]['Real results'][measure]
    boot_risk_averse_results = Programs[boot_risk_averse]['Real results'][measure]
    mc_risk_neutral_results = Programs[mc_risk_neutral]['Real results'][measure]
    mc_risk_averse_results = Programs[mc_risk_averse]['Real results'][measure]
    print(f'Results for {day_str}')
    # print(f'Selected with risk-neutral bootstrap: {boot_risk_neutral}, '
    #       f'results: {np.round(boot_risk_neutral_results,2)}€')
    # print(f'Selected with risk-averse bootstrap: {boot_risk_averse}, '
    #       f'results: {np.round(boot_risk_averse_results,2)}€')
    # print(f'Selected with risk-neutral Monte Carlo: {mc_risk_neutral}, '
    #       f'results: {np.round(mc_risk_neutral_results,2)}€')
    # print(f'Selected with risk-averse Monte Carlo: {mc_risk_averse}, '
    #       f'results: {np.round(mc_risk_averse_results,2)}€')
    # print(f'Best program: {best_program_ID}, results: {np.round(best_program_ID_results,2)}€')
    # print(f'Results with direct approach: {np.round(direct_results,2)}€')
    # print(f'Ideal results: {np.round(ideal_results,2)}€')
    # Storing results
    Daily_Results['boot_risk_neutral_results'] = boot_risk_neutral_results
    Daily_Results['boot_risk_averse_results'] = boot_risk_averse_results
    Daily_Results['mc_risk_neutral_results'] = mc_risk_neutral_results
    Daily_Results['mc_risk_averse_results'] = mc_risk_averse_results
    Daily_Results['direct_results'] = direct_results
    Daily_Results['ideal_results'] = ideal_results
    Daily_Results['best_candidate_results'] = best_program_results
    # Daily_Results['best_candidate_results'] = best_program_ID_results[measure]
    Results[f'{day_str}'] = Daily_Results
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')

# Generate results lists
Boot_risk_neutral_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_risk_neutral_results'))
boot_risk_averse_results_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_risk_averse_results'))
mc_risk_neutral_results_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_risk_neutral_results'))
mc_risk_averse_results_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_risk_averse_results'))
direct_results_list_acc = list_accumulator(results_case_list_builder(Results, 'direct_results'))
ideal_results_list_acc = list_accumulator(results_case_list_builder(Results, 'ideal_results'))
best_results_list_acc = list_accumulator(results_case_list_builder(Results, 'best_candidate_results'))

# Display:
print(f'Simulation finished, elapsed time: {np.round((time.time()-algorithm_start)/60,2)} min')
print(f'Results from {starting_day} to {ending_day}:')
print(f'\t Risk neutral Bootstrap: {np.round(Boot_risk_neutral_list_acc[-1],2)} €')
print(f'\t Risk averse Bootstrap: {np.round(boot_risk_averse_results_list_acc[-1],2)} €')
print(f'\t Risk neutral Monte Carlo: {np.round(mc_risk_neutral_results_list_acc[-1],2)} €')
print(f'\t Risk averse Monte Carlo: {np.round(mc_risk_averse_results_list_acc[-1],2)} €')
print(f'\t Direct approach: {np.round(direct_results_list_acc[-1],2)} €')
print(f'\t Ideal approach: {np.round(ideal_results_list_acc[-1],2)} €')
print(f'\t Best candidate approach: {np.round(best_results_list_acc[-1],2)} €')

# Plot
figure = plt.figure('Results')
# plt.plot(Boot_risk_neutral_list_acc, label='RN Boot')
plt.plot(boot_risk_averse_results_list_acc, label='Bootstrap')
# plt.plot(mc_risk_neutral_results_list_acc, label='RN MC')
plt.plot(mc_risk_averse_results_list_acc, label='Monte Carlo')
plt.plot(direct_results_list_acc, label='Deterministic')
plt.plot(ideal_results_list_acc, label='Ideal')
# plt.plot(best_results_list_acc, label='Best candidate')
plt.legend()
plt.ylabel(f'{measure} (€)')
plt.xlabel('Days')
plt.grid()
plt.show()
plt.savefig(f'plots/Results/{measure} {starting_day} to {ending_day}.png')

# Saving results
final_results = {}
final_results['Bootstrap'] = boot_risk_averse_results_list_acc
final_results['Monte Carlo'] = mc_risk_averse_results_list_acc
final_results['Deterministic'] = direct_results_list_acc
final_results['Ideal'] = ideal_results_list_acc
np.save(f'{results_npy_name}.npy', final_results)
#%% Robustness parameter sweep
# data_folder = 'E:\Datos RHU Stochastic/'
# starting_day =  '2020-01-01'
# ending_day =  '2020-09-20'
# Results = {}
# day = starting_day
# # measure = 'Electricity Market'
# # measure = 'Hydrogen Market'
# # measure = 'Deviation Costs'
# # measure = 'AEL Purchases'
# measure = 'Cash Flow'
# algorithm_start = time.time()
# while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
#     day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
#     Daily_Results = {}
#     # Load samples
#     Programs = np.load(data_folder + f'Samples/ev_{day_str}.npy', allow_pickle=True).item()
#     Bootstrap_samples= np.load(data_folder + f'Samples/boot_{day_str}.npy', allow_pickle=True).item()
#     Benchmarks= np.load(data_folder + f'Benchmarks/{day_str}.npy', allow_pickle=True).item()
#     # Apply stochastic approaches
#     boot_025 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Bootstrapping', param=0.25)
#     boot_05 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Bootstrapping', param=0.5)
#     boot_075 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Bootstrapping', param=0.75)
#     boot_1 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk-averse Bootstrapping', param=1)
#     mc_025 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Monte Carlo', param=0.25)
#     mc_05 = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Monte Carlo', param=0.25)
#     mc_075  = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Monte Carlo', param=0.75)
#     mc_1  = apply_stochastic_algorithm(Programs, Bootstrap_samples, 'Risk Neutral Monte Carlo', param=-1)
#
#     # Load benchmarks
#     direct_results = Benchmarks['Direct approach results'][measure]
#     ideal_results = Benchmarks['Ideal results'][measure]
#     # Compare algorithm results
#     boot_025_results = Programs[boot_025]['Real results'][measure]
#     boot_05_results = Programs[boot_05]['Real results'][measure]
#     boot_075_results = Programs[boot_075]['Real results'][measure]
#     boot_1_results = Programs[boot_1]['Real results'][measure]
#     mc_025_results = Programs[mc_025]['Real results'][measure]
#     mc_05_results = Programs[mc_05]['Real results'][measure]
#     mc_075_results = Programs[mc_075]['Real results'][measure]
#     mc_1_results = Programs[mc_1]['Real results'][measure]
#     # best_program_ID, best_program_ID_results = find_best_program(Programs)
#     print(f'Results for {day_str}')
#     # print(f'Selected with risk-neutral bootstrap: {boot_risk_neutral}, '
#     #       f'results: {np.round(boot_risk_neutral_results,2)}€')
#     # print(f'Selected with risk-averse bootstrap: {boot_risk_averse}, '
#     #       f'results: {np.round(boot_risk_averse_results,2)}€')
#     # print(f'Selected with risk-neutral Monte Carlo: {mc_risk_neutral}, '
#     #       f'results: {np.round(mc_risk_neutral_results,2)}€')
#     # print(f'Selected with risk-averse Monte Carlo: {mc_risk_averse}, '
#     #       f'results: {np.round(mc_risk_averse_results,2)}€')
#     # print(f'Best program: {best_program_ID}, results: {np.round(best_program_ID_results,2)}€')
#     # print(f'Results with direct approach: {np.round(direct_results,2)}€')
#     # print(f'Ideal results: {np.round(ideal_results,2)}€')
#     # Storing results
#     Daily_Results['boot_025_results'] = boot_025_results
#     Daily_Results['boot_05_results'] = boot_05_results
#     Daily_Results['boot_075_results'] = boot_075_results
#     Daily_Results['boot_1_results'] = boot_1_results
#     Daily_Results['mc_025_results'] = mc_025_results
#     Daily_Results['mc_05_results'] = mc_05_results
#     Daily_Results['mc_075_results'] = mc_075_results
#     Daily_Results['mc_1_results'] = mc_1_results
#     Daily_Results['Ideal_results'] = ideal_results
#     Daily_Results['Direct_results'] = direct_results
#     Results[f'{day_str}'] = Daily_Results
#     # Updating day
#     day = pd.Timestamp(day) + pd.Timedelta('1d')
#
# # Generate results lists
# boot_025_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_025_results'))
# boot_05_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_05_results'))
# boot_075_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_075_results'))
# boot_1_list_acc = list_accumulator(results_case_list_builder(Results, 'boot_1_results'))
# mc_025_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_025_results'))
# mc_05_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_05_results'))
# mc_075_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_075_results'))
# mc_1_results_list_acc = list_accumulator(results_case_list_builder(Results, 'mc_1_results'))
# Ideal_results_list_acc = list_accumulator(results_case_list_builder(Results, 'Ideal_results'))
# Direct_results_list_acc = list_accumulator(results_case_list_builder(Results, 'Direct_results'))
#
# # Display:
# print(f'Simulation finished, elapsed time: {np.round((time.time()-algorithm_start)/60,2)} min')
# print(f'Results from {starting_day} to {ending_day}:')
# print(f'\t Bootstrap with 0.25: {np.round(boot_025_list_acc[-1],2)} €')
# print(f'\t Bootstrap with 0.5: {np.round(boot_05_list_acc[-1],2)} €')
# print(f'\t Bootstrap with 0.75: {np.round(boot_075_list_acc[-1],2)} €')
# print(f'\t Bootstrap with 1: {np.round(boot_1_list_acc[-1],2)} €')
# print(f'\t Monte Carlo with 0.25: {np.round(mc_025_list_acc[-1],2)} €')
# print(f'\t Monte Carlo with 0.5: {np.round(mc_05_list_acc[-1],2)} €')
# print(f'\t Monte Carlo with 0.75: {np.round(mc_075_list_acc[-1],2)} €')
# print(f'\t Monte Carlo with 1: {np.round(mc_1_results_list_acc[-1],2)} €')
# print(f'\t Ideal: {np.round(Ideal_results_list_acc[-1],2)} €')
# print(f'\t Direct: {np.round(Direct_results_list_acc[-1],2)} €')
#
# # Plot
# figure = plt.figure('Results')
# plt.plot(boot_025_list_acc, label='Boot 0.25')
# plt.plot(boot_05_list_acc, label='Boot 0.5')
# plt.plot(boot_075_list_acc, label='Boot 0.75')
# plt.plot(boot_1_list_acc, label='Boot 1')
# # plt.plot(mc_025_list_acc, label='MC 0.25')
# # plt.plot(mc_05_list_acc, label='MC 0.5')
# # plt.plot(mc_075_list_acc, label='MC 0.75')
# # plt.plot(mc_1_results_list_acc, label='MC 1')
# plt.plot(Direct_results_list_acc, label='Direct')
# plt.plot(Ideal_results_list_acc, label='Ideal')
# plt.legend()
# plt.ylabel(f'{measure} (€)')
# plt.grid()
# plt.show()
# plt.savefig(f'plots/Results/{measure} {starting_day} to {ending_day}.png')
