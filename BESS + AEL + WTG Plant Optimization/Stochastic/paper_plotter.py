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
#%% Force tight layout
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True

# Disabling showing figures
plt.ioff()
#%% Set day & hourly ticks
day =  '2020-03-25'
program_id = "['8', '2']"
ticks_x = np.arange(0, len(hourly_xticks(0)), 1)
final_results_CF = np.load('final_results_CF.npy', allow_pickle=True).item()
final_results_ben_el = np.load('final_results_ben_el.npy', allow_pickle=True).item()
final_results_ben_h2 = np.load('final_results_ben_h2.npy', allow_pickle=True).item()
final_results_ben_devs = np.load('final_results_ben_devs.npy', allow_pickle=True).item()
#%% Load required data
RHU_Parameters = get_RHU_parameters()
data_folder = 'E:\Datos RHU Stochastic/'
Dataset = np.load('Dataset_10.npy', allow_pickle=True).item()[day]
Programs = np.load(data_folder + f'Samples/ev_{day}.npy', allow_pickle=True).item()
Bootstrap_samples= np.load(data_folder + f'Samples/boot_{day}.npy', allow_pickle=True).item()
Benchmarks= np.load(data_folder + f'Benchmarks/{day}.npy', allow_pickle=True).item()
bootrstap_samples, mc_samples = get_samples(program_id, Programs, Bootstrap_samples)
#%% Plot deterministic forecast for single day
# Extract data from dataset
windspe_real = Dataset['windspe_real']
windspe_pred_DM = Dataset['windspe_pred_DM']
price_real = Dataset['price_real']
price_pred = Dataset['price_pred']
# Generate figure
fig = plt.figure('Deterministic forecasts')
price_pred_subplot = fig.add_subplot(2, 1, 1)
plt.xticks(np.arange(0, len(hourly_xticks(0)), 1), hourly_xticks(0), rotation=45)
plt.plot(price_pred, label='Predicted')
plt.plot(price_real, label='Real')
plt.ylabel('Price (€/MWh)')
plt.grid()
plt.legend()
windspe_pred_subplot = fig.add_subplot(2, 1, 2)
plt.xticks(np.arange(0, len(hourly_xticks(0)), 1), hourly_xticks(0), rotation=45)
plt.plot(windspe_pred_DM, label='Predicted')
plt.plot(windspe_real, label='Real')
plt.ylabel('Wind speed (m/s)')
plt.grid()
plt.legend()
plt.show()
plt.savefig('paper plots/deterministic_fore.png')
plt.savefig('paper plots/deterministic_fore.svg')
#%% Plot deterministic operation for single day
# Obtain generated power
P_gen_pred = Siemens_SWT_30_113_curve(windspe_pred_DM)
P_gen_real = Siemens_SWT_30_113_curve(windspe_real)
# Launch day-ahead optimization function
Results_DM = RHU_DM(RHU_Parameters, P_gen_pred, windspe_pred_DM, 1)
# Launch real-time optimization function
DM_commitments = [a + b for a, b in zip(Results_DM['P_WTG_Grid'], Results_DM['P_BESS_Grid'])]
Purch_commitments = Results_DM['P_Grid_AEL']
Purch_commitments_AEL = Results_DM['P_Grid_AEL']
Purch_commitments_BESS = Results_DM['P_Grid_BESS']
Results_RT = RHU_RT(RHU_Parameters, P_gen_real, price_real, DM_commitments,
                    Purch_commitments_AEL, Purch_commitments_BESS,  1)
# Plot day-ahead operation
Results = Results_DM
dates_label = []  # X axis dates label
for i in range(len(Results['P_AEL'])):  # Filling X axis dates label
    dates_label.append('{}:00'.format(i))
x = np.arange(len(Results['P_AEL']))
# Initialize RHU powers plot
fig = plt.figure(f'Day-ahead operation for {day}')
# plt.suptitle(f'Day-ahead operation for {day}')
# WTG Powers
WTG_plot = fig.add_subplot(4, 1, 1)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(x + 0.25, Results['P_WTG_Grid'], color='orange',  width=0.25, label='To grid', edgecolor='black')
plt.bar(x + 0.50, Results['P_WTG_BESS'], color='purple',  width=0.25, label='To BESS', edgecolor='black')
plt.bar(x + 0.75, Results['P_WTG_AEL'], color='cyan', width=0.25, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('WTG (MW)')
plt.grid()

# AEL Powers
AEL_plot = fig.add_subplot(4, 1, 2)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(x + 0.25, Results['P_on_AEL'], color= 'blue' ,width=0.25, label='On', edgecolor='black')
plt.bar(x + 0.75, Results['P_start_AEL'], color='red', width=0.25, label='Start', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AEL (MW)')
plt.grid()

# BESS Powers
BESS_plot = fig.add_subplot(4, 1, 3)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(ticks_x + 0.19, [-a for a in Results['P_Grid_BESS']], color='orange',  width=0.2, edgecolor='black')
plt.bar(ticks_x + 0.39, [-a for a in Results['P_WTG_BESS']], color='green',  width=0.2, label='From WTG', edgecolor='black')
plt.bar(ticks_x + 0.59, Results['P_BESS_Grid'], color='orange',  width=0.2, label='To/from grid', edgecolor='black')
plt.bar(ticks_x + 0.79, Results['P_BESS_AEL'], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('BESS (MW)')
plt.grid()

# Grid Powers
Grid_plot = fig.add_subplot(4, 1, 4)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), dates_label, rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(ticks_x + 0.19, Results['P_WTG_Grid'], color='green', width=0.2, label='From WTG', edgecolor='black')
plt.bar(ticks_x + 0.39, Results['P_BESS_Grid'], color='purple', width=0.2, label='To/from BESS', edgecolor='black')
plt.bar(ticks_x + 0.59, [-a for a in Results['P_Grid_BESS']], color='purple',  width=0.2, edgecolor='black')
plt.bar(ticks_x + 0.79, [-a for a in Results['P_Grid_AEL']], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Grid (MW)')
plt.grid()

# Launching the plot
plt.show()
plt.ioff()
plt.savefig('paper plots/day_ahead_example.png')
plt.savefig('paper plots/day_ahead_example.svg')


# Plot real-time operation
Results = Results_RT
dates_label = []  # X axis dates label
for i in range(len(Results['P_AEL'])):  # Filling X axis dates label
    dates_label.append('{}:00'.format(i))
x = np.arange(len(Results['P_AEL']))
# Initialize RHU powers plot
fig = plt.figure(f'Real-time operation for {day}')
# plt.suptitle(f'Real-time operation for {day}')
# WTG Powers
WTG_plot = fig.add_subplot(4, 1, 1)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(x + 0.25, Results['P_WTG_Grid'], color='orange',  width=0.25, label='To grid', edgecolor='black')
plt.bar(x + 0.50, Results['P_WTG_BESS'], color='purple',  width=0.25, label='To BESS', edgecolor='black')
plt.bar(x + 0.75, Results['P_WTG_AEL'], color='cyan', width=0.25, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('WTG (MW)')
plt.grid()

# AEL Powers
AEL_plot = fig.add_subplot(4, 1, 2)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(x + 0.25, Results['P_on_AEL'], color= 'blue' ,width=0.25, label='On', edgecolor='black')
plt.bar(x + 0.75, Results['P_start_AEL'], color='red', width=0.25, label='Start', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AEL (MW)')
plt.grid()

# BESS Powers
BESS_plot = fig.add_subplot(4, 1, 3)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(ticks_x + 0.19, [-a for a in Results['P_Grid_BESS']], color='orange',  width=0.2, edgecolor='black')
plt.bar(ticks_x + 0.39, [-a for a in Results['P_WTG_BESS']], color='green',  width=0.2, label='From WTG', edgecolor='black')
plt.bar(ticks_x + 0.59, Results['P_BESS_Grid'], color='orange',  width=0.2, label='To/from grid', edgecolor='black')
plt.bar(ticks_x + 0.79, Results['P_BESS_AEL'], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('BESS (MW)')
plt.grid()

# Grid Powers
Grid_plot = fig.add_subplot(4, 1, 4)  # Creating subplot
ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
plt.xticks(np.arange(0, len(Results['P_AEL']), 1), dates_label, rotation=45)
axes = plt.gca()
axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
axes.set_axisbelow(True)
plt.bar(ticks_x + 0.19, Results['P_WTG_Grid'], color='green', width=0.2, label='From WTG', edgecolor='black')
plt.bar(ticks_x + 0.39, Results['P_BESS_Grid'], color='purple', width=0.2, label='To/from BESS', edgecolor='black')
plt.bar(ticks_x + 0.59, [-a for a in Results['P_Grid_BESS']], color='purple',  width=0.2, edgecolor='black')
plt.bar(ticks_x + 0.79, [-a for a in Results['P_Grid_AEL']], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Grid (MW)')
plt.grid()

# Launching the plot
plt.show()
plt.ioff()
plt.savefig('paper plots/rt_example.png')
plt.savefig('paper plots/rt_example.svg')

#%% Comparing day-ahead and rt for better explanations in the paper
daily_dm_vs_real(day, P_gen_pred, price_pred, P_gen_real, price_pred, RHU_Parameters, Results_DM, Results_RT)
#%% Plot wind & price scenarios for single day
# Generate figure
fig = plt.figure('Probabilistic forecasts')
price_pred_subplot = fig.add_subplot(2, 1, 1)
plt.xticks(np.arange(0, len(hourly_xticks(0)), 1), hourly_xticks(0), rotation=45)
for key in Dataset['price_scenarios'].keys():
    plt.plot(Dataset['price_scenarios'][key])
plt.ylabel('Price (€/MWh)')
plt.grid()
windspe_pred_subplot = fig.add_subplot(2, 1, 2)
plt.xticks(np.arange(0, len(hourly_xticks(0)), 1), hourly_xticks(0), rotation=45)
for key in Dataset['windspe_scenarios_DM'].keys():
    plt.plot(Dataset['windspe_scenarios_DM'][key])
plt.ylabel('Wind speed (m/s)')
plt.grid()
plt.show()
plt.savefig('paper plots/probabilistic_fore.png')
plt.savefig('paper plots/probabilistic_fore.svg')
#%% Samples for single day
fig = plt.figure('Monte Carlo samples')
plt.hist(mc_samples, bins=len(mc_samples))
plt.ylim([0, 8])
plt.xlabel('Second stage costs (€)')
plt.show()
plt.savefig('paper plots/mc_samples.png')
plt.savefig('paper plots/mc_samples.svg')
fig = plt.figure('Bootstrap samples')
plt.hist(bootrstap_samples, bins=len(bootrstap_samples))
plt.xlabel('Second stage costs means (€)')
plt.ylim([0, 8])
plt.show()
plt.savefig('paper plots/boot_samples.png')
plt.savefig('paper plots/boot_samples.svg')
#%% Plot complete simulation results
def plot_final_results(results_dict, ylabel):
    figure = plt.figure(f'Final results: {ylabel}')
    cases = [ 'Monte Carlo', 'Deterministic', 'Bootstrap', 'Ideal']
    measure_values = [results_dict[a][-1] for a in cases]
    print(measure_values)
    x = np.arange(len(cases))
    plt.xticks(x, cases)
    plt.bar(x, measure_values, edgecolor='black', zorder=3)
    plt.ylim([580000, 750000])
    plt.grid(zorder=0)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(f'paper plots/final_results.png')
    plt.savefig(f'paper plots/final_results.svg')
plot_final_results(final_results_CF, 'Revenues (€)')
# plot_final_results(final_results_ben_el, 'Electricity Market Benefits (€)')
# plot_final_results(final_results_ben_h2, 'Hydrogen Market Benefits (€)')
# plot_final_results(final_results_ben_devs, 'Deviation Costs (€)')