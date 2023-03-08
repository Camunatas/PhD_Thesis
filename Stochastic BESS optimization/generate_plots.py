#%% Importing libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
import datetime
from pyomo.environ import *
from pyomo.opt import SolverFactory
import statistics
import seaborn as sns
import os
import warnings
from common_funcs_v3 import scen_eval, energy
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling Statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

#%% General parameters
day = '2016-01-01 00:00:00'             # Day to plot
now = datetime.datetime.now()           # Simulation time
# Input data folders
analysis_folder = "Analysis/" + "Brute benefits & K_1"
schedules_directory = "2016_2020"
direct_pred_directory = "2016_2020"
# Creating plots directory
plots_directory = "Plots/" + now.strftime("%Y_%m_%d_%H_%M")
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)
# Auxiliary variables
figurecount = 0                         # Figure counter
dates_label = []                        # X axis dates label
for i in range(24):                     # Filling X axis dates label
    dates_label.append('{}:00'.format(i))
now = datetime.datetime.now()           # Simulation time
#%% BESS parameters
Batt_Enom = 50                              # [MWh] Battery nominal capacity
Batt_Pnom = Batt_Enom/4                     # [MW] Battery nominal power
Batt_ChEff = 0.95                           # BESS charging efficiency
Batt_Cost= 37.33*Batt_Enom*1000             # [€] BESS cost
Batt_SOC_init = 0                           # Initial SOC
BESS_EOL = 0.8                              # [%/100] Capacity of BESS EOL
#%% Loading data
print('Loading data')
# Prices
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","Hour"], parse_dates=['Hour'], index_col="Hour")
# prices_df = prices_df.asfreq('H')
# Montecarlo schedule & scenarios for the current day
day_results_folder = "Results/"+schedules_directory+\
                     "/{}".format(pd.Timestamp(day).strftime("%Y_%m_%d"))
Schedules_P = np.load(day_results_folder+'/Schedules_P.npy', allow_pickle=True).item()
Schedules_SOC = np.load(day_results_folder+'/Schedules_SOC.npy', allow_pickle=True).item()
scenarios = np.load(day_results_folder+'/scenarios.npy', allow_pickle=True).item()
# Direct forecasts & schedules
direct_forecasts = np.load('Direct predictions/'+direct_pred_directory+
                           '/direct_forecasts.npy', allow_pickle=True)
direct_Ps = np.load('Direct predictions/'+direct_pred_directory+
                    '/direct_Ps.npy', allow_pickle=True)
direct_SOCs = np.load('Direct predictions/'+direct_pred_directory+
                      '/direct_SOCs.npy', allow_pickle=True)
# Analysis
Results_ES = np.load(analysis_folder+'/Results_ES.npy')
Results_mean = np.load(analysis_folder+'/Results_mean.npy')
Results_ETR = np.load(analysis_folder+'/Results_ETR.npy')
Results_direct = np.load(analysis_folder+'/Results_direct.npy')
Energy_ES = np.load(analysis_folder+'/Energy_ES.npy')
Energy_direct = np.load(analysis_folder+'/Energy_direct.npy')
Energy_mean = np.load(analysis_folder+'/Energy_mean.npy')
Energy_ETR = np.load(analysis_folder+'/Energy_ETR.npy')
Energy_ideal = np.load(analysis_folder+'/Energy_ideal.npy')
#%% Obtaining & saving scenarios
print('Obtaining & saving scenarios')
# figurecount = 0     # Figure window counter
# X axis dates label
dates_label = []
for i in range(24):
    dates_label.append('{}:00'.format(i))

for i in range(len(scenarios)):
    plt.figure(figurecount)
    plt.plot(scenarios["{}".format(i)], c=np.random.rand(3), alpha=0.1)
    plt.plot(scenarios["{}".format(i)], c=np.random.rand(3))
    plt.xlabel("Hour")
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    plt.ylabel("Price (€/MWh)")
plt.grid()
# plt.savefig(plots_directory+'/scenarios.png')
plt.savefig(plots_directory+'/scenarios.svg')
figurecount = figurecount + 1
#%% Obtaining & saving power cluster
Charges_prev = []               # Array of arrays for charge operations
Discharges_prev = []            # Array of arrays for discharge operations
Zeros_prev = []                     # Array of arrays for zero-charge operations
for key in Schedules_P:
    h = 0
    for i in Schedules_P['{}'.format(key)]:
        if i > 0:
            Charges_prev.append([i, h])
        if i < 0:
            Discharges_prev.append([i, h])
        h = h + 1

Charges = []                    # Array of charge operations with its weights
Discharges = []                 # Array of discchargue operations with its weights
Zeros = []                          # Array of zero-charge operations with its weights
Organized_operations = []       # Array of operations already organized

for key in Schedules_P:
    h = 0
    for i in Schedules_P['{}'.format(key)]:
        if i > 0:
            Charges_prev.append([i, h])
        if i < 0:
            Discharges_prev.append([i, h])
        if i == 0:
            Zeros_prev.append([i, h])
        h = h + 1
# Grouping charge operations
for i in Charges_prev:
    count = Charges_prev.count(i)
    if Organized_operations.count(i) == 0:
        Organized_operations.append(i)
        h = i.copy()
        h.append(count)
        Charges.append(h)
# Grouping discharge operations
for i in Discharges_prev:
    count = Discharges_prev.count(i)
    if Organized_operations.count(i) == 0:
        Organized_operations.append(i)
        h = i.copy()
        h.append(count)
        Discharges.append(h)
# Grouping zero energy operations
for i in Zeros_prev:
    count = Zeros_prev.count(i)
    if Organized_operations.count(i) == 0:
        Organized_operations.append(i)
        h = i.copy()
        h.append(count)
        Zeros.append(h)
# Plotting powers cluster
plt.figure(figurecount)
for i in Charges:
    plt.scatter(i[1],i[0], s=i[2],c=i[2], cmap='plasma')
for i in Discharges:
    plt.scatter(i[1],i[0], s=i[2],c=i[2], cmap='plasma')
for i in Zeros:
    plt.scatter(i[1], i[0], s=i[2], c=i[2], cmap='plasma')
plt.xlabel("Hour")
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
plt.ylabel("Power (MW)")
plt.grid()
plt.show()
plt.savefig(plots_directory+'/powers_cluster.png')
plt.savefig(plots_directory+'/powers_cluster.eps')
figurecount = figurecount + 1
#%% Obtaining & saving schedules histograms
print('Obtaining & saving schedules histograms')
# Initializing daily variables
Schedules_ESs = []                  # Array of daily schedules expected shortfalls
Schedules_ETRs = []                 # Array of daily schedules ETRs
Schedules_ben_means = []            # Array with benefits means
schedule_benefits = []              # Benefit of schedule with each scenario
for j in range(len(Schedules_P)):
    schedule_benefits = []  # Benefit of each schedule with each scenario
    for i in range(len(scenarios)):  # Obtaining montecarlo of schedule benefits
        schedule_benefit, schedule_deg = scen_eval(Schedules_P['{}'.format(j)], scenarios['{}'.format(i)],
                                                   Schedules_SOC['{}'.format(j)], Batt_Cost, Batt_Enom)
        schedule_benefits.append(schedule_benefit)
    # Skipping empty schedules
    if sum(schedule_benefits) == 0:
        Schedules_ESs.append(0)
        Schedules_ETRs.append(0)
        Schedules_ben_means.append(0)
    else:
        # Obtaining and storing mean of benefits
        ben_mean = statistics.mean(schedule_benefits)
        Schedules_ben_means.append(ben_mean)
        # Obtaining and storing Expected Shortfall
        schedule_benefits = np.array(schedule_benefits)
        var_95 = np.percentile(schedule_benefits, 5)
        Schedule_ES = schedule_benefits[schedule_benefits <= var_95].mean()
        Schedules_ESs.append(Schedule_ES)
        # Obtaining and storing
        var_5 = np.percentile(schedule_benefits, 95)
        Schedule_ETR = schedule_benefits[schedule_benefits >= var_5].mean()
        Schedules_ETRs.append(Schedule_ETR)
# Means of benefits
plt.figure(figurecount)
plt.hist(Schedules_ben_means, edgecolor='black', align='mid', bins=1000)
plt.xlabel("Benefits (€)")
plt.ylabel("Scenarios")
plt.grid()
plt.savefig(plots_directory+'/Schedules_ben_means.png')
plt.savefig(plots_directory+'/Schedules_ben_means.svg')
figurecount = figurecount + 1
# ETRs
plt.figure(figurecount)
plt.hist(Schedules_ETRs, edgecolor='black', align='mid', bins=1000)
plt.xlabel("Expected Tail Return (€)")
plt.ylabel("Scenarios")
plt.grid()
plt.savefig(plots_directory+'/Schedules_ETRs.png')
plt.savefig(plots_directory+'/Schedules_ETRs.svg')
figurecount = figurecount + 1
# ESs
plt.figure(figurecount)
plt.hist(Schedules_ESs, edgecolor='black', align='mid', bins=1000)
plt.xlabel("Expected Shortfall (€)")
plt.ylabel("Scenarios")
plt.grid()
plt.savefig(plots_directory+'/Schedules_ESs.png')
plt.savefig(plots_directory+'/Schedules_ESs.svg')
figurecount = figurecount + 1
#%% Obtaining & saving distributions
print('Obtaining & saving kdes with each criteria')
# Evaluating best scenario with each Montecarlo approach
y_ESs = []                          # Mean and expected shortfall relations for each schedule
y_ETRs = []                         # Mean and expected tail return relations for each schedule
for i in range(len(Schedules_ben_means)):
    y_ES = Schedules_ben_means[i] + 0.2*Schedules_ESs[i]
    y_ETR = Schedules_ben_means[i] + 0.2*Schedules_ETRs[i]
    if Schedules_ben_means[i] == 0:
        y_ESs.append(0)
        y_ETRs.append(0)
    else:
        y_ESs.append(y_ES)
        y_ETRs.append(y_ETR)
# Obtaining results of best schedule using ES criteria
best_schedule_ES = np.argmax(y_ESs)
print("With ES: {}#".format(best_schedule_ES))
schedule_benefits_ES = []  # Benefits of schedule selected by ES criteria
for i in range(len(scenarios)):  # Obtaining montecarlo of schedule benefits
    schedule_benefit, schedule_deg = scen_eval(Schedules_P['{}'.format(best_schedule_ES)],
                                               scenarios['{}'.format(i)],
                                               Schedules_SOC['{}'.format(best_schedule_ES)],
                                               Batt_Cost, Batt_Enom)
    schedule_benefits_ES.append(schedule_benefit)
# Obtaining results of best schedule using ETR criteria
best_schedule_ETR = np.argmax(y_ETRs)
print("With ETR: {}#".format(best_schedule_ETR))
schedule_benefits_ETR = []  # Benefits of schedule selected by ES criteria
for i in range(len(scenarios)):  # Obtaining montecarlo of schedule benefits
    schedule_benefit, schedule_deg = scen_eval(Schedules_P['{}'.format(best_schedule_ETR)],
                                               scenarios['{}'.format(i)],
                                               Schedules_SOC['{}'.format(best_schedule_ETR)],
                                               Batt_Cost, Batt_Enom)
    schedule_benefits_ETR.append(schedule_benefit)
# Obtaining results of best schedule using only the mean
best_schedule_mean = np.argmax(Schedules_ben_means)
print("With mean: {}#".format(best_schedule_mean))
schedule_benefits_mean = []  # Benefits of schedule selected by ES criteria
for i in range(len(scenarios)):  # Obtaining montecarlo of schedule benefits
    schedule_benefit, schedule_deg = scen_eval(Schedules_P['{}'.format(best_schedule_mean)],
                                               scenarios['{}'.format(i)],
                                               Schedules_SOC['{}'.format(best_schedule_mean)],
                                               Batt_Cost, Batt_Enom)
    schedule_benefits_mean.append(schedule_benefit)
# Plotting & saving KDEs of each
plt.figure(figurecount)
sns.kdeplot(schedule_benefits_ES, label='ES')
sns.kdeplot(schedule_benefits_ETR, label='ETR')
sns.kdeplot(schedule_benefits_mean, label='mean')
plt.xlabel('Benefits (€)')
plt.grid()
plt.legend()
plt.savefig(plots_directory+'/methods_KDEs_comparison.png')
plt.savefig(plots_directory+'/methods_KDEs_comparison.svg')
plt.show()
figurecount = figurecount + 1

#%% Plotting KDE& histogram of single schedule
schedule_id = 256
schedule_benefits = []
for i in range(len(scenarios)):  # Obtaining montecarlo of schedule benefits
    schedule_benefit, schedule_deg = scen_eval(Schedules_P['{}'.format(best_schedule_ES)],
                                               scenarios['{}'.format(i)],
                                               Schedules_SOC['{}'.format(best_schedule_ES)],
                                               Batt_Cost, Batt_Enom)
    schedule_benefits.append(schedule_benefit)


# Getting percentiles
avocado = pd.DataFrame(schedule_benefits)
quant_5, quant_95 = avocado.quantile(0.05), avocado.quantile(0.95)
# Plot
plt.figure(figurecount)
sns.histplot(data=schedule_benefits, bins=1000, kde=True)
plt.vlines(quant_5,0,1,color='red')
plt.text(quant_5-100, 1, 'Percentile 5%')
plt.vlines(quant_95,0,1,color='red')
plt.text(quant_95-100, 1, 'Percentile 95%')
plt.xlabel('Benefits(€)')
plt.savefig(plots_directory+'/KDE_example.png')
plt.savefig(plots_directory+'/KDE_example.eps')
plt.show()

#%% Plotting single program results
# Considering degradation
direct_price = direct_forecasts[750]
Batt_Cost= 37.33*Batt_Enom*1000                   # [€] BESS cost
Power_deg, SOC_deg = arbitrage(Batt_SOC_init, direct_price, Batt_Enom, Batt_Pnom, Batt_ChEff,
                               Batt_Cost)
# Plotting results
fig = plt.figure("Program results considering degradation")  # Creating the figure
# Energy price
price_plot = fig.add_subplot(3, 1, 1)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([min(direct_price) * 0.9, max(direct_price) * 1.1])  # X axis limits
# Inyecting the data
plt.bar(ticks_x, direct_price, align='edge', width=1, edgecolor='black', color='r')
# Adding labels
plt.ylabel('Price (€/MWh)')
plt.grid()
# SOC
SOC_plot = fig.add_subplot(3, 1, 2)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([0, 110])  # X axis limits
# Inyecting the data
plt.plot(SOC_deg, 'r', label='Predicted price')
# Adding labels
plt.ylabel('SOC (%)')
plt.grid()

# Power
P_output_plot = fig.add_subplot(3, 1, 3)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)  # Vertical grid spacing
minor_ticks_y = np.arange(-Batt_Pnom * 1.5, Batt_Pnom * 1.5, 2.5)  # Thin horizontal grid  spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
P_output_plot.set_yticks(minor_ticks_y, minor=True)
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])
axes.set_ylim([-ceil(Batt_Pnom), ceil(Batt_Pnom)])  # X axis limits
# Inyecting the data
x = np.arange(24)
plt.bar(x, Power_deg, width=1, color='r', edgecolor='black', align='edge', label='Predicted price')
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')
plt.grid()

# Launching & saving the plot
plt.show()
plt.savefig(plots_directory+'/arb_deg.png')
plt.savefig(plots_directory+'/arb_deg.eps')


# Not considering degradation
direct_price = direct_forecasts[750]
Batt_Cost= 0.00000000001
Power_deg, SOC_deg = arbitrage(Batt_SOC_init, direct_price, Batt_Enom, Batt_Pnom, Batt_ChEff,
                               Batt_Cost)
# Plotting results
fig = plt.figure("Program results not considering degradation")  # Creating the figure
# Energy price
price_plot = fig.add_subplot(3, 1, 1)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([min(direct_price) * 0.9, max(direct_price) * 1.1])  # X axis limits
# Inyecting the data
plt.bar(ticks_x, direct_price, align='edge', width=1, edgecolor='black', color='r')
# Adding labels
plt.ylabel('Price (€/MWh)')
plt.grid()
# SOC
SOC_plot = fig.add_subplot(3, 1, 2)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([0, 110])  # X axis limits
# Inyecting the data
plt.plot(SOC_deg, 'r', label='Predicted price')
# Adding labels
plt.ylabel('SOC (%)')
plt.grid()

# Power
P_output_plot = fig.add_subplot(3, 1, 3)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 25, 1)  # Vertical grid spacing
minor_ticks_y = np.arange(-Batt_Pnom * 1.5, Batt_Pnom * 1.5, 2.5)  # Thin horizontal grid  spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
P_output_plot.set_yticks(minor_ticks_y, minor=True)
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])
axes.set_ylim([-ceil(Batt_Pnom), ceil(Batt_Pnom)])  # X axis limits
# Inyecting the data
x = np.arange(24)
plt.bar(x, Power_deg, width=1, color='r', edgecolor='black', align='edge', label='Predicted price')
# Adding labels
plt.xlabel('Time (Hours)')
plt.ylabel('Power (MW)')
plt.grid()

# Launching & saving the plot
plt.show()
plt.savefig(plots_directory+'/arb_nodeg.png')
plt.savefig(plots_directory+'/arb_nodeg.eps')

