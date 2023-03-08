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
from common_funcs_v3 import *
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
# Disabling Statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
#%% Initializing parameters
# Control variables
starting_day =  '2016-01-01 00:00:00'               # First day to evaluate
ending_day =  '2020-12-31 00:00:00'                 # last day to evaluate
# Input data folders
schedules_directory = "2016_2020_nodeg"
now = datetime.datetime.now()           # Simulation time
#%% Importing price data from csv
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","Hour"], parse_dates=['Hour'], index_col="Hour")
# prices_df = prices_df.asfreq('H')
#%% BESS parameters
Batt_Enom = 50                              # [MWh] Battery nominal capacity
Batt_Cost= 37.33*Batt_Enom*1000             # [€] BESS cost
#%% Analizing results day by day by importing generated scenarios & schedules
# Launching analysis
day = starting_day
d = 0
montecarlo_analysis = {}
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    run_start = datetime.datetime.now()

    # Identifying daily results folder
    day_results_folder = "Results/"+schedules_directory+\
                         "/{}".format(pd.Timestamp(day).strftime("%Y_%m_%d"))

    # Importing daily schedule & scenarios
    Schedules_P = np.load(day_results_folder+'/Schedules_P.npy', allow_pickle=True).item()
    Schedules_SOC = np.load(day_results_folder+'/Schedules_SOC.npy', allow_pickle=True).item()
    scenarios = np.load(day_results_folder+'/scenarios.npy', allow_pickle=True).item()

    # Obtaining results with Montecarlo approaches
    # Obtaining benefits means, expected shortfalls and expected tail returns of each schedule
    Schedules_ESs = []                  # Array of daily schedules expected shortfalls
    Schedules_ETRs = []                 # Array of daily schedules ETRs
    Schedules_ben_means = []            # Array with benefits means
    for j in range(len(Schedules_P)):
        schedule_benefits = []          # Benefit of each schedule with each scenario
        for i in range(len(scenarios)): # Obtaining montecarlo of schedule benefits
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

    # Saving current day montecarlo analysis
    montecarlo_analysis['{}'.format(day)] = [Schedules_ESs, Schedules_ETRs, Schedules_ben_means]
    # Displaying daily simulation elapsed time
    run_end = datetime.datetime.now()
    run_duration = run_end - run_start
    # Printing daily results
    print("*********************************************************************************")
    print("Day: {}, Elapsed time: {}s".format(day, run_duration))
    # Changing starting day for next iteration, restarting day count when one year has been surpased
    day = pd.Timestamp(day) + pd.Timedelta('1d')
    d = d+1

elapsed_time = datetime.datetime.now() - now
print("*********************************************************************************")
print("Total elapsed time: {}s".format(elapsed_time))
print("Number of days evaluated: {}".format(d))

# Saving results
np.save('montecarlo_analysis_nodeg.npy', montecarlo_analysis)
