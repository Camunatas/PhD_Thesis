#%% Importing libraries
import pandas as pd
from pandas.core.frame import DataFrame
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
import concurrent.futures
from common_funcs_v3 import arbitrage,scen_eval
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
# Disabling Statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
#%% Initializing parameters
# Control variables
day = pd.Timestamp('2016-01-08 00:00:00 00:00:00')      # First day to evaluate
simulation_length = 365*5+2                             # Number of days to simulate
n_scenarios = 1000                                      # Number of scenarios
# Auxiliary variables
figurecount = 0                                         # Figure counter
dates_label = []                                        # X axis dates label
for i in range(24):                                     # Filling X axis dates label
    dates_label.append('{}:00'.format(i))
now = datetime.datetime.now()                           # Simulation time
directory = '2016_2020_nodeg'
if not os.path.exists("Results/"+directory):
    os.makedirs("Results/"+directory)
#%% Importing price data from csv
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","datetime"],
                        parse_dates=['datetime'],index_col="datetime")
prices_df = prices_df.asfreq('h')
scenarios_folder = 'scenarios'

#%% BESS parameters
Batt_Enom = 50                              # [MWh] Battery nominal capacity
Batt_Pnom = Batt_Enom/4                     # [MW] Battery nominal power
Batt_ChEff = 0.95                           # BESS charging efficiency
Batt_Cost= 37.33*1000*Batt_Enom             # [€] BESS cost
Batt_SOC_init = 0                           # Initial SOC

#%% Arbitrage helper function
def arbitrage_helper(scenario_id, initial_SOC, energy_price, batt_capacity, batt_maxpower, 
              batt_efficiency, cost):
    POW, SOC = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower, 
              batt_efficiency, cost)
    return {scenario_id: POW}, {scenario_id: SOC}

#%% Launching algorithm
if __name__ == '__main__':
    ndays = 0
    for i in range(1, simulation_length+1):
        # Initializing daily variables  
        scenarios = {}                      # Dictionary where the scenarios powers are stored
        Schedules_P = {}                    # Dictionary where the schedules powers are stored
        Schedules_SOC = {}                  # Dictionary where the schedules SOCs are stored
        prices_real = []                    # Array with real prices
        run_start = datetime.datetime.now() # Initializing daily run chronometer
    
        # Generating daily results folder if enabled
        day_results_folder = "Results/"+directory+"/{}".format(day.strftime("%Y_%m_%d"))
        os.makedirs(day_results_folder)
        # Obtaining test set & real prices array
        test_start = day
        test_end = test_start + pd.Timedelta('23h')
        test_set = prices_df[test_start:test_end]
        for i in range(24):
            prices_real.append(test_set["Price"][i])
    
        # Loading scenarios
        scenarios_df = pd.read_csv(os.path.join(scenarios_folder, day.strftime("%Y_%m_%d")+'_price_scenarios.csv'))
        for j in range(n_scenarios):
            scenarios["{}".format(j)] = scenarios_df["{}".format(j)].to_list()
    
        # Generate optimum schedules for all scenarios
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(len(scenarios)):
                id = "{}".format(i)
                results.append(
                    executor.submit(arbitrage_helper, id, Batt_SOC_init, scenarios[id], Batt_Enom, Batt_Pnom,
                        Batt_ChEff, Batt_Cost)
                )
        if __name__ == '__main__':
            for r in results:
                BESS, SOC = r.result()
                Schedules_P.update(BESS)
                Schedules_SOC.update(SOC)

        # Generating & saving scenarios figure
        for i in range(len(scenarios)):
            plt.figure(figurecount)
            plt.plot(scenarios["{}".format(i)], c=np.random.rand(3))
            plt.xlabel("Hour")
            plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
            plt.ylabel("Price (€/MWh)")
        plt.grid()
        plt.savefig(day_results_folder+'/scenarios.png')
        # plt.savefig(day_results_folder+'/scenarios.eps')
        plt.close()
        figurecount = figurecount + 1

        # Saving daily results
        np.save(day_results_folder+'/Scenarios.npy', scenarios)                 # Scenarios
        np.save(day_results_folder+'/Schedules_P.npy', Schedules_P)             # Schedules powers
        np.save(day_results_folder+'/Schedules_SOC.npy', Schedules_SOC)         # Schedules SOCs

        # Displaying daily simulation elapsed time
        run_end = datetime.datetime.now()
        run_duration = run_end - run_start
        # Printing daily results
        print("*********************************************************************************")
        print("Day: {}, Elapsed time: {}s".format(day, run_duration))
        # Changing starting day for next iteration
        day = pd.Timestamp(day) + pd.Timedelta('1d')
        ndays += 1
        elapsed_time = datetime.datetime.now() - now
        print("Total elapsed time: {}s".format(elapsed_time))
        print("Number of days evaluated so far: {}".format(ndays))

