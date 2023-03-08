#%% Importing libraries
import pandas as pd
from pandas.core.frame import DataFrame
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
import datetime
from pyomo.environ import *
from pyomo.opt import SolverFactory
import warnings
import concurrent.futures
import os
from datetime import timezone
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
starting_day =  '2016-01-01 00:00:00'               # First day to evaluate
ending_day =  '2020-12-31 00:00:00'                 # last day to evaluate
# SARIMa model parameters
train_length = 100                          # Training set length (days)
model_order = (2, 1, 3)                      # SARIMA order
model_seasonal_order = (1, 0, 1, 24)        # SARIMA seasonal order
# Auxiliary variables
direct_forecasts = []                        # Array with direct forecasts
direct_Ps = []                              # Array with direct schedules powers
direct_SOCs = []                            # Array with direct schedules SOCs
now = datetime.datetime.now()           # Simulation time
directory = "Direct predictions/" + "2016_2020"
if not os.path.exists(directory):
    os.makedirs(directory)
#%% Importing price data from csv
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","datetime"], parse_dates=['datetime'],
                        index_col="datetime")
prices_df = prices_df.asfreq('h')

#%% BESS parameters
Batt_Enom = 50                              # [MWh] Battery nominal capacity
Batt_Pnom = Batt_Enom/4                     # [MW] Battery nominal power
Batt_ChEff = 0.95                           # BESS charging efficiency
Batt_DchEff = 0.9                           # BESS discharging efficiency
Batt_Cost= 37.33*Batt_Enom*1000           # [€] BESS cost
Batt_Eff = 0.9                              # Provisional Battery efficiency
Batt_SOC_init = 0                           # Initial SOC

#%% Launching algorithm
day = starting_day
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    # Initializing daily variables
    daily_direct_forecast = []                # Array with forecasted prices
    daily_direct_P = []                       # Array with direct schedule powers
    daily_direct_SOC = []                     # Array with direct schedule SOCs

    # Generating training set
    day_utc = pd.Timestamp(day).replace(tzinfo=timezone.utc)
    train_end = day_utc - pd.Timedelta('1h')
    train_start = train_end - pd.Timedelta('{}d 24h'.format(train_length))
    train_set = prices_df[train_start:train_end-pd.Timedelta('1h')]
    # Defining test set
    test_start = day_utc - pd.Timedelta('1h')
    test_end = test_start + pd.Timedelta('23h')
    test_set = prices_df[test_start:test_end]
    # Generating SARIMA model from doi 10.1109/SSCI44817.2019.9002930
    model = sm.tsa.SARIMAX(train_set, order=model_order, seasonal_order=model_seasonal_order,
                           initialization='approximate_diffuse')
    model_fit = model.fit(disp=False)

    # Generating prediction & storing on daily array
    prediction = model_fit.forecast(len(test_set))
    for i in range(len(test_set)):
        daily_direct_forecast.append(prediction.iloc[i])

    # Generating schedule
    daily_direct_P, daily_direct_SOC = arbitrage(Batt_SOC_init, daily_direct_forecast, Batt_Enom, Batt_Pnom,
                                           Batt_ChEff, Batt_Cost)

    # Storing data
    direct_forecasts.append(daily_direct_forecast)
    direct_Ps.append(daily_direct_P)
    direct_SOCs.append(daily_direct_SOC)
    print('Generated for {}'.format(day))
    # Updating day variables
    day = pd.Timestamp(day) + pd.Timedelta('1d')


# Saving results
np.save(directory+'/direct_forecasts.npy', direct_forecasts)
np.save(directory+'/direct_Ps.npy', direct_Ps)
np.save(directory+'/direct_SOCs.npy', direct_SOCs)

