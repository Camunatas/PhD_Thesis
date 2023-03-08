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
import time
from aux_fcns import *
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
# Disabling Statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
#%% Initializing parameters
# Control variables
starting_day =  '2020-01-01 00:00:00'               # First day to evaluate
ending_day =  '2020-12-31 00:00:00'                 # last day to evaluate
# starting_day =  '2021-03-01 00:00:00'               # First day to evaluate
# ending_day =  '2021-03-31 00:00:00'                 # last day to evaluate
# SARIMa model parameters
train_length = 100                          # Training set length (days)
model_order = (2, 1, 3)                     # SARIMA order
model_seasonal_order = (1, 0, 1, 24)        # SARIMA seasonal order
n_scenarios = 10                          # Number of daily scenarios
# Auxiliary variables
Price_pred_dict = {}
now = time.time()           # Simulation time
#%% Importing price data from csv
prices_df = pd.read_csv('Data/Prices_DK1.csv', sep=';', usecols=["Price","datetime"], parse_dates=['datetime'],
                        index_col="datetime")
# prices_df = prices_df.asfreq('h')

#%% Launching algorithm
day = starting_day
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    day_pred_start = time.time()
    # Initializing daily variables
    daily_direct_forecast = []                # Array with forecasted prices

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
    # Generating prediction scenarios
    new_scenarios = model_fit.simulate(nsimulations=len(test_set), repetitions=n_scenarios, anchor='end')
    scenarios = {}
    for scenario in range(n_scenarios):
        scenarios[f'{scenario}'] = new_scenarios[('Price', scenario)].values
    # Storing data
    daily_dict = {}
    daily_dict['Real prices'] = test_set
    daily_dict['Price pred'] = daily_direct_forecast
    daily_dict['Price scenarios']  = scenarios
    Price_pred_dict[pd.Timestamp(day).strftime("%Y-%m-%d")] = daily_dict
    print('Generated for {}'.format(pd.Timestamp(day).strftime("%Y-%m-%d")))
    # Updating day variables
    day = pd.Timestamp(day) + pd.Timedelta('1d')
    print(f'Day elapsed time: {round(time.time() - day_pred_start, 2)}s')
print(f'Total elapsed time: {round(time.time() - now, 2)}s')
#%% Plotting last day predicted (for paper)
# hour_ticks = hourly_xticks(0)
# ticks_x = np.arange(0, len(hour_ticks), 1)
# plt.xticks(np.arange(0, len(hour_ticks), 1), hour_ticks, rotation=45)
# plt.plot(daily_direct_forecast, label='Predicted')
# plt.plot(test_set.values, label='Real')
# plt.ylabel('Price (€/MWh)')
# plt.grid()
# plt.legend()
# plt.show()
#%% Plotting last day scenarios (for paper)
# fig = plt.figure('Scenarios)
# for i in range(len(scenarios)):
#     plt.plot(scenarios["{}".format(i)], c=np.random.rand(3))
#     plt.xlabel("Hour")
#     plt.ylabel('Price (€/MWh)')
# plt.grid()
#%% Saving results
np.save('Price_pred_10.npy', Price_pred_dict)

