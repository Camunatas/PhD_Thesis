#%% Importing libraries
import numpy as np
import pandas as pd

#%% Parameters
starting_day =  '2020-01-01 00:00:00'               # First day to evaluate
ending_day =  '2020-12-31 00:00:00'                 # last day to evaluate
Dataset = {}
#%% Loading external datasets
Prices_dataset = np.load('Price_pred_10.npy', allow_pickle=True).item()
windspe_dataset = np.load('Windspe_pred_10.npy', allow_pickle=True).item()
#%% Extracting data and saving data
day = starting_day
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    # Generating daily data dictionary
    Daily_dict = {}
    daily_key = pd.Timestamp(day).strftime("%Y-%m-%d")
    # Storing wind speed data
    windspe_day_dict = windspe_dataset[daily_key]
    for key in windspe_day_dict:
        Daily_dict[f'{key}'] = windspe_day_dict[f'{key}']
    # Storing DM prices
    prices_day_dict = Prices_dataset[daily_key]
    Daily_dict['price_pred'] = prices_day_dict['Price pred']
    Daily_dict['price_scenarios'] = prices_day_dict['Price scenarios']
    Daily_dict['price_real'] = [float(a) for a in prices_day_dict['Real prices'].values]
    # Storing on general dataset
    Dataset[pd.Timestamp(day).strftime("%Y-%m-%d")] = Daily_dict
    print(f"Saved day {day}")
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')

print('Dataset generated')
#%% Saving data into dictionary
np.save('Dataset_10.npy', Dataset)

#%%

