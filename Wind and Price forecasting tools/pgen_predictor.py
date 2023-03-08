'CURRENT CONFIGURATION : GENERATE REAL WIND SPEED DATASET'
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import os
# from forecast_fcns import *
import statsmodels.api as sm
from aux_fcns import *
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling Statsmodels warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
#%% General parameters
# Forecast parameters
hours = [0]
labels = ['DM']
# Data import parameters
df_day_start = '2012-01-01'
df_day_end = '2021-12-31'
df_day_start_ts = pd.Timestamp(df_day_start)
df_day_end_ts = pd.Timestamp(df_day_end) +5  pd.Timedelta('23h')
# SARIMA parameters
SARIMA_train_length = 5                     # [days] Training set length for SARIMA prediction
model_order = (2,0,1)                       # Order
model_seasonal_order = (2,0,3,12)           # Seasonal order
n_scenarios = 10                          # Number of daily scenarios
# Storage variables
Pgen_pred_dict = {}
save_folder = 'Tests plots' + '/Testing pgen scenario generation/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
#%% Preparing dataset
# Importing data
dataset_raw = pd.read_csv('Data/Brande_Dataset.csv', sep=';',parse_dates=['Date'], index_col="Date")
# Slicing data
dataset = dataset_raw.loc[df_day_start_ts:df_day_end_ts]['Speed']
# Removing nans
dataset = dataset.fillna(method='ffill')
# Plotting historical data for paper
# dataset_raw[df_day_start_ts:df_day_end_ts]['Speed'].astype(float).plot(xlabel=' ', ylabel='Wind speed (m/s)')
#%% Implement wind turbine power curve
def WTG_curve(windspeed):
    # Gamesa G128/4500
    speed = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,
             15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27]
    power = [0,0,0,0,75,120,165,230,300,450,600,760,967,1250,1533,1870,2200,2620,3018,3450,3774,4080,4314,4430,4490,
             4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4403,4306,4210,4113,4016,3919,3823,3725,3629,
             3532,3435,3339,3242,3145,3048,2950,2855,2758]
    WTG_curve = {}
    for p,v in enumerate(speed):
        WTG_curve[f'{v}'] = power[p]/1000
    Pgen = []
    for v in windspeed:
        v = round(v * 2) / 2
        if v < 0 or v > speed[-1]:
            Pgen.append(WTG_curve[f'{0}'])
        else:
            Pgen.append(WTG_curve[f'{round(v)}'])
    return(Pgen)
#%% Generate wind power prediction
day_start ='2020-01-01'
day_end = '2020-12-31'
day_start_ts = pd.Timestamp(day_start)
day_end_ts = pd.Timestamp(day_end) + pd.Timedelta('23h')
day = day_start_ts
pred_start = time.time()
while day != day_end_ts + pd.Timedelta('1h'):
    print('***************************************************************')
    print(f'Generating prediction for {day.strftime("%Y-%m-%d")}')
    day_pred_start = time.time()
    Predictions = {}
    windspe_errors = []
    Pgen_errors = []
    for i, hour in enumerate(hours):
        hour_rf = pd.Timestamp(day) + pd.Timedelta(f'{hour}h')
        # Generating training sets
        SARIMA_train = SARIMA_train_set_gen(day, dataset, hour, SARIMA_train_length)
        # Generating test sets
        if hour < 1:
            test_start = pd.Timestamp(day)
        elif hour == 0:
            test_start = pd.Timestamp(day)
        else:
            test_start = pd.Timestamp(day) + pd.Timedelta('{}h'.format((hour)))
        test_end = pd.Timestamp('{} {}'.format(pd.Timestamp(day).strftime('%Y-%m-%d'), '23:00:00'))
        windspe_real = dataset[test_start:test_end].values
        windspe_real = windspe_real.astype(float)
        Pgen_real = WTG_curve(windspe_real)
        # Generating wind speed prediction
        print('******************************************************************')
        print('Generating wind speed forecast, hour: {}'.format(hour))
        SARIMA_start = time.time()
        # Declare model: SARIMA
        model = sm.tsa.SARIMAX(SARIMA_train, order=model_order, seasonal_order=model_seasonal_order,
                               initialization='approximate_diffuse')
        model_fit = model.fit(disp=False)
        print(f'Prediction generation time: {round(time.time() - SARIMA_start, 2)}s')
        prediction = model_fit.forecast(24 - hour)
        if hour < 0:
            prediction = prediction[-24:]
        else:
            prediction = prediction[-(24 - hour):]
        windspe_pred = prediction
        # Analizing and storing prediction results
        windspe_error = round(np.mean(windspe_real-windspe_pred),2)
        windspe_errors.append(windspe_error)
        # Saving real values (only in first iteration)
        if i == 0:
            Predictions['windspe_real'] = windspe_real
        # Generating prediction scenarios
        new_scenarios = model_fit.simulate(nsimulations = len(windspe_real), repetitions = n_scenarios, anchor = 'end')
        scenarios = {}
        for scenario in range(n_scenarios):
            scenarios[f'{scenario}'] = []
            for hour in range(len(windspe_real)):
                scenarios[f'{scenario}'].append(abs(new_scenarios[hour][0][scenario]))
        # Saving prediction arrays
        Predictions['windspe_pred_{}'.format(labels[i])] = windspe_pred
        Predictions['windspe_scenarios_{}'.format(labels[i])] = scenarios
        # Printing results
        print("Wind speed prediction error: {} m/s".format(windspe_error))
    Predictions['hours'] = hours
    Pgen_pred_dict[day.strftime("%Y-%m-%d")] = Predictions
    print(f'Day elapsed time: {round(time.time() - day_pred_start, 2)}s')
    day = day + pd.Timedelta('1d')
print(f'Total elapsed time: {round((time.time() - pred_start)/3600, 2)}h ({round((time.time() - pred_start), 2)})s')
#%% Saving results
np.save('Windspe_pred_10.npy', Pgen_pred_dict)
#%% Plotting results
# Pgen_pred_dict = np.load('Pgen_pred.npy', allow_pickle=True).item()
# day_start ='2021-04-01'
# day_end = '2021-05-01'
# day_start_ts = pd.Timestamp(day_start)
# day_end_ts = pd.Timestamp(day_end) + pd.Timedelta('23h')
# day = day_start_ts
# while day != day_end_ts + pd.Timedelta('1h'):
#     Predictions = Pgen_pred_dict[day.strftime("%Y-%m-%d")]
#     # fig = plt.figure('Prediction results for {}'.format(day))
#     # # plt.suptitle('Prediction results for {}'.format(day))
#     # hour_ticks = hourly_xticks(Predictions['hours'][0])
#     # ticks_x = np.arange(0, len(hour_ticks), 1)  # Vertical grid spacing
#     # plt.xticks(np.arange(0, len(hour_ticks), 1), '', rotation=45)
#     # for i,hour in enumerate(Predictions['hours']):
#     #     windspe_pred_list = Predictions['windspe_pred_{}'.format(labels[i])].tolist()
#     #     while len(windspe_pred_list) != len(hour_ticks):
#     #         windspe_pred_list.insert(0,None)            # Filling with nones to the left to adjust plot length
#     #     plt.plot(windspe_pred_list, label=f'{labels[i]}')
#     # plt.plot(Predictions['windspe_real'], '--', label='Observation')
#     # plt.ylabel('Wind speed (m/s)')
#     # plt.legend()
#     # plt.grid()
#     # plt.savefig(save_folder +
#     #             f'Prediction results {day.strftime("%Y-%m-%d")}.png')
#     # # plt.savefig(save_folder +
#     # #             f'Prediction results {day.strftime("%Y-%m-%d")}.svg')
#     # plt.close()
#     # fig = plt.figure('Scenarios for {}'.format(day))
#     scenarios = Predictions['scenarios_DM']
#     windspe_real = Predictions['windspe_real']
#     for i in range(len(scenarios)):
#         plt.plot(scenarios["{}".format(i)], c=np.random.rand(3))
#         plt.xlabel("Hour")
#         plt.ylabel("Wind speed (m/s)")
#     plt.plot(windspe_real, '--')
#     plt.grid()
#     plt.savefig(save_folder +
#                 f'Scenarios {day.strftime("%Y-%m-%d")}.png')
#     plt.close()
#     day = day + pd.Timedelta('1d')
