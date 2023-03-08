import statsmodels.api as sm
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import statsmodels.api as sm
import time
import warnings
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_absolute_error
# Disabling Statsmodels ConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% Prediction with  SARIMA
def pred_DM_SARIMA(day_str, prices_df):
    day= pd.Timestamp(day_str)  # Converting day string to timestamp
    daily_direct_forecast = []  # Array for storing direct forecast

    # SARIMA model parameters
    train_length = 100                          # Training set length (days)
    model_order = (2, 1, 3)                     # SARIMA order
    model_seasonal_order = (1, 0, 1, 24)        # SARIMA seasonal order

    # Generating training set
    train_end = day - pd.Timedelta('1h')
    train_start = train_end - pd.Timedelta('{}d 23h'.format(train_length))
    train_set = prices_df[train_start:train_end]

    # Generating SARIMA model from doi 10.1109/SSCI44817.2019.9002930
    model = sm.tsa.SARIMAX(train_set, order=model_order, seasonal_order=model_seasonal_order,
                           initialization='approximate_diffuse')
    model_fit = model.fit(disp=False)

    # Generating prediction & storing on daily array
    prediction = model_fit.forecast(24)
    for i in range(24):
        daily_direct_forecast.append(round(prediction[i],2))

    return daily_direct_forecast


#%% Prediction with ANN
def ANN_pred(X_train, y_train, X_test, y_test):
    print("Generating ANN Prediction")
    start = time.time()
    ann = tf.keras.models.Sequential()
    # Hidden layers
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # Output layer
    ann.add(tf.keras.layers.Dense(units=1))
    # Configuring optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compiling
    ann.compile(optimizer=opt, loss='mean_absolute_percentage_error')
    # Training
    ann.fit(X_train, y_train, batch_size=32, epochs=100,verbose=0)
    y_pred_ANN = ann.predict(X_test)
    # y_pred_ANN = y_pred_ANN.tolist()
    pred_error = pred_evaluator(y_pred_ANN, y_test)
    print("ANN prediction generated")
    print(f'Elapsed time: {round(time.time() - start,2)}s')
    print(f'Error: {np.round(pred_error,2)}%')

    return y_pred_ANN

#%% Error calculation
def pred_evaluator(y_pred, y_test):
    pred_errors = []
    for i in range(min(len(y_pred),len(y_test))):
        pred_error = 100 * (abs(y_test[i] - y_pred[i])) / y_test[i]
        pred_errors.append(pred_error)
    return round(np.mean(pred_errors),2)


#%% Random forest estimator function
def pgen_estimator(rf_train_x, rf_train_y, rf_depth, windspe_pred):
    print('Generating wind power estimator with random forest')
    # Building forest
    forest_start = time.time()
    rf_WindPower = rf(bootstrap = False, max_depth = rf_depth)
    rf_WindPower.fit(rf_train_x, rf_train_y)
    Pgen_pred = rf_WindPower.predict(windspe_pred.reshape(-1, 1))
    print(f'Estimator generation time: {round(time.time() - forest_start, 2)}s')

    return Pgen_pred
#%% SARIMA prediction function
def windspe_predictor(train_set, model_order, model_seasonal_order, hour):
    print('******************************************************************')
    print('Generating wind speed forecast, hour: {}'.format(hour))
    SARIMA_start = time.time()
    model = sm.tsa.SARIMAX(train_set, order=model_order, seasonal_order=model_seasonal_order,
                            initialization='approximate_diffuse')
    model_fit = model.fit()
    # model_fit = model.fit(disp=False)
    print(f'Prediction generation time: {round(time.time() - SARIMA_start, 2)}s')
    prediction = model_fit.forecast(24-hour)
    if hour < 0:
        prediction = prediction[-24:]
    else:
        prediction = prediction[-(24-hour):]
    return prediction