import pandas as pd
import numpy as np

# Train set splitter for SARIMA
def SARIMA_train_set_gen(day, dataset, hour, train_length):
    day_utc = pd.Timestamp(day)
    train_end = day_utc + pd.Timedelta('{}h'.format(hour-1))
    train_start = train_end - pd.Timedelta('{}d'.format(train_length))
    # train_set = dataset[train_start:train_end].values
    train_set = dataset.loc[train_start.strftime('%Y-%m-%d'):train_end.strftime('%Y-%m-%d')].values
    train_set = train_set.astype(float)
    return train_set

# Train set splitter for random forest
def rf_train_set_gen(hour_rf, dataset, rf_train_length):
    rf_train_end = hour_rf - pd.Timedelta('1h')
    rf_train_start = rf_train_end - pd.Timedelta(f'{rf_train_length}d')
    rf_dataset = dataset[rf_train_start:rf_train_end]
    rf_train_x = dataset.iloc[:, 0].values.reshape(-1, 1)
    rf_train_y = dataset.iloc[:, 2].values
    return rf_train_x, rf_train_y

# Hourly xlabel ticks for plotting
def hourly_xticks(hour):
    hour_ticks = []  # X axis dates label
    for i in range(hour, 24):  # Filling X axis dates label
        if i < 0:
            pass
        else:
            hour_ticks.append('{}:00'.format(i))
    return hour_ticks

#