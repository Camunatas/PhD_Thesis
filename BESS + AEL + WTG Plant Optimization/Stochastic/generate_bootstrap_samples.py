#%% Import libraries
import numpy as np
import time
from random import sample
import pandas as pd
#%% Load samples
#%% Bootstrapping sampling process
data_folder = 'E:\Datos RHU Stochastic/'
starting_day =  '2020-09-20'
ending_day =  '2020-12-31'
day = starting_day
repetitions = 10000
print('Initializing Bootstrap sampling')
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    Programs = np.load(data_folder + f'Samples/ev_{day_str}.npy', allow_pickle=True).item()
    Bootstrap_samples = {}
    bootstrap_timer = time.time()
    # Get original sample
    for key in Programs:
        sums = []
        means = []
        Program = Programs[key]
        original_sample = [Program['Costs samples'][i]['Total Cost'] for i in range(len(Program['Costs samples']))]
        sums.append(sum(original_sample))
        means.append(np.mean(original_sample))
        # Resample with repetition
        for i in range(repetitions):
            sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
            sums.append(sum(sample))
            means.append(np.mean(sample))
        Bootstrap_samples[key] = {}
        # Bootstrap_samples[key]['Sample of sums'] = sums
        Bootstrap_samples[key]['Sample of means'] = means

    print(f'Generated {day_str}, elapsed time: {np.round(time.time() - bootstrap_timer,2)} s')
    np.save(data_folder + f'Samples/boot_{day_str}.npy', Bootstrap_samples)
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')