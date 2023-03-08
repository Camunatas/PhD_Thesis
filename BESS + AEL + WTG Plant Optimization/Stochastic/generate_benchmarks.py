#%% Load python libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time

#%% Load other project files
from aux_fcns import *
from RHU_model import *
from plt_fcns import *
from sim_fcns import *
#%% Get parameters
RHU_Parameters = get_RHU_parameters()
Price_H2 = 4            # [€/kg] Hydrogen price
# Hydrogen price
#%% Load dataset
Global_dataset = np.load('Dataset_10.npy', allow_pickle=True).item()
#%% Generate benchmarks
data_folder = 'E:\Datos RHU Stochastic/'
starting_day =  '2020-01-01'
ending_day =  '2020-12-31'
day = starting_day
gen_start = time.time()
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    daily_inputs = Global_dataset[day_str]
    daily_benchmarks_dict = {}
    # Extracting inputs
    windspe_pred = daily_inputs['windspe_pred_DM']
    Pgen_pred = Siemens_SWT_30_113_curve(windspe_pred)
    price_pred = daily_inputs['price_pred']
    windspe_real = daily_inputs['windspe_real']
    Pgen_real = Siemens_SWT_30_113_curve(windspe_real)
    price_real = daily_inputs['price_real']
    # Generate programs
    direct_DM = RHU_DM(RHU_Parameters, Pgen_pred, price_pred, Price_H2)
    ideal_DM = RHU_DM(RHU_Parameters, Pgen_real, price_real, Price_H2)
    # Solve real-time operation for direct approach program
    DM_commitments_direct= [a + b for a, b in zip(direct_DM['P_WTG_Grid'], direct_DM['P_BESS_Grid'])]
    Purch_commitments_direct = direct_DM['P_Grid_AEL']
    direct_RT = RHU_RT(RHU_Parameters, Pgen_real, price_real,
                       DM_commitments_direct, Purch_commitments_direct, Price_H2)
    # Obtain programs results
    results_direct = calculate_CF(direct_RT, price_real, Price_H2, 'Real time mode')
    results_ideal = calculate_CF(ideal_DM, price_real, Price_H2, 'Day-ahead ')

    # Store results
    daily_benchmarks_dict['Direct approach results'] = results_direct
    daily_benchmarks_dict['Ideal results'] = results_ideal
    np.save(data_folder + f'Benchmarks/{day_str}.npy', daily_benchmarks_dict)
    print(f'Generated benchmarks for {day_str}')
    # Updating day
    day = pd.Timestamp(day) + pd.Timedelta('1d')
# Display timer
gen_end = time.time()
print(f'Benchmark solutions generated, elapsed time: {round((gen_end - gen_start)/60,4)} min,')


