import numpy as np
from matplotlib import pyplot as plt
from aux_fcns import *
import numpy_financial as npf
import pandas as pd
#%% Parameters
# For comparing ESS configuration cases
sim_folder = 'Results/' + '07_19_10_21_Year simulation for paper'
cases = ['Ideal', 'DM + ID', 'ID', 'DM', 'DM + SE', 'SE', 'CF']
# For comparing standalone vs CF
# sim_folder = 'Results/06_03_08_18_Standalone vs Capacity Firming'
# cases = ['CF', 'Standalone']
#%% Manipulating pyplot default parameters
# Forcing tight layout
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling showing figures
plt.ioff()
#%% Loading results .npy files
Cases_Results = {}
for case in cases:
    Cases_Results[f'{case}'] = np.load(sim_folder + f'/{case}/Case_Results.npy', allow_pickle=True).item()
#%% Creating measurements arrays
measures = ['Ben_DM_Exp', 'Dev_costs', 'Purch_costs', 'Ben_DM_Real']
Ben_DM_Exp_accs = []
Dev_costs_accs = []
Purch_costs_accs = []
Ben_DM_Real_accs = []
Cases_measures = {}
for case_name in Cases_Results.keys():
    print(f'Case "{case_name}":')
    Ben_DM_Exp_acc = measure_accumulator(Cases_Results, case_name, 'Ben_DM_Exp')
    Dev_costs_acc = measure_accumulator(Cases_Results, case_name, 'Dev_costs')
    Purch_costs_acc = measure_accumulator(Cases_Results, case_name, 'Purch_costs')
    Ben_DM_Real_acc = measure_accumulator(Cases_Results, case_name, 'Ben_DM_Real')
    print(f'\t - Expected benefits: {np.round(Ben_DM_Exp_acc/1000000,3)}M€')
    print(f'\t - Deviation costs: {np.round(Dev_costs_acc/1000000,3)}M€')
    print(f'\t - ID purchases costs: {np.round(Purch_costs_acc/1000000,3)}M€')
    print(f'\t - Real benefits: {np.round(Ben_DM_Real_acc/1000000,3)}M€')
    Ben_DM_Exp_accs.append(Ben_DM_Exp_acc)
    Dev_costs_accs.append(Dev_costs_acc)
    Purch_costs_accs.append(Purch_costs_acc)
    Ben_DM_Real_accs.append(Ben_DM_Real_acc)

#%% Plotting benefits
x = np.arange(len(cases))
fig = plt.figure('Simulation results')
plt.bar(x - 0.3, Ben_DM_Exp_accs, width=0.2, label='Expected benefits', edgecolor='black')
plt.bar(x - 0.1, Dev_costs_accs, width=0.2, label='Deviation costs/bonus', edgecolor='black')
plt.bar(x + 0.1, Purch_costs_accs, width=0.2, label='ID purchases costs', edgecolor='black')
plt.bar(x + 0.3, Ben_DM_Real_accs, width=0.2, label='Real benefits', edgecolor='black')
plt.xticks(x, cases)
plt.legend()
plt.grid()
plt.ylabel('€')
plt.show()
#%% Plotting Degradation
measure = 'ESS_deg'
measures_values = []
for case_name in Cases_Results.keys():
    measures_values.append(measure_accumulator(Cases_Results, case_name, measure))
fig = plt.figure(f'{measure}')
x = np.arange(len(cases))
plt.xticks(x, cases)
plt.bar(x, measures_values,  label='Ideal', edgecolor='black', zorder=3)
plt.grid(zorder=0)
plt.ylabel('Degradation (%)')
plt.show()
#%% Plotting Cycled energy
measure = 'ESS_E_Real'
measures_values = []
for case_name in Cases_Results.keys():
    measures_values.append(measure_accumulator(Cases_Results, case_name, measure))
fig = plt.figure(f'{measure}')
x = np.arange(len(cases))
plt.xticks(x, cases)
plt.bar(x, measures_values,  label='Ideal', edgecolor='black', zorder=3)
plt.grid(zorder=0)
plt.ylabel('Cycled Energy (MWh)')
plt.show()
#%% Plotting Relative ÍD purchases
measure = 'ID_purch_rel'
measures_values = []
for case_name in Cases_Results.keys():
    measures_values.append(measure_accumulator(Cases_Results, case_name, measure))
fig = plt.figure(f'{measure}')
x = np.arange(len(cases))
plt.xticks(x, cases)
plt.bar(x, measures_values,  label='Ideal', edgecolor='black', zorder=3)
plt.grid(zorder=0)
plt.ylabel('Relative energy purchased (%)')
plt.show()
#%% Plotting real benefits vs degradation
measure = 'ESS_deg'
measures_values = []
for case_name in Cases_Results.keys():
    measures_values.append(measure_accumulator(Cases_Results, case_name, measure))
Ben_vs_Deg = [a/b for a,b in zip(Ben_DM_Real_accs, measures_values)]
# Removing ideal results for best readability
Ben_vs_Deg = Ben_vs_Deg[-6:]
cases = cases[-6:]
fig = plt.figure('Benefits per degradation')
x = np.arange(len(cases))
plt.xticks(x, cases)
plt.bar(x, Ben_vs_Deg, edgecolor='black', zorder=3)
plt.ylim([110000, 142000])
plt.grid(zorder=0)
plt.ylabel('Real benefits per 1% of capacity lost (€)')
plt.show()

#%% NPV analysis
# Obtain daily CF
CFs_daily_avg = []
for Ben in Ben_DM_Real_accs:
    CFs_daily_avg.append(Ben/365)
# Obtain daily deg
measure = 'ESS_deg'
Degs_daily_avg = []
for case_name in Cases_Results.keys():
    Degs_daily_avg.append(measure_accumulator(Cases_Results, case_name, measure)/365/100)
# Determine discount rate
discount_yr = 7.5
discount_yr = discount_yr/100
discount_day = (1 + discount_yr) ** (1 / 365) - 1
# Extrapolate project
Projects_CFs = []
Projects_NPVs = []
print('Extrapolating')
for i in range(len(cases)):
    daily_cf = CFs_daily_avg[i]
    daily_deg = Degs_daily_avg[i]
    deg_acc = 0
    Project_CFs = []
    while deg_acc < 0.2:
        Project_CFs.append(daily_cf)
        deg_acc = daily_deg + deg_acc
        print(deg_acc)
    # Calculate NPV
    NPV = npf.npv(discount_day, Project_CFs)
    Projects_NPVs.append(NPV)
    # Save results
    Projects_CFs.append(Project_CFs)
print('Extrapolated')
# Print results
for i in range(len(cases)):
    case_CF_avg = CFs_daily_avg[i]
    case_deg_avg = Degs_daily_avg[i]*100
    case_dur = len(Projects_CFs[i])/365
    case_NPV = Projects_NPVs[i]
    print(f'Results with a discount rate of {discount_yr*100}%')
    print(f'Case "{cases[i]}":')
    print(f'\t Average daily CF: {np.round(case_CF_avg,2)} €')
    print('\t Average daily degradation: {:e} %'.format(case_deg_avg))
    print(f'\t Duration: {np.round(case_dur,2)} years')
    print(f'\t NPV: {np.round(case_NPV,3)} €')
    print(f'\t NPV: {np.round(case_NPV/1000000,3)} M€')


#%% Evaluating forecasts performance
from sklearn.metrics import mean_absolute_error as mape
starting_day = '2018-01-01'
ending_day = '2018-12-31'
day = pd.Timestamp(starting_day)
# Initializing MAPE lists
MAPEs_DM = []
MAPEs_ID2 = []
MAPEs_ID3 = []
MAPEs_ID4 = []
MAPEs_ID5 = []
MAPEs_ID6 = []
while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
    Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()
    day_str = day.strftime("%Y-%m-%d")
    Dataset = Global_dataset[day_str]
    # Obtaining real wind speed lists
    windspe_real = Dataset['windspe_real']
    windspe_real_DM = windspe_real
    windspe_real_ID2 = windspe_real[-len(Dataset['windspe_pred_ID2']):]
    windspe_real_ID3 = windspe_real[-len(Dataset['windspe_pred_ID3']):]
    windspe_real_ID4 = windspe_real[-len(Dataset['windspe_pred_ID4']):]
    windspe_real_ID5 = windspe_real[-len(Dataset['windspe_pred_ID5']):]
    windspe_real_ID6 = windspe_real[-len(Dataset['windspe_pred_ID6']):]
    # Saving daily MAPEs
    MAPEs_DM.append(mape(windspe_real_DM, Dataset['windspe_pred_DM']))
    MAPEs_ID2.append(mape(windspe_real_ID2, Dataset['windspe_pred_ID2']))
    MAPEs_ID3.append(mape(windspe_real_ID3, Dataset['windspe_pred_ID3']))
    MAPEs_ID4.append(mape(windspe_real_ID4, Dataset['windspe_pred_ID4']))
    MAPEs_ID5.append(mape(windspe_real_ID5, Dataset['windspe_pred_ID5']))
    MAPEs_ID6.append(mape(windspe_real_ID6, Dataset['windspe_pred_ID6']))
    # Changing to nex day
    day = day + pd.Timedelta('1d')

print('MAPEs:')
print(f'\t DM: {np.mean(MAPEs_DM)}')
print(f'\t ID2: {np.mean(MAPEs_ID2)}')
print(f'\t ID3: {np.mean(MAPEs_ID3)}')
print(f'\t ID4: {np.mean(MAPEs_ID4)}')
print(f'\t ID5: {np.mean(MAPEs_ID5)}')
print(f'\t ID6: {np.mean(MAPEs_ID6)}')

# Creating plotting lists
markets = ['DM', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6']
MAPE_avgs = [np.mean(MAPEs_DM), np.mean(MAPEs_ID2), np.mean(MAPEs_ID3),
             np.mean(MAPEs_ID4), np.mean(MAPEs_ID5), np.mean(MAPEs_ID6)]
#Plotting MAPEs averages
fig = plt.figure('MAPEs')
plt.plot(['DM', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6'], MAPE_avgs)
plt.ylabel('Wind speed forecast MAPE (%)')
plt.xlabel('Market')
plt.grid()
plt.show()


