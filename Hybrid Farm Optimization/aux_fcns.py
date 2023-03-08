' Auxiliary functions'
import numpy as np

#%% Expected benefits calculation
def Ben_Exp_Calc(PCC_P_Fore, Price_Fore):
    Ben_Exp = 0
    for h, P in enumerate(PCC_P_Fore):
        Ben_Exp = Ben_Exp + (P * Price_Fore[h])

    return Ben_Exp

#%% PCC powers calculation
def PCC_Powers(WTG_Psold, ESS_D, ESS_P):
    PCC_P = [a + b + c for a,b,c in zip(WTG_Psold, ESS_D, ESS_P)]

    return PCC_P

#%% Hourly xlabel ticks for plotting
def hourly_xticks(hour):
    hour_ticks = []  # X axis dates label
    for i in range(hour, 24):  # Filling X axis dates label
        if i < 0:
            pass
        else:
            hour_ticks.append('{}:00'.format(i))

    return hour_ticks

#%% Real benefits after deviations
def Ben_Real_Calc(Pgen_Real, PCC_P_Fore, DM_Price_Real, Dev_Way, Dev_Coef):
    Ben_Real = []       # Real benefits
    Dev_Costs = []      # Deviation costs
    Dev_Ps = []         # Deviation powers
    # Real benefits = Forecasted benefits +- Deviation costs/benefits
    for h in range(len(Dev_Way)):
        Dev_P = PCC_P_Fore[h] - Pgen_Real[h]
        Dev_Cost_h = 0              # Default deviation cost is zero if no deviations take place
        if Dev_P > 0:               # Upward deviation
            if Dev_Way[h] == 1:     # Upward deviation is against the system
                Dev_Cost_h = - DM_Price_Real[h] * (1 - Dev_Coef[h])
            else:                   # Upward deviation is in favor of the system
                Dev_Cost_h = - DM_Price_Real[h]

        if Dev_P > 0:               # Downward deviation
            if Dev_Way[h] == -1:    # Downward deviation is against the system
                Dev_Cost_h = DM_Price_Real[h] *  (1 + Dev_Coef[h])
            else:                   # Downward deviation is in favor of the system
                Dev_Cost_h = DM_Price_Real[h]
        Dev_Costs.append(Dev_Cost_h)
        Dev_Ps.append(-Dev_P)
        Ben_Real.append(PCC_P_Fore[h] * DM_Price_Real[h] - Dev_Cost_h)

    return sum(Ben_Real), sum(Dev_Costs), Dev_Costs, Dev_Ps

#%% Dev costs calculator
def Dev_Costs_Calc(Dev_Way, Dev_Coef, sense):
    Dev_Costs_Up = []
    Dev_Costs_Down = []
    for h in range(len(Dev_Way)):
        if sense == 'pred':  # When predicted all deviations are against the system with a coeficient fo 21%
            Dev_Cost_Down = (1 + 0.01 * 21)
            Dev_Cost_Up = -(1 - 0.01 * 21)
        if sense == 'real':
            if Dev_Way[h] == 1:  # Upward deviation is against the system, downward deviation is in favor
                Dev_Cost_Up = -(1 - 0.01*Dev_Coef[h])
                Dev_Cost_Down = 1
            if Dev_Way[h] == -1:  # Downward deviation is against the system, upward deviation is in favor
                Dev_Cost_Down = (1 + 0.01*Dev_Coef[h])
                Dev_Cost_Up = -1
        Dev_Costs_Up.append(Dev_Cost_Up)
        Dev_Costs_Down.append(Dev_Cost_Down)

    return  np.array(Dev_Costs_Down), np.array(Dev_Costs_Up)

#%% Hourly xlabel ticks for plotting
def hourly_xticks(hour):
    hour_ticks = []  # X axis dates label
    for i in range(hour, 24):  # Filling X axis dates label
        if i < 0:
            pass
        else:
            hour_ticks.append('{}:00'.format(i))
    return hour_ticks

#%% Accumulating all values of different daily measures for each case
def measure_accumulator(Cases_Results, casename, measure):
    Case_dict = Cases_Results[f'{casename}']
    measure_acc = 0
    E_acc = 0
    if measure == 'Ben_DM_Exp_MWh':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}']['Ben_DM_Exp']
            E_acc = E_acc + Case_dict[f'{day}']['Daily_Egen']
        measure_acc = measure_acc/E_acc
    elif measure == 'Ben_DM_Real_MWh':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}']['Ben_DM_Real']
            E_acc = E_acc + Case_dict[f'{day}']['Daily_Egen']
        measure_acc = measure_acc/E_acc
    elif measure == 'ID_purch_rel':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}'][f'{measure}']
        measure_acc = measure_acc / len(Case_dict.keys())
    else:
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}'][f'{measure}']
    if measure == 'ESS_deg':
        measure_acc = measure_acc * 100

    return measure_acc
#%% Accumulating all values of different daily measures for the dictionary of a single case
def local_measure_accumulator(Case_dict, measure):
    measure_acc = 0
    E_acc = 0
    if measure == 'Ben_DM_Exp_MWh':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}']['Ben_DM_Exp']
            E_acc = E_acc + Case_dict[f'{day}']['Daily_Egen']
        measure_acc = measure_acc/E_acc
    elif measure == 'Ben_DM_Real_MWh':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}']['Ben_DM_Real']
            E_acc = E_acc + Case_dict[f'{day}']['Daily_Egen']
        measure_acc = measure_acc/E_acc
    elif measure == 'ID_purch_rel':
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}'][f'{measure}']
        measure_acc = measure_acc / len(Case_dict.keys())
    else:
        for day in Case_dict.keys():
            measure_acc = measure_acc + Case_dict[f'{day}'][f'{measure}']
    if measure == 'ESS_deg':
        measure_acc = measure_acc * 100

    return measure_acc

#%% PCC Scheduled powers updater
from matplotlib import pyplot as plt
def PCC_Sch_updater(PCC_Sch, PCC_Sch_new, hour_i, hour_end):
    h_new= 0
    # print('*************************************')
    for h in range(hour_i, hour_end):
        # print(f'Hour {h}: {PCC_Sch[h]} + {PCC_Sch_new[h_new]} = {PCC_Sch[h] + PCC_Sch_new[h_new]}')
        PCC_Sch[h] = PCC_Sch[h] + PCC_Sch_new[h_new]
        h_new = h_new+1
    return np.round(PCC_Sch,2)

#%% Calendar degradation model
def Deg_Cal_model(daily_SOCs):
    Daily_deg_cal = 0
    for SOC in daily_SOCs:
        if SOC < 0.01:
            Daily_deg_cal = Daily_deg_cal + 0.002/24
        else:
            Daily_deg_cal = Daily_deg_cal + SOC * 0.00012/24

    return Daily_deg_cal/100



