# --    AUXILIARY FUNCTIONS FUNCTIONS     --
#%% Importing libraries
import pandas as pd
import numpy as np

#%% Dataset loader
def dataset_loader(day, dataset, string):
    return  dataset[day][f'{string}']

#%% Wind turbine curves
# Gamesa G128/4500
def Gamesa_G128_4500_curve(windspeed):
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


#%% Calculate daily cash flow
def calculate_CF(prices, H2_price, H, Powers):
    Ben_arb = 0
    # Calculate arbitrage benefits
    for i in range(min(len(prices), len(Powers))):
        Ben_arb = Ben_arb + prices[i]*Powers[i]
    # Calculate hydrogen production benefits
    H2_arb = H2_price*H

    return Ben_arb, H2_arb

#%% Calculate daily degradation
def calculate_degradation(Day_Results, RHU_Parameters):
    # Extract data from input dictionaries
    AEL_Pstarts = Day_Results['P_start_AEL']
    AEL_Pmax = RHU_Parameters['AEL Maximum power']
    AEL_t_start = RHU_Parameters['Cold start time']
    AEL_cycles = RHU_Parameters['Lifetime cycles']
    SOC = Day_Results['SOC']
    batt_capacity = RHU_Parameters['Batt_E']
    batt_powers = [a + b for a,b in zip(Day_Results['P_C_BESS'],Day_Results['P_C_BESS'])]
    EOL_Capacity = RHU_Parameters['Batt_EOL']

    # BESS degradation model
    deg_cost = 0
    deg_cost1 = 0
    cost = 1
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    deg_cost_per_cycle = [0., cost / 1000000., cost / 200000., cost / 60000., cost / 40000.,
                          cost / 20000., cost / 15000., cost / 11000., cost / 10000., cost / 8000., cost / 7000.,
                          cost / 6000.]

    benefits = []
    en100 = sum(batt_powers) / 2 / batt_capacity * 100
    DOD = max(SOC) - min(SOC)
    for d in range(len(DOD_index) - 1):
        if DOD >= DOD_index[d] and DOD <= DOD_index[d + 1]:
            deg_cost = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                    DOD - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    DOD1 = max(en100 - DOD, 0)
    if DOD1 > 100:
        deg_cost1 = deg_cost_per_cycle[-1]
    for d in range(len(DOD_index) - 1):
        if DOD1 >= DOD_index[d] and DOD1 <= DOD_index[d + 1]:
            deg_cost1 = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                    DOD1 - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break
    BESS_RUL_loss = (deg_cost + deg_cost1) / cost
    BESS_deg = BESS_RUL_loss * (1 - EOL_Capacity)
    # AEL degradation model
    deg_AEL = 0
    for t in range(len(AEL_Pstarts)):
        deg_AEL = deg_AEL  + AEL_Pstarts[t] / (AEL_Pmax * AEL_t_start * AEL_cycles)

    return BESS_deg, deg_AEL