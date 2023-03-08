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
# Run simulation
def run_simulation(start, end, RHU_Parameters_Global, H2_price, dataset, sim_name):
    sim_timer = time.time()
    # Simulation results dictionary
    Sim_Results = {}
    Sim_Results['BESS accumulated degradation'] = [0]
    Sim_Results['AEL accumulated degradation'] = [0]
    Sim_Results['AEL cycles'] = 0
    Sim_Results['BESS circulated energy'] = [0]
    Sim_Results['Generated H2'] = [0]
    Sim_Results['Energy market accumulated benefits'] = 0
    Sim_Results['Hydrogen production benefits'] = 0
    day = pd.Timestamp(start)
    RHU_Parameters = RHU_Parameters_Global.copy()            # Creating local parameters copy
    RHU_Parameters['Batt_E_nom'] = RHU_Parameters['Batt_E']
    RHU_Parameters['AEL nominal efficiency'] = RHU_Parameters['AEL efficiency']
    print('***********************************************************************')
    print(f'Launching {sim_name} case')
    while day != pd.Timestamp(end) + pd.Timedelta('1d'):
        # print(f'Running {day.strftime("%Y-%m-%d")}, {sim_name} case')
        # Load data
        Price_El = dataset_loader(day.strftime("%Y-%m-%d"), dataset, 'Price_pred_DM')
        windspe = dataset_loader(day.strftime("%Y-%m-%d"), dataset, 'windspe_pred_DM')
        P_Gen = Gamesa_G128_4500_curve(windspe)
        # Run optimization model
        Day_Results = RHU_model(RHU_Parameters, P_Gen, Price_El, H2_price)
        # Compute degradation and obtain new BESS capacity & AEL efficiency
        BESS_deg, AEL_deg = calculate_degradation(Day_Results, RHU_Parameters)
        # Calculate daily CF
        H_day = Day_Results['H_AEL'][-1] - RHU_Parameters['Initial hydrogen level']
        CF_day_arb, CF_day_H2 = calculate_CF(Price_El, H2_price, H_day, Day_Results['P_ex'])
        # Calculate degradation
        BESS_deg, AEL_deg = calculate_degradation(Day_Results, RHU_Parameters)
        # Print a warning when more than 2 BESS cycles have been made
        BESS_daily_E = sum(Day_Results['P_C_BESS']) + sum(Day_Results['P_D_BESS'])
        cycles = BESS_daily_E / (2 * RHU_Parameters['Batt_E'])
        if cycles > 2 * RHU_Parameters['Batt_E']:
            print(f'At {day}, {np.round(cycles, 2)} cycles were made by the BESS')
        # Save results
        Sim_Results['BESS accumulated degradation'].append(Sim_Results['BESS accumulated degradation'][-1] + BESS_deg)
        Sim_Results['AEL accumulated degradation'].append(Sim_Results['AEL accumulated degradation'][-1] + AEL_deg)
        print(Sim_Results['AEL accumulated degradation'][-1])
        for P in Day_Results['P_start_AEL']:
            if P > 0:
                Sim_Results['AEL cycles'] += 1
        Sim_Results['BESS circulated energy'].append(Sim_Results['BESS circulated energy'][-1] + BESS_daily_E)
        Sim_Results['Generated H2'].append(Sim_Results['Generated H2'][-1] + H_day)
        Sim_Results['Energy market accumulated benefits'] += CF_day_arb
        Sim_Results['Hydrogen production benefits'] += CF_day_H2
        # Update values for next day
        RHU_Parameters['Initial hydrogen level'] = Day_Results['H_AEL'][-1]
        RHU_Parameters['Initial state'] = Day_Results['state_AEL'][-1]*5/100
        RHU_Parameters['AEL initial H2 in O2'] = Day_Results['imp_AEL'][-1]
        RHU_Parameters['Batt_SOCi'] = Day_Results['SOC'][-1]*RHU_Parameters['Batt_E']/100
        RHU_Parameters['Batt_E'] = RHU_Parameters['Batt_E'] - RHU_Parameters['Batt_E_nom']*BESS_deg
        RHU_Parameters['AEL efficiency'] = RHU_Parameters['AEL efficiency'] \
                                           - RHU_Parameters['AEL nominal efficiency']*AEL_deg
        # Change to nex day
        day = day + pd.Timedelta('1d')
    # Finishing case run and printing timer
    print(f'Finished {sim_name} case:')
    BESS_deg_Acc = Sim_Results['BESS accumulated degradation'][-1]*100
    print(f'\t - BESS degradation: {np.round(BESS_deg_Acc,5)} %')
    AEL_deg_Acc = Sim_Results['AEL accumulated degradation'][-1]*100
    print(f'\t - AEL degradation: {np.round(AEL_deg_Acc,2)} %')
    AEL_cycles = Sim_Results['AEL cycles']
    print(f'\t - AEL cycles: {AEL_cycles} ')
    BESS_E_circ = Sim_Results['BESS circulated energy'][-1]
    print(f'\t - Circulated energy by ESS: {np.round(BESS_E_circ,2)} MWh ({np.round(BESS_E_circ/20,2)} cycles)')
    AEL_H2_gen = Sim_Results['Generated H2'][-1]
    print(f'\t - Generated hydrogen: {np.round(AEL_H2_gen,2)} kg')
    Energy_ben = Sim_Results['Energy market accumulated benefits']
    print(f'\t - Energy market benefits: {np.round(Energy_ben,2)} €')
    H2_ben = Sim_Results['Hydrogen production benefits']
    print(f'\t - Hydrogen market benefits: {np.round(H2_ben,2)} €')
    if round((time.time() - sim_timer)/3600) > 1:
        print(f'Elapsed time: {round((time.time() - sim_timer)/3600)}h')
    else:
        print(f'Elapsed time: {round((time.time() - sim_timer))}s')

    return Sim_Results
