#%% Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import os
from common_funcs_v3 import *
#%% Manipulating libraries parameters for suiting the code
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
#%% Initializing parameters
# Control variables
starting_day = '2016-01-01 00:00:00'               # First day to evaluate
ending_day = '2020-12-31 00:00:00'                 # Last day to evaluate
Ks = [0, 0.25, 0.5, 0.75, 1,2 ,3 ,4]
# Ks = [0.5]
# Input data folders
schedules_directory = "2016_2020"
direct_pred_directory = "2016_2020"
# Montecarlo analysis results
montecarlo_analysis = np.load('montecarlo_analysis.npy', allow_pickle=True).item()
# Generating output folder
analysis_folder = "Analysis/" + "Brute benefits & K_05"
if not os.path.exists(analysis_folder):
    os.makedirs(analysis_folder)
# Global results storage arrays
Daily_Bens_ES = []
Daily_Bens_ETR = []
Daily_Bens_ES_ETR = []
Daily_Bens_mean = []
Daily_Bens_direct = []
Daily_Bens_ideal = []
Deg_accs_ES = []
Deg_accs_ETR = []
Deg_accs_ES_ETR = []
Deg_accs_mean = []
Deg_accs_direct = []
Deg_accs_ideal = []
Ben_acc_ES = []
Ben_acc_ETR = []
Ben_acc_ES_ETR = []
Ben_acc_mean = []
En_acc_ES = []
En_acc_ETR = []
En_acc_ES_ETR = []
En_acc_mean = []
# Auxiliary variables
figurecount = 0                         # Figure counter
dates_label = []                        # X axis dates label
for i in range(24):                     # Filling X axis dates label
    dates_label.append('{}:00'.format(i))
#%% Importing global data
# Electricity prices
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","Hour"], parse_dates=['Hour'], index_col="Hour")
# prices_df = prices_df.asfreq('H')
#%% BESS parameters
Batt_Enom = 50                              # [MWh] Battery nominal capacity
Batt_Pnom = Batt_Enom/4                     # [MW] Battery nominal power
Batt_ChEff = 0.95                           # BESS charging efficiency
Batt_Cost= 37.33*Batt_Enom*1000             # [€] BESS cost
Batt_SOC_init = 0                           # Initial SOC
EOL_Capacity = 0.8                          # [%/100] Capacity of BESS EOL
Daily_deg_cal = 2.03/100/365               # [%/100] Daily calendar degradation
#%% Analizing results day by day by importing generated scenarios & schedules
constant_results = True
k_eval_global_start = datetime.datetime.now() # Initializing k evaluation run chronometer
for K in Ks:
    now = datetime.datetime.now()           # Simulation time
    # Launching analysis
    day = starting_day
    d = 0
    # Initializing storage arrays for non iterative results
    if constant_results:
        Results_direct = []     # Array with results using direct forecast
        Results_ideal = []      # Array with results with perfect forecast
        Results_mean = []       # Array with results using mean
        Energy_direct = []      # Array with circulated energy using direct forecast
        Energy_ideal = []       # Array with circulated energy with perfect forecast
        Energy_mean = []        # Array with circulated energy using mean
        Deg_direct = []         # Array with daily degradation using direct forecast
        Deg_ideal = []          # Array with daily degradation with perfect forecast
        Deg_mean = []           # Array with daily degradation using mean
        deg_acc_direct = [0]    # Array with accumulated degradation with direct criteria
        deg_acc_ideal = [0]     # Array with accumulated degradation with perfect forecast
        deg_acc_mean = [0]      # Array with accumulated degradation using mean
    # Local results storage arrays (only iterative)
    Results_ES = []             # Array with results using expected shortfall
    Results_ETR = []            # Array with results using expected tail return
    Results_ES_ETR = []         # Array with results using both ES and ETR
    Energy_ES = []              # Array with circulated energy using expected shortfall
    Energy_ETR = []             # Array with circulated energy using expected tail return
    Energy_ES_ETR = []          # Array with circulated energy using both ES and ETR
    Deg_ES = []                 # Array with daily degradation using expected shortfall
    Deg_ETR = []                # Array with daily degradation using expected tail return
    Deg_ES_ETR = []             # Array with daily degradation using both ES and ETR
    deg_acc_ES = [0]            # Array with accumulated degradation expected shortfall
    deg_acc_ETR = [0]           # Array with accumulated degradation with ETR criteria
    deg_acc_ES_ETR  = [0]       # Array with accumulated degradation with both ES and ETR
    while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
        run_start = datetime.datetime.now() # Initializing daily run chronometer
        # Obtaining real prices array
        day_start = pd.Timestamp(day)
        prices_real = []  # Array with real prices
        day_end = day_start + pd.Timedelta('23h')
        day_set = prices_df[day_start:day_end]
        for j in range(len(day_set)):
            prices_real.append(day_set["Price"][j])

        # Importing daily schedule & scenarios
        day_results_folder = "Results/" + schedules_directory + \
                             "/{}".format(pd.Timestamp(day).strftime("%Y_%m_%d"))
        Schedules_P = np.load(day_results_folder + '/Schedules_P.npy', allow_pickle=True).item()
        Schedules_SOC = np.load(day_results_folder + '/Schedules_SOC.npy', allow_pickle=True).item()
        scenarios = np.load(day_results_folder + '/scenarios.npy', allow_pickle=True).item()

        # Running ideal, mean an direct results only for first loop iteration
        if constant_results:
            # Obtaining results with ideal approach
            ideal_powers, ideal_SOCs = arbitrage(Batt_SOC_init, prices_real, Batt_Enom, Batt_Pnom, Batt_ChEff,
                                                 Batt_Cost)
            ideal_circulated_energy = energy(ideal_powers)
            ideal_benefits, ideal_daily_EOL_loss = scen_eval(ideal_powers, prices_real, ideal_SOCs, Batt_Cost, Batt_Enom)
            daily_deg_ideal = ideal_daily_EOL_loss * (1 - EOL_Capacity)
            ideal_benefits, ideal_circulated_energy, daily_deg_ideal = \
                Deg_scaler(ideal_powers, ideal_benefits, ideal_SOCs, deg_acc_ideal[-1], EOL_Capacity, daily_deg_ideal,
                           Batt_Cost, Daily_deg_cal)
            deg_acc_ideal.append(deg_acc_ideal[-1] + daily_deg_ideal)
            Results_ideal.append(ideal_benefits)
            Energy_ideal.append(ideal_circulated_energy)
            Deg_ideal.append(daily_deg_ideal)
            # Obtaining results with direct approach
            direct_forecasts = np.load('Direct predictions/'+direct_pred_directory+
                                       '/direct_forecasts.npy', allow_pickle=True)
            direct_Ps = np.load('Direct predictions/'+direct_pred_directory+
                                '/direct_Ps.npy', allow_pickle=True)
            direct_SOCs = np.load('Direct predictions/'+direct_pred_directory+
                                  '/direct_SOCs.npy', allow_pickle=True)
            direct_powers = direct_Ps[d]
            direct_schedule_SOC = direct_SOCs[d]
            daily_results_direct, daily_EOL_LOSS_direct = scen_eval(direct_powers, prices_real,
                                                                    direct_schedule_SOC, Batt_Cost, Batt_Enom)
            daily_deg_direct = daily_EOL_LOSS_direct * (1-EOL_Capacity)
            daily_e_acc_direct = energy(direct_powers)
            daily_results_direct, daily_e_acc_direct, daily_deg_direct = \
                Deg_scaler(direct_powers, daily_results_direct, direct_schedule_SOC, deg_acc_direct[-1], EOL_Capacity,
                           daily_deg_direct, Batt_Cost, Daily_deg_cal)
            Results_direct.append(daily_results_direct)
            Energy_direct.append(daily_e_acc_direct)
            Deg_direct.append(daily_deg_direct)
            deg_acc_direct.append(deg_acc_direct[-1] + daily_deg_direct)
            # Obtaining results with mean approach
            Schedules_ben_means = montecarlo_analysis['{}'.format(day)][2]
            best_schedule_id_mean = np.argmax(Schedules_ben_means)          # Best schedule using only the mean
            mean_powers = Schedules_P['{}'.format(best_schedule_id_mean)]
            mean_SOCs = Schedules_SOC['{}'.format(best_schedule_id_mean)]
            daily_results_mean, daily_EOL_LOSS_mean = scen_eval(mean_powers, prices_real, mean_SOCs,
                                                           Batt_Cost, Batt_Enom)
            daily_deg_mean = daily_EOL_LOSS_mean * (1-EOL_Capacity)
            daily_e_acc_mean = energy(mean_powers)
            daily_results_mean, daily_e_acc_mean, daily_deg_mean = \
                Deg_scaler(mean_powers, daily_results_mean, mean_SOCs, deg_acc_mean[-1], EOL_Capacity, daily_deg_mean,
                           Batt_Cost, Daily_deg_cal)
            Results_mean.append(daily_results_mean)
            Energy_mean.append(daily_e_acc_mean)
            Deg_mean.append(daily_deg_mean)
            deg_acc_mean.append(deg_acc_mean[-1] + daily_deg_mean)


        # Obtaining results with ES and ETR approaches
        Schedules_ESs = montecarlo_analysis['{}'.format(day)][0]
        Schedules_ETRs = montecarlo_analysis['{}'.format(day)][1]
        y_ESs = []
        y_ETRs = []
        y_ES_ETRs = []
        for i in range(len(Schedules_ben_means)):
            y_ES = Schedules_ben_means[i] + K*Schedules_ESs[i]      # Selecting best schedule with ETR
            y_ETR = Schedules_ben_means[i] + K*Schedules_ETRs[i]    # Selecting best schedule with ETR
            y_ES_ETR = Schedules_ben_means[i] + K*Schedules_ETRs[i]  \
                    + K*Schedules_ESs[i]                            # Selecting best schedule with ETR & ES
            if Schedules_ben_means[i] == 0:
                y_ESs.append(0)
                y_ETRs.append(0)
                y_ES_ETRs.append(0)
            else:
                y_ESs.append(y_ES)
                y_ETRs.append(y_ETR)
                y_ES_ETRs.append(y_ES_ETR)
        best_schedule_id_ES = np.argmax(y_ESs)
        ES_powers = Schedules_P['{}'.format(best_schedule_id_ES)]
        ES_SOCs = Schedules_SOC['{}'.format(best_schedule_id_ES)]
        daily_results_ES, daily_EOL_LOSS_ES = scen_eval(ES_powers, prices_real, ES_SOCs,
                                                   Batt_Cost, Batt_Enom)
        daily_deg_ES = daily_EOL_LOSS_ES * (1-EOL_Capacity)
        daily_e_acc_ES = energy(ES_powers)
        daily_results_ES, daily_e_acc_ES, daily_deg_ES = \
            Deg_scaler(ES_powers, daily_results_ES, ES_SOCs, deg_acc_ES[-1], EOL_Capacity, daily_deg_ES,
                       Batt_Cost, Daily_deg_cal)
        Results_ES.append(daily_results_ES)
        Energy_ES.append(daily_e_acc_ES)
        Deg_ES.append(daily_deg_ES)
        deg_acc_ES.append(deg_acc_ES[-1] + daily_deg_ES)
        best_schedule_id_ETR = np.argmax(y_ETRs)
        ETR_powers = Schedules_P['{}'.format(best_schedule_id_ETR)]
        ETR_SOCs = Schedules_SOC['{}'.format(best_schedule_id_ETR)]
        daily_results_ETR, daily_EOL_LOSS_ETR= scen_eval(ETR_powers, prices_real, ETR_SOCs,
                                                    Batt_Cost, Batt_Enom)
        daily_deg_ETR = daily_EOL_LOSS_ETR * (1-EOL_Capacity)
        daily_e_acc_ETR = energy(ETR_powers)
        daily_results_ETR, daily_e_acc_ETR, daily_deg_ETR = \
            Deg_scaler(ETR_powers, daily_results_ETR, ETR_SOCs, deg_acc_ETR[-1], EOL_Capacity, daily_deg_ETR,
                       Batt_Cost, Daily_deg_cal)
        Results_ETR.append(daily_results_ETR)
        Energy_ETR.append(daily_e_acc_ETR)
        Deg_ETR.append(daily_deg_ETR)
        deg_acc_ETR.append(deg_acc_ETR[-1] + daily_deg_ETR)
        best_schedule_id_ES_ETR = np.argmax(y_ES_ETRs)
        ES_ETR_powers = Schedules_P['{}'.format(best_schedule_id_ES_ETR)]
        ES_ETR_SOCs = Schedules_SOC['{}'.format(best_schedule_id_ES_ETR)]
        daily_results_ES_ETR, daily_EOL_LOSS_ES_ETR= scen_eval(ES_ETR_powers, prices_real, ES_ETR_SOCs,
                                                    Batt_Cost, Batt_Enom)
        daily_deg_ES_ETR = daily_EOL_LOSS_ES_ETR * (1-EOL_Capacity)
        daily_e_acc_ES_ETR = energy(ES_ETR_powers)
        daily_results_ES_ETR, daily_e_acc_ES_ETR, daily_deg_ES_ETR = \
            Deg_scaler(ES_ETR_powers, daily_results_ES_ETR, ES_ETR_SOCs, deg_acc_ES_ETR[-1], EOL_Capacity,
                       daily_deg_ES_ETR, Batt_Cost, Daily_deg_cal)
        Results_ES_ETR.append(daily_results_ES_ETR)
        Energy_ES_ETR.append(daily_e_acc_ES_ETR)
        Deg_ES_ETR.append(daily_deg_ES_ETR)
        deg_acc_ES_ETR.append(deg_acc_ES_ETR[-1] + daily_deg_ES_ETR)
        # Displaying daily simulation elapsed time
        run_end = datetime.datetime.now()
        run_duration = run_end - run_start
        # Printing daily results (only for one K evaluated)
        if len(Ks) == 1:
            print("*********************************************************************************")
            print("Day: {}, Elapsed time: {}".format(day, run_duration))
        # Changing starting day for next iteration, restarting day count when one year has been surpased
        day = pd.Timestamp(day) + pd.Timedelta('1d')
        d = d+1

    elapsed_time = datetime.datetime.now() - now
    # Saving global iterative results
    if constant_results:
        Ben_acc_ideal = (sum(Results_ideal) / 1000000)
        Ben_acc_direct = (sum(Results_direct) / 1000000)
        Ben_acc_mean = (sum(Results_mean)/1000000)
        En_acc_ideal = (sum(Energy_ideal))
        En_acc_direct = (sum(Energy_direct))
        En_acc_mean = (sum(Energy_mean))
    Daily_Bens_ES.append(Results_ES)
    Daily_Bens_ETR.append(Results_ETR)
    Daily_Bens_ES_ETR.append(Results_ES_ETR)
    Daily_Bens_mean.append(Results_mean)
    Daily_Bens_direct.append(Results_direct)
    Daily_Bens_ideal.append(Results_ideal)
    Deg_accs_ES.append(deg_acc_ES)
    Deg_accs_ETR.append(deg_acc_ETR)
    Deg_accs_ES_ETR.append(deg_acc_ES_ETR)
    Deg_accs_mean.append(deg_acc_mean)
    Deg_accs_direct.append(deg_acc_direct)
    Deg_accs_ideal.append(deg_acc_ideal)
    Ben_acc_ES.append(sum(Results_ES)/1000000)
    Ben_acc_ETR.append(sum(Results_ETR)/1000000)
    Ben_acc_ES_ETR.append(sum(Results_ES_ETR)/1000000)
    En_acc_ES.append(sum(Energy_ES))
    En_acc_ETR.append(sum(Energy_ETR))
    En_acc_ES_ETR.append(sum(Energy_ES_ETR))
    # Disabling constants results run (DEACTIVATED)
    constant_results = True
    print("*********************************************************************************")
    print("Evaluation for  K={}".format(K))
    print("Total elapsed time: {}".format(elapsed_time))
    print("Number of days evaluated: {}".format(d))
    print("Ideal benefits: {}€".format(round(Ben_acc_ideal*1000000, 2)))
    print("Ideal circulated energy: {}MWh".format(round(En_acc_ideal, 2)))
    print("With mean criteria:")
    print("\t -Benefits are {}€, circulated energy is {} MWh"
          .format(round(Ben_acc_mean*1000000, 2), round(En_acc_mean, 2)))
    print("With direct forecast:")
    print("\t -Benefits are {}€, circulated energy is {} MWh"
          .format(round(Ben_acc_direct*1000000, 2), round(En_acc_direct, 2)))
    print("With ES criteria:")
    print("\t -Benefits are {}€, circulated energy is {} MWh"
          .format(round(Ben_acc_ES[-1]*1000000, 2), round(En_acc_ES[-1], 2)))
    print("With ETR criteria:")
    print("\t -Benefits are {}€, circulated energy is {} MWh"
          .format(round(Ben_acc_ETR[-1]*1000000, 2), round(En_acc_ETR[-1], 2)))
    print("With ES & ETR criteria:")
    print("\t -Benefits are {}€, circulated energy is {} MWh"
          .format(round(Ben_acc_ES_ETR[-1]*1000000, 2), round(En_acc_ES_ETR[-1], 2)))

print('K evaluation global elapsed time: {}'.format(datetime.datetime.now() - k_eval_global_start))
#%% Plotting and saving results with various Ks
if len(Ks) != 1:
    # Extrapolating benefits
    from NPV_evaluator import extrapolate_CF, NPV_calculator
    NPVs_ES = []
    NPVs_ETR = []
    NPVs_ES_ETR = []
    for i in range(len(Ks)):
        CF_ES_b, CF_acc_ES_K = extrapolate_CF(Daily_Bens_ES[i], Deg_accs_ES[i], False)
        CF_ETR_b, CF_acc_ETR_K = extrapolate_CF(Daily_Bens_ETR[i], Deg_accs_ETR[i], False)
        CF_ES_ETR_b, CF_acc_ES_ETR_K = extrapolate_CF(Daily_Bens_ES_ETR[i], Deg_accs_ES_ETR[i], False)
        CF_mean_b, CF_acc_mean = extrapolate_CF(Daily_Bens_mean[i], Deg_accs_mean[i], False)
        CF_direct_b, CF_acc_direct = extrapolate_CF(Daily_Bens_direct[i], Deg_accs_direct[i], False)
        CF_ideal_b, CF_acc_ideal = extrapolate_CF(Daily_Bens_ideal[i], Deg_accs_ideal[i], False)
        r = 7 / 100
        ideal_NPV_b = NPV_calculator(CF_ideal_b, (1 + r) ** (1 / 365) - 1, 'ideal')
        ES_NPV_b = NPV_calculator(CF_ES_b, (1 + r) ** (1 / 365) - 1, 'ES')
        ETR_NPV_b = NPV_calculator(CF_ETR_b, (1 + r) ** (1 / 365) - 1, 'ETR')
        ES_ETR_NPV_b = NPV_calculator(CF_ES_ETR_b, (1 + r) ** (1 / 365) - 1, 'ES & ETR')
        mean_NPV_b = NPV_calculator(CF_mean_b, (1 + r) ** (1 / 365) - 1, 'mean')
        direct_NPV_b = NPV_calculator(CF_direct_b, (1 + r) ** (1 / 365) - 1, 'direct')
        NPVs_ES.append(ES_NPV_b)
        NPVs_ETR.append(ETR_NPV_b)
        NPVs_ES_ETR.append(ES_ETR_NPV_b)
    plt.figure('Grid Search Results')
    plt.plot(Ks, NPVs_ES, label='ES', marker="8")
    plt.plot(Ks, NPVs_ETR, label='ETR', marker = "^")
    plt.plot(Ks, NPVs_ES_ETR, label='ES & ETR', marker = "s")
    plt.axhline(y=mean_NPV_b, color='g', linestyle='-', label='SAA')
    plt.axhline(y=ideal_NPV_b, color='r', linestyle='-', label='Ideal')
    plt.axhline(y=direct_NPV_b , color='b', linestyle='--', label='Deterministic')
    plt.xlabel('K')
    plt.ylabel('NPV (€)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('Analysis/K_gridsearch.png')
    plt.savefig('Analysis/K_gridsearch.eps')
    plt.savefig('Analysis/K_gridsearch.svg')
    figurecount = figurecount + 1

#%% Specific results save and plot for only 1 K value
if len(Ks) == 1:
    np.save(analysis_folder+'/Results_ES.npy', Results_ES)
    np.save(analysis_folder+'/Results_mean.npy', Results_mean)
    np.save(analysis_folder+'/Results_ETR.npy', Results_ETR)
    np.save(analysis_folder+'/Results_ES_ETR.npy', Results_ES_ETR)
    np.save(analysis_folder+'/Results_direct.npy', Results_direct)
    np.save(analysis_folder+'/Results_ideal.npy', Results_ideal)
    np.save(analysis_folder+'/Energy_ES.npy', Energy_ES)
    np.save(analysis_folder+'/Energy_mean.npy', Energy_mean)
    np.save(analysis_folder+'/Energy_ETR.npy', Energy_ETR)
    np.save(analysis_folder+'/Energy_ES_ETR.npy', Energy_ES_ETR)
    np.save(analysis_folder+'/Energy_ideal.npy', Energy_ideal)
    np.save(analysis_folder+'/Energy_direct.npy', Energy_direct)
    np.save(analysis_folder+'/deg_acc_ES.npy', deg_acc_ES)
    np.save(analysis_folder+'/deg_acc_ETR.npy', deg_acc_ETR)
    np.save(analysis_folder+'/deg_acc_ES_ETR.npy', deg_acc_ES_ETR)
    np.save(analysis_folder+'/deg_acc_mean.npy', deg_acc_mean)
    np.save(analysis_folder+'/deg_acc_direct.npy', deg_acc_direct)
    np.save(analysis_folder+'/deg_acc_ideal.npy', deg_acc_ideal)

    # Mining data
    # Accumulated benefits
    ideal_benefits_accumulator = 0
    ideal_benefits_acc = [0]
    ES_benefits_accumulator = 0
    ES_benefits_acc = [0]
    ETR_benefits_accumulator = 0
    ETR_benefits_acc = [0]
    ES_ETR_benefits_accumulator = 0
    ES_ETR_benefits_acc = [0]
    mean_benefits_accumulator = 0
    mean_benefits_acc = [0]
    direct_benefits_accumulator = 0
    direct_benefits_acc = [0]
    # Circulated energy
    ideal_energy_accumulator = 0
    ideal_energy_acc = [0]
    ES_energy_accumulator = 0
    ES_energy_acc = [0]
    ETR_energy_accumulator = 0
    ETR_energy_acc = [0]
    ES_ETR_energy_accumulator = 0
    ES_ETR_energy_acc = [0]
    mean_energy_accumulator = 0
    mean_energy_acc = [0]
    direct_energy_accumulator = 0
    direct_energy_acc = [0]
    # Creating arrays of accumulated benefits & circulated energy
    for i in range(d):
        # Accumulated benefits
        ideal_benefits_acc.append(ideal_benefits_acc[i] + Results_ideal[i])
        ES_benefits_acc.append(ES_benefits_acc[i] + Results_ES[i])
        ETR_benefits_acc.append(ETR_benefits_acc[i] + Results_ETR[i])
        ES_ETR_benefits_acc.append(ES_ETR_benefits_acc[i] + Results_ES_ETR[i])
        mean_benefits_acc.append(mean_benefits_acc[i] + Results_mean[i])
        direct_benefits_acc.append(direct_benefits_acc[i] + Results_direct[i])
        # Circulated energy
        ideal_energy_acc.append(ideal_energy_acc[i] + Energy_ideal[i])
        ES_energy_acc.append(ES_energy_acc[i] + Energy_ES[i])
        ETR_energy_acc.append(ETR_energy_acc[i] + Energy_ETR[i])
        ES_ETR_energy_acc.append(ES_ETR_energy_acc[i] + Energy_ES_ETR[i])
        mean_energy_acc.append(mean_energy_acc[i] + Energy_mean[i])
        direct_energy_acc.append(direct_energy_acc[i] + Energy_direct[i])

    np.save(analysis_folder+'/ideal_benefits_acc.npy', ideal_benefits_acc)
    np.save(analysis_folder+'/ideal_energy_acc.npy', ideal_energy_acc)
    np.save(analysis_folder+'/ES_benefits_acc.npy', ES_benefits_acc)
    np.save(analysis_folder+'/ES_energy_acc.npy', ES_energy_acc)
    np.save(analysis_folder+'/ETR_benefits_acc.npy', ETR_benefits_acc)
    np.save(analysis_folder+'/ETR_energy_acc.npy', ETR_energy_acc)
    np.save(analysis_folder+'/ES_ETR_benefits_acc.npy', ES_ETR_benefits_acc)
    np.save(analysis_folder+'/ES_ETR_energy_acc.npy', ES_ETR_energy_acc)
    np.save(analysis_folder+'/mean_benefits_acc.npy', mean_benefits_acc)
    np.save(analysis_folder+'/mean_energy_acc.npy', mean_energy_acc)
    np.save(analysis_folder+'/direct_benefits_acc.npy', direct_benefits_acc)
    np.save(analysis_folder+'/direct_energy_acc.npy', direct_energy_acc)
    #Plotting accumulated benefits
    plt.figure('Accumulated benefits')
    plt.plot(ES_benefits_acc, label='ES')
    plt.plot(ETR_benefits_acc, label='ETR')
    plt.plot(ES_ETR_benefits_acc, label='ES & ETR')
    plt.plot(mean_benefits_acc, label='SAA')
    plt.plot(direct_benefits_acc, label='Deterministic')
    plt.plot(ideal_benefits_acc,'--', label='Ideal')
    plt.xlabel('Day')
    plt.ylabel('Accumulated benefits (€)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(analysis_folder+'/accumulated_benefits.png')
    plt.savefig(analysis_folder+'/accumulated_benefits.eps')
    figurecount = figurecount + 1
    # Plotting accumulated energy
    plt.figure('Accumulated energy')
    plt.plot(ES_energy_acc, label='ES')
    plt.plot(ETR_energy_acc, label='ETR')
    plt.plot(ES_ETR_energy_acc, label='ES & ETR')
    plt.plot(mean_energy_acc, label='SAA')
    plt.plot(direct_energy_acc, label='Deterministic')
    plt.plot(ideal_energy_acc, '--', label='Ideal')
    plt.xlabel('Day')
    plt.ylabel('Circulated energy (MWh)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(analysis_folder+'/accumulated_energy.png')
    plt.savefig(analysis_folder+'/accumulated_energy.eps')
    figurecount = figurecount + 1
    # Plotting accumulated degradation
    plt.figure('Accumulated degradation')
    plt.plot([a*100 for a in deg_acc_ES], label='ES')
    plt.plot([a*100 for a in deg_acc_ETR], label='ETR')
    plt.plot([a*100 for a in deg_acc_ES_ETR], label='ES & ETR')
    plt.plot([a*100 for a in deg_acc_mean], label='SAA')
    plt.plot([a*100 for a in deg_acc_direct], label='Deterministic')
    plt.plot([a*100 for a in deg_acc_ideal], '--', label='Ideal')
    plt.xlabel('Day')
    plt.ylabel('Accumulated degradation (%)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(analysis_folder+'/accumulated_deg.png')
    plt.savefig(analysis_folder+'/accumulated_deg.eps')
    figurecount = figurecount + 1
    # Plotting accumulated benefits vs accumulated energy
    plt.figure('Accumulated benefits vs accumulated energy')
    plt.plot(ES_energy_acc, ES_benefits_acc, label='ES')
    plt.plot(ETR_energy_acc, ETR_benefits_acc, label='ETR')
    plt.plot(ES_ETR_energy_acc, ES_ETR_benefits_acc, label='ES & ETR')
    plt.plot(mean_energy_acc, mean_benefits_acc, label='SAA')
    plt.plot(direct_energy_acc, direct_benefits_acc,'--', label='Deterministic')
    # plt.plot(ideal_energy_acc, ideal_benefits_acc,'--', label='Ideal')
    plt.ylabel('Accumulated benefits (€)')
    plt.xlabel('Circulated energy (MWh)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(analysis_folder+'/accumulated_benefits_vs_energy.png')
    plt.savefig(analysis_folder+'/accumulated_benefits_vs_energy.eps')
    figurecount = figurecount + 1
    # Plotting accumulated benefits vs accumulated degradation
    plt.figure('Accumulated benefits vs accumulated degradation')
    plt.plot([a*100 for a in deg_acc_ES], [a/1000000 for a in ES_benefits_acc], label='ES')
    plt.plot([a*100 for a in deg_acc_ETR], [a/1000000 for a in ETR_benefits_acc], label='ETR')
    plt.plot([a*100 for a in deg_acc_ES_ETR], [a/1000000 for a in ES_ETR_benefits_acc], label='ES & ETR')
    plt.plot([a*100 for a in deg_acc_mean], [a/1000000 for a in mean_benefits_acc],label='SAA')
    plt.plot([a*100 for a in deg_acc_direct],[a/1000000 for a in direct_benefits_acc],'--', label='Deterministic')
    # plt.plot([a*100 for a in deg_acc_ideal], ideal_benefits_acc,'--', label='Ideal')
    plt.ylabel('Accumulated benefits (M€)')
    plt.xlabel('Accumulated degradation (%)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(analysis_folder+'/accumulated_benefits_vs_deg.png')
    plt.savefig(analysis_folder+'/accumulated_benefits_vs_deg.eps')
    figurecount = figurecount + 1






