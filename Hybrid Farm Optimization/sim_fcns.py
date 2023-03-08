#%% Importing libraries
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from matplotlib import pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import time
#%% Importing external files
from arb_fcns import *
from aux_fcns import *
from plot_fcns import *
from rt_fcns import *
#%% Running simulation
def simulator_runner(HyF_Parameters, case_name, sim_name, starting_day, ending_day, Results, sim_time):
    # Creating output folder
    sim_folder = 'Results/' + sim_time.strftime("%m_%d_%H_%M_") + sim_name
    output_folder = sim_folder + '/' + case_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Initializing
    sim_timer = time.time()
    Case_Results = {}
    day = pd.Timestamp(starting_day)
    SOC_day_prev = 0
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'Running case: "{case_name}"')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    while day != pd.Timestamp(ending_day) + pd.Timedelta('1d'):
        Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()
        # Initializing daily run
        Daily_results = {}
        day_str = day.strftime("%Y-%m-%d")
        Dataset = Global_dataset[day_str]
        day_timer = time.time()
        Pgen_real = Dataset['Pgen_real']
        print(f'Running {day_str}')
        # Creating daily output folder (if daily plotting is enabled)
        if HyF_Parameters['Config']['Daily plotting']:
            daily_output_folder = output_folder + f'/{day_str}'
            if not os.path.exists(daily_output_folder):
                os.makedirs(daily_output_folder)
        # Matching day length input lengths when hours are != 24 (only March the 25th)
        lenmin = min(len(Dataset['Price_pred_DM']), len(Dataset['Price_real_ID2']))
        lencut = 24 - lenmin
        Dataset['Price_pred_DM'] = Dataset['Price_pred_DM'][0:lenmin]
        Dataset['Price_real_DM'] = Dataset['Price_real_DM'][0:lenmin]
        Dataset['Price_real_ID2'] = Dataset['Price_real_ID2'][0:lenmin]
        Dataset['Pgen_pred_DM'] = Dataset['Pgen_pred_DM'][0:lenmin]
        Dataset['Pgen_pred_ID2'] = Dataset['Pgen_pred_ID2'][0:lenmin]
        Dataset['Price_real_ID3'] = Dataset['Price_real_ID3'][:len(Dataset['Price_real_ID3']) - lencut]
        Dataset['Price_real_ID4'] = Dataset['Price_real_ID4'][:len(Dataset['Price_real_ID4']) - lencut]
        Dataset['Price_real_ID5'] = Dataset['Price_real_ID5'][:len(Dataset['Price_real_ID5']) - lencut]
        Dataset['Price_real_ID6'] = Dataset['Price_real_ID6'][:len(Dataset['Price_real_ID6']) - lencut]
        Dataset['Pgen_real'] = Dataset['Pgen_real'][0:lenmin]
        P_ID_Purchs = list(np.zeros(lenmin))
        # Obtaining deviation costs
        Dev_Costs_Down, Dev_Costs_Up = \
            Dev_Costs_Calc(Dataset['Dev_against_way'], Dataset['Dev_against_coef'], sense='real')
        Dev_Costs_Down_pred, Dev_Costs_Up_pred = \
            Dev_Costs_Calc(Dataset['Dev_against_way'], Dataset['Dev_against_coef'], sense='pred')

        # Setting initial SOC at 0 for this market
        HyF_Parameters['ESS Initial SOC'] = 0
        # Daily market bidding
        if HyF_Parameters['Config']['Ideal']:
            Gen_Pred_DM, WTG_Psold_Pred_DM, ESS_C_Pred_DM, ESS_D_DM, ESS_P_DM, SOC_Pred_DM \
                = DM(HyF_Parameters, Pgen_real, Dataset['Price_real_DM']) # Generating schedule
        if not HyF_Parameters['Config']['Ideal']:
            Gen_Pred_DM, WTG_Psold_Pred_DM, ESS_C_Pred_DM, ESS_D_DM, ESS_P_DM, SOC_Pred_DM \
                = DM(HyF_Parameters, Dataset['Pgen_pred_DM'], Dataset['Price_pred_DM']) # Generating schedule
        P_PCC_toDel = PCC_Powers(WTG_Psold_Pred_DM, ESS_D_DM, ESS_P_DM)     # PCC Power to Deliver
        P_Sch_DM = P_PCC_toDel.copy()       # Saving power commited to DM market separately for benefit calculations
        if HyF_Parameters['Config']['Daily plotting']:
            plot_dm(Dataset['Price_pred_DM'], Gen_Pred_DM, WTG_Psold_Pred_DM, ESS_C_Pred_DM, ESS_D_DM, ESS_P_DM,
                SOC_Pred_DM, f'Expected operation {day_str}', daily_output_folder)  # Generating plot
        if HyF_Parameters['Config']['ID Participation']:
            # --  Intraday market 2 bidding --
            # Getting real SOC just after market session
            HyF_Parameters['ESS Initial SOC'] = SOC_day_prev
            # Initializing arrays
            ESS_P_Prev_ID2 = ESS_P_DM       # Initializing previous ESS purchases array
            Dataset['Pgen_pred_ID2'][0] = Dataset['Pgen_real'][0]   # Filling Pgen array first slot
            # Programming SOC surplus dump
            if day != pd.Timestamp(starting_day):
                P_SOCdump_ID2 = SOCdump_ID(HyF_Parameters, Dataset['Price_real_ID2'], 2)
                # Updating scheduled powers with new commitment caused by SOC dump
                P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, P_SOCdump_ID2, 0, 4) # PCC real powers to deliver
                # Generating ID2 expected operation
                Gen_Pred_ID2, WTG_Psold_Pred_ID2, WTG_Pdel_Pred_ID2, ESS_C_Pred_ID2, ESS_D_Pred_ID2, ESS_P_ID2, \
                ESS_S_ID2, SOC_Pred_ID2, Purch_ID2 =  \
                    ID(HyF_Parameters, Dataset['Pgen_pred_ID2'], P_PCC_toDel, Dataset['Price_real_DM'],
                       Dataset['Price_real_ID2'].reshape(len(Dataset['Price_real_ID2']),), Dev_Costs_Down_pred,
                       Dev_Costs_Up_pred, ESS_P_Prev_ID2, P_SOCdump_ID2)
                # Generating and saving plot
                if HyF_Parameters['Config']['Daily plotting']:
                    plot_id(Purch_ID2, Gen_Pred_DM, Gen_Pred_ID2, WTG_Psold_Pred_ID2, ESS_C_Pred_ID2,
                        ESS_D_Pred_ID2, ESS_S_ID2, SOC_Pred_ID2, HyF_Parameters,
                        f'ID 2 Operation {day_str}', daily_output_folder, P_SOCdump_ID2, P_PCC_toDel)
            if day == pd.Timestamp(starting_day):
                P_SOCdump_ID2 = list(np.zeros(lenmin))
                Gen_Pred_ID2 = Gen_Pred_DM
                ESS_P_ID2 = list(np.zeros(lenmin))
                Purch_ID2 = list(np.zeros(lenmin))
                WTG_Psold_Pred_ID2 = list(np.zeros(lenmin))
                ESS_S_ID2 = list(np.zeros(lenmin))
            # Updating PCC power to deliver  & ID purchases list
            P_ID_Purchs = Purch_ID2
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, [-a for a in Purch_ID2], 0, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, ESS_S_ID2, 0, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, WTG_Psold_Pred_ID2, 0, lenmin)

            # -- Intraday market 3 bidding --
            # Initializing arrays
            ID_len = len(Dataset['Price_real_ID3'])   # Obtaining market sesion length
            ESS_P_Prev_ID3 = ESS_P_ID2[-ID_len:]      # Previous ESS purchase powers
            # Obtaining real PCC output power (P_PCC_Real) and expected SOC before market delivery
            SOC_Prev_ID3, P_ESS_Schm, \
            P_ESS_Real, P_PCC_Real = RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters)
            HyF_Parameters['ESS Initial SOC'] = SOC_Prev_ID3[4]  # Changing initial SOC before ID3
            # Dumping SOC surplus before ID4 starts
            P_SOCdump_ID3 = SOCdump_ID(HyF_Parameters, Dataset['Price_real_ID3'], 3)
            # Updating scheduled powers with new commitment caused by SOC dum
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, P_SOCdump_ID3, 4, 7)
            # Generating ID3 expected operation
            Gen_Pred_ID3, WTG_Psold_Pred_ID3, WTG_Pdel_Pred_ID3, ESS_C_Pred_ID3, ESS_D_Pred_ID3, ESS_P_ID3, \
            ESS_S_ID3, SOC_Pred_ID3, Purch_ID3 =  \
                    ID(HyF_Parameters, Dataset['Pgen_pred_ID3'][-ID_len:], P_PCC_toDel[-ID_len:],
                   Dataset['Price_real_DM'][-ID_len:],
                   Dataset['Price_real_ID3'].reshape(ID_len, ), Dev_Costs_Down_pred[-ID_len:],
                   Dev_Costs_Up_pred[-ID_len:], ESS_P_Prev_ID3, P_SOCdump_ID3)
            # Generating and saving plot
            if HyF_Parameters['Config']['Daily plotting']:
                plot_id(Purch_ID3, Gen_Pred_ID2[-ID_len:], Gen_Pred_ID3, WTG_Psold_Pred_ID3, ESS_C_Pred_ID3,
                    ESS_D_Pred_ID3, ESS_S_ID3, SOC_Pred_ID3, HyF_Parameters,
                    f'ID 3 Operation {day_str}', daily_output_folder, P_SOCdump_ID3, P_PCC_toDel[-ID_len:])
            # Updating PCC power to deliver & ID purchases arrays
            P_ID_Purchs = PCC_Sch_updater(P_ID_Purchs, Purch_ID3, 4, lenmin-1)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, [-a for a in Purch_ID3], 4, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, ESS_S_ID3, 4, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, WTG_Psold_Pred_ID3, 4, lenmin)

            # -- Intraday market 4 bidding --
            # Initializing arrays
            ID_len = (len(Dataset['Price_real_ID4']))
            ESS_P_Prev_ID4 = ESS_P_ID3[-ID_len:]
            # Obtaining real operation and SOC before market session
            SOC_Prev_ID4, P_ESS_Schm, \
            P_ESS_Real, P_PCC_Real = RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters)  # Getting SOCs before ID4
            HyF_Parameters['ESS Initial SOC'] = SOC_Prev_ID4[7]  # Changing initial SOC before ID4
            # Dumping SOC surplus
            P_SOCdump_ID4 = SOCdump_ID(HyF_Parameters, Dataset['Price_real_ID4'], 4)
            # Updating scheduled powers with new commitment caused by SOC dump
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, P_SOCdump_ID4, 7, 12)
            # Generating ID4 expected operation
            Gen_Pred_ID4, WTG_Psold_Pred_ID4, WTG_Pdel_Pred_ID4, ESS_C_Pred_ID4, ESS_D_Pred_ID4, ESS_P_ID4, \
            ESS_S_ID4, SOC_Pred_ID4, Purch_ID4 =  \
                    ID(HyF_Parameters, Dataset['Pgen_pred_ID4'][-ID_len:], P_PCC_toDel[-ID_len:],
                   Dataset['Price_real_DM'][-ID_len:], Dataset['Price_real_ID4'].reshape(ID_len, ),
                   Dev_Costs_Down_pred[-ID_len:],Dev_Costs_Up_pred[-ID_len:], ESS_P_Prev_ID4, P_SOCdump_ID4)
            # Generating and saving plot
            if HyF_Parameters['Config']['Daily plotting']:
                plot_id(Purch_ID4, Gen_Pred_ID3[-ID_len:], Gen_Pred_ID4, WTG_Psold_Pred_ID4, ESS_C_Pred_ID4,
                    ESS_D_Pred_ID4, ESS_S_ID4, SOC_Pred_ID4, HyF_Parameters,
                    f'ID 4 Operation {day_str}', daily_output_folder, P_SOCdump_ID4, P_PCC_toDel[-ID_len:])
            # Updating PCC power to deliver & ID purchases arrays
            P_ID_Purchs = PCC_Sch_updater(P_ID_Purchs, Purch_ID4, 7, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, [-a for a in Purch_ID4], 7, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, ESS_S_ID4, 7, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, WTG_Psold_Pred_ID4, 7, lenmin)

            # -- Intraday market 5 bidding --
            # Initializing arrays
            ID_len = len(Dataset['Price_real_ID5'])
            ESS_P_Prev_ID5 = ESS_P_ID4[-ID_len:]
            # Obtaining real operation and SOC before market session
            SOC_Prev_ID5, P_ESS_Schm, \
            P_ESS_Real, P_PCC_Real = RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters)  # Getting SOCs before ID5
            HyF_Parameters['ESS Initial SOC'] = SOC_Prev_ID5[12]  # Changing initial SOC before ID5
            # Dumping SOC surplus
            P_SOCdump_ID5 = SOCdump_ID(HyF_Parameters, Dataset['Price_real_ID5'], 5)
            # Updating scheduled powers with new commitment caused by SOC dump
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, P_SOCdump_ID5, 11, 15)
            # Generating ID5 expected operation
            Gen_Pred_ID5, WTG_Psold_Pred_ID5, WTG_Pdel_Pred_ID5, ESS_C_Pred_ID5, ESS_D_Pred_ID5, ESS_P_ID5, \
            ESS_S_ID5, SOC_Pred_ID5, Purch_ID5 =  \
                    ID(HyF_Parameters, Dataset['Pgen_pred_ID5'][-ID_len:], P_PCC_toDel[-ID_len:],
                   Dataset['Price_real_DM'][-ID_len:], Dataset['Price_real_ID5'].reshape(ID_len, ),
                   Dev_Costs_Down_pred[-ID_len:], Dev_Costs_Up_pred[-ID_len:], ESS_P_Prev_ID5, P_SOCdump_ID5)
            # Generating and saving plot
            if HyF_Parameters['Config']['Daily plotting']:
                plot_id(Purch_ID5, Gen_Pred_ID4[-ID_len:], Gen_Pred_ID5, WTG_Psold_Pred_ID5, ESS_C_Pred_ID5,
                    ESS_D_Pred_ID5, ESS_S_ID5, SOC_Pred_ID5, HyF_Parameters,
                    f'ID 5 Operation {day_str}', daily_output_folder, P_SOCdump_ID5, P_PCC_toDel[-ID_len:])
            # Updating PCC power to deliver & ID purchases arrays
            P_ID_Purchs = PCC_Sch_updater(P_ID_Purchs, Purch_ID5, 12, 23)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, [-a for a in Purch_ID5], 11, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, ESS_S_ID5, 11, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, WTG_Psold_Pred_ID5, 11, lenmin)

            # -- Intraday market 6 bidding --
            # Initializing arrays
            ID_len = len(Dataset['Price_real_ID6'])
            ESS_P_Prev_ID6 = ESS_P_ID5[-ID_len:]
            # Obtaining real operation and SOC before market session
            SOC_Prev_ID6, P_ESS_Schm, \
            P_ESS_Real, P_PCC_Real = RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters)  # Getting SOCs before ID5
            HyF_Parameters['ESS Initial SOC'] = SOC_Prev_ID6[15]  # Changing initial SOC before ID5
            # Dumping SOC surplus
            P_SOCdump_ID6 = SOCdump_ID(HyF_Parameters, Dataset['Price_real_ID6'], 6)
            # Updating scheduled powers with new commitment caused by SOC dump
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, P_SOCdump_ID6, 15, lenmin)
            # Generating ID6 expected operation
            Gen_Pred_ID6, WTG_Psold_Pred_ID6, WTG_Pdel_Pred_ID6, ESS_C_Pred_ID6, ESS_D_Pred_ID6, ESS_P_ID6, \
            ESS_S_ID6, SOC_Pred_ID6, Purch_ID6 =  \
                    ID(HyF_Parameters, Dataset['Pgen_pred_ID6'][-ID_len:], P_PCC_toDel[-ID_len:],
                   Dataset['Price_real_DM'][-ID_len:], Dataset['Price_real_ID6'].reshape(ID_len,),
                   Dev_Costs_Down_pred[-ID_len:], Dev_Costs_Up_pred[-ID_len:], ESS_P_Prev_ID6, P_SOCdump_ID6)
            # Generating and saving plot
            if HyF_Parameters['Config']['Daily plotting']:
                plot_id(Purch_ID6, Gen_Pred_ID5[-ID_len:], Gen_Pred_ID6, WTG_Psold_Pred_ID6, ESS_C_Pred_ID6,
                    ESS_D_Pred_ID6, ESS_S_ID6, SOC_Pred_ID6, HyF_Parameters,
                    f'ID 6 Operation {day_str}', daily_output_folder, P_SOCdump_ID6, P_PCC_toDel[-ID_len:])
            # Updating PCC power to deliver & ID purchases arrays
            P_ID_Purchs = PCC_Sch_updater(P_ID_Purchs, Purch_ID6, 15, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, [-a for a in Purch_ID6], 15, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, ESS_S_ID6, 15, lenmin)
            P_PCC_toDel = PCC_Sch_updater(P_PCC_toDel, WTG_Psold_Pred_ID6, 15, lenmin)

        # Generating predictions plot
        if HyF_Parameters['Config']['Daily plotting']:
            windspe_pred_plot(day_str, Global_dataset, daily_output_folder)
            price_pred_plot(day_str, Global_dataset, daily_output_folder)

        # -- Real-time operation --
        # Running real-time operation of complete day to obtain real PCC powers
        HyF_Parameters['ESS Initial SOC'] = SOC_day_prev
        SOC, P_ESS_Sch, P_ESS_Real, P_PCC_Real = RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters)
        # Restarting power commitment list with ID purchases
        P_PCC_toDel_withIDpurchs = [a + b for a,b in zip(P_PCC_toDel, P_ID_Purchs)]
        # Analyzing real-time operation
        ESS_E_Real = sum([abs(Power) for Power in P_ESS_Real])  # Calculating net energy cycled by ESS
        Dataset['Pgen_pred_DM'] = [i for i in Dataset['Pgen_pred_DM'] if i] # Removing nones
        Pgen_error = np.mean([a - b for a, b in zip(Dataset['Pgen_real'], Dataset['Pgen_pred_DM'])]) # Forecast error
        # Plotting real-time operation
        if HyF_Parameters['Config']['Daily plotting']:
            rt_plot(P_PCC_toDel_withIDpurchs,P_PCC_Real, Dataset['Pgen_real'], P_ESS_Sch, P_ESS_Real, P_ID_Purchs, SOC,
                    f'Real-time operation {day_str}', daily_output_folder)
        # Obtaining expected benefits
        Ben_Exp_DM = Ben_Exp_Calc(P_Sch_DM, Dataset['Price_real_DM'])           # On DM market
        if HyF_Parameters['Config']['ID Participation']:
            P_todel_ID2 = [a + b + c for a,b,c in zip(P_SOCdump_ID2, WTG_Psold_Pred_ID2, ESS_S_ID2)]
            Ben_Exp_ID2 = Ben_Exp_Calc(P_todel_ID2, Dataset['Price_real_ID2'])    # On ID2
            P_todel_ID3 = [a + b + c for a,b,c in zip(P_SOCdump_ID3, WTG_Psold_Pred_ID3, ESS_S_ID3)]
            Ben_Exp_ID3 = Ben_Exp_Calc(P_todel_ID3, Dataset['Price_real_ID3'])    # On ID3
            P_todel_ID4 = [a + b + c for a,b,c in zip(P_SOCdump_ID4, WTG_Psold_Pred_ID4, ESS_S_ID4)]
            Ben_Exp_ID4 = Ben_Exp_Calc(P_todel_ID4, Dataset['Price_real_ID4'])    # On ID4
            P_todel_ID5 = [a + b + c for a,b,c in zip(P_SOCdump_ID5, WTG_Psold_Pred_ID5, ESS_S_ID5)]
            Ben_Exp_ID5 = Ben_Exp_Calc(P_todel_ID5, Dataset['Price_real_ID5'])    # On ID5
            P_todel_ID6 = [a + b + c for a,b,c in zip(P_SOCdump_ID6, WTG_Psold_Pred_ID6, ESS_S_ID6)]
            Ben_Exp_ID6 = Ben_Exp_Calc(P_todel_ID6, Dataset['Price_real_ID6'])    # On ID6
            Ben_Exp_ID= Ben_Exp_ID2 + Ben_Exp_ID3 + Ben_Exp_ID4 + Ben_Exp_ID5 + Ben_Exp_ID6
        if not HyF_Parameters['Config']['ID Participation']:
            Ben_Exp_ID = 0
        Ben_Exp = Ben_Exp_DM + Ben_Exp_ID
        # Obtaining deviations & real deviations costs (Assuming deviations costs on IDs are equal as on DM)
        Devs = [a - b for a, b in zip(P_PCC_Real, P_PCC_toDel)]
        Dev_costs = 0
        for h, dev in enumerate(Devs): # Calculating deviation costs
            if dev > 0:         # Deviation up
                Dev_costs += dev * Dataset['Price_real_DM'][h] * Dev_Costs_Up[h]
            if dev < 0:         # Deviation down
                Dev_costs -= dev * Dataset['Price_real_DM'][h] * Dev_Costs_Down[h]
        # Obtaining total ID purhcases powers & costs
        if HyF_Parameters['Config']['ID Participation']:
            ID_Purch = sum(Purch_ID2) + sum(Purch_ID3) + sum(Purch_ID4) + sum(Purch_ID5) + sum(Purch_ID6)
            ID_P_Cost = Ben_Exp_Calc(Purch_ID2, Dataset['Price_real_ID2']) + Ben_Exp_Calc(Purch_ID3,
                                                                                          Dataset['Price_real_ID3']) + \
                        Ben_Exp_Calc(Purch_ID4, Dataset['Price_real_ID4']) + Ben_Exp_Calc(Purch_ID5,
                                                                                          Dataset['Price_real_ID5']) + \
                        Ben_Exp_Calc(Purch_ID6, Dataset['Price_real_ID6'])  # Calculating ID purchases prices
        if not HyF_Parameters['Config']['ID Participation']:                # Disabling if ID is disabled
            ID_Purch = 0
            ID_P_Cost = 0
        # Obtaining relative amount of power commitments covered with ID purchases
        if sum(P_PCC_toDel) == 0:
            ID_purch_rel = 0    # Protecting division by 0 if no DM schedule is taking place
        else:
            ID_purch_rel = (ID_Purch * 100) / (sum(P_PCC_toDel_withIDpurchs) + ID_Purch)
        # Calculating real benefits
        Ben_Real = Ben_Exp - Dev_costs - ID_P_Cost
        # Calculating degradation
        Deg_cyc = deg_model(P_ESS_Real, SOC, HyF_Parameters['ESS Capacity'], HyF_Parameters['ESS EOL']) # Degradation
        Deg_cal = Deg_Cal_model(SOC)
        ESS_deg = Deg_cyc + Deg_cal
        # Disabling degradation if ESS capacity is low (no ESS case)
        if HyF_Parameters['ESS Capacity'] < 0.001:
            ESS_deg = 0

        # Printing results
        if HyF_Parameters['Config']['Daily plotting']:
            print(f'\t - Expected benefits: {round(Ben_Exp, 2)} €')
            print(f'\t - Real benefits: {round(Ben_Real, 2)} €')
            print(f'\t - Deviation costs are: {round(Dev_costs, 2)} €')
            if HyF_Parameters['Config']['ID Participation']:
                print(f'\t - Proportion of deviations solved by ID: {round(ID_purch_rel, 2)} %')
            if HyF_Parameters['Config']['ID Participation']:
                print(f'\t - ID purchases costs are: {round(ID_P_Cost, 2)} €')
            print(f'\t - Average hourly generated power prediction deviation: {round(Pgen_error, 2)} MWh')
            print(f'\t - Energy cycled by ESS: {round(ESS_E_Real, 2)} MWh')
            print(f'\t - ESS capacity loss: {round(ESS_deg * 100, 5)}%')
        # Writting daily results
        if HyF_Parameters['Config']['Daily plotting']:
            with open(daily_output_folder + '/Results.txt', 'w') as f:
                f.writelines(f'- Daily market expected benefits: {round(Ben_Exp, 2)} €')
                f.writelines(f'\n- Real benefits: {round(Ben_Real, 2)} €')
                f.writelines(f'\n- Deviation costs are: {round(Dev_costs, 2)} €')
                if HyF_Parameters['Config']['ID Participation']:
                    f.writelines(f'\n- Proportion of deviations solved by ID: {round(ID_purch_rel, 2)} %')
                if HyF_Parameters['Config']['ID Participation']:
                    f.writelines(f'\n- ID purchases costs are: {round(ID_P_Cost, 2)} €')
                f.writelines(f'\n- Average hourly generated power prediction deviation: {round(Pgen_error, 2)} MWh')
                f.writelines(f'\n- Energy cycled by ESS: {round(ESS_E_Real, 2)} MWh')
                f.writelines(f'\n- ESS capacity loss: {round(ESS_deg * 100, 5)}%')
        # Saving results
        Daily_results['Ben_DM_Exp'] = Ben_Exp
        Daily_results['Daily_Egen'] = sum(Dataset['Pgen_real'])
        Daily_results['Dev_costs'] = Dev_costs
        Daily_results['Purch_costs'] = ID_P_Cost
        Daily_results['ID_purch_rel'] = ID_purch_rel
        Daily_results['Ben_DM_Real'] = Ben_Real
        Daily_results['Pgen_error'] = Pgen_error
        Daily_results['ESS_deg'] = ESS_deg
        Daily_results['ESS_E_Real'] = ESS_E_Real
        Case_Results[f'{day_str}'] = Daily_results
        # ESS parameters for next day
        SOC_day_prev = SOC[-1]
        # Updating capacity and nominal power with degradation
        HyF_Parameters['ESS Capacity'] = HyF_Parameters['ESS Capacity'] * (1-ESS_deg)
        HyF_Parameters['ESS Nominal Power'] = HyF_Parameters['ESS Nominal Power'] * (1-ESS_deg)
        # Changing to nex day
        day = day + pd.Timedelta('1d')
        # Stopping timer
        print(f'Elapsed time: {round(time.time() - day_timer, 2)}s')

    # Finishing case run
    print(f'Finished "{case_name}" case')
    Results[f'{case_name}'] = Case_Results
    if round((time.time() - sim_timer)/3600) > 1:
        print(f'Total elapsed time: {round((time.time() - sim_timer)/3600)}h')
    else:
        print(f'Total elapsed time: {round((time.time() - sim_timer))}s')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    np.save(output_folder + '/Case_Results.npy', Case_Results)
    return Case_Results




