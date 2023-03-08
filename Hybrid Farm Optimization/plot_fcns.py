from matplotlib import pyplot as plt
import numpy as np
from aux_fcns import *
#%% Manipulating pyplot default parameters
# Forcing tight layout
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling showing figures
plt.ioff()
#%% Daily market operation
def plot_dm(prices, WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC, figurename, output_folder):
    # WTG_Pbatt = [g - s for g, s in zip(WTG_Pgen, WTG_Psold)]
    dates_label = []  # X axis dates label
    for i in range(len(prices)):  # Filling X axis dates label
        dates_label.append('{}:00'.format(i))
    x = np.arange(len(prices))

    # Initializing figure
    fig = plt.figure(figurename)  # Creating the figure
    fig.suptitle(output_folder)

    # Energy price
    price_plot = fig.add_subplot(5, 1, 1)  # Creating subplot
    ticks_x = np.arange(0, len(prices), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(prices), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(prices)])  # X axis limits
    axes.set_ylim([min(prices) * 0.9, max(prices) * 1.1])  # X axis limits
    plt.bar(ticks_x, prices, align='edge', width=1, edgecolor='black', color='r')
    plt.ylabel('Price (€/MWh)')
    plt.grid()

    # Generation
    generation_plot = fig.add_subplot(5, 1, 2)  # Creating subplot
    ticks_x = np.arange(0, len(prices), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(prices), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(prices)])  # X axis limits
    plt.bar(x + 0.00, WTG_Pgen, color='b', width=0.25, label='Generated', edgecolor='black')
    plt.bar(x + 0.25, WTG_Psold, color='g', width=0.25, label='Sent to the grid', edgecolor='black')
    plt.bar(x + 0.50, [-a for a in ESS_C], color='r', width=0.25, label='Sent to the battery', edgecolor='black')
    plt.legend()
    plt.ylabel('WTG (MW)')
    plt.grid()

    # ESS Powers
    ESS_plot = fig.add_subplot(5, 1, 3)  # Creating subplot
    ticks_x = np.arange(0, len(prices), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(prices), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(prices)])  # X axis limits
    plt.bar(x + 0.25, ESS_C, color='b', width=0.25, label='Charge', edgecolor='black')
    plt.bar(x + 0.50, ESS_P, color='g', width=0.25, label='Purchased', edgecolor='black')
    plt.bar(x + 0.75, ESS_D, color='r', width=0.25, label='Discharged', edgecolor='black')
    plt.legend()
    plt.ylabel('ESS (MW)')
    plt.grid()

    # SOC
    SOC_plot = fig.add_subplot(5, 1, 4)  # Creating subplot
    ticks_x = np.arange(0, len(prices), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(prices), 1), dates_label, rotation=45)
    plt.plot(SOC)
    axes = plt.gca()
    axes.set_xlim([0, len(prices)])  # X axis limits
    axes.set_ylim([0, 110])  # X axis limits
    plt.ylabel('SOC (%)')
    plt.grid()

    # PCC
    PCC_plot = fig.add_subplot(5, 1, 5)  # Creating subplot
    ticks_x = np.arange(0, len(prices), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(prices), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(prices)])  # X axis limits
    plt.bar(x + 0.25, WTG_Psold, color='b', width=0.25, label='From WTG', edgecolor='black')
    plt.bar(x + 0.5, [a+b for a,b in zip(ESS_D,ESS_P)], color='g', width=0.25, label='From ESS', edgecolor='black')
    plt.legend()
    plt.ylabel('PCC (MW)')
    plt.grid()
    # Launching the plot
    plt.tight_layout()
    # Saving the plot
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_folder + '/DM Expected Operation.png')
    # Closing plot
    plt.close()

#%% Intraday market operation
def plot_id(ID_Purchases, Pgen_pred, Pgen_real, Pgen_sold, ESS_C, ESS_D, ESS_S, SOC, HyF_Parameters, figurename,
            output_folder, P_SOCdump, P_PCC_toDel):
    dates_label = []  # X axis dates label
    for i in range(24-len(Pgen_pred), 24):  # Filling X axis dates label
        dates_label.append('{}:00'.format(i))
    ESS_Pnom = HyF_Parameters['ESS Nominal Power']

    # Adjusting hours to ID session window
    ID_len = len(ID_Purchases)
    hour_i = len(Pgen_pred)-ID_len
    dates_label = dates_label[-ID_len:]
    x = np.arange(ID_len)

    # Generating y axis limits
    y_max = 1.1*max(max(P_PCC_toDel), max(ID_Purchases), max(P_SOCdump), max(Pgen_pred),
                    max(Pgen_real), max(Pgen_sold), max(ESS_S))
    y_min = 1.1*min(min(P_PCC_toDel), min(ID_Purchases), min(P_SOCdump), min(ESS_D), min(Pgen_real))

    # Initializing figure
    fig = plt.figure(figurename)  # Creating the figure
    fig.suptitle(output_folder)

    # Purchased power on ID
    price_plot = fig.add_subplot(5, 1, 1)  # Creating subplot
    ticks_x = np.arange(hour_i, len(Pgen_pred), 1)  # Vertical grid spacing
    plt.xticks(np.arange(hour_i, len(Pgen_pred), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([hour_i, len(Pgen_pred)])  # X axis limits
    axes.set_ylim([-0.1, y_max])             # Y axis limits
    plt.bar(ticks_x, ID_Purchases, align='edge', width=1, edgecolor='black', color='r')
    plt.ylabel('Purch (MW)')
    plt.grid()

    # Generated power
    generation_plot = fig.add_subplot(5, 1, 2)  # Creating subplot
    ticks_x = np.arange(hour_i, len(Pgen_pred), 1)  # Vertical grid spacing
    plt.xticks(np.arange(hour_i, len(Pgen_pred), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([hour_i, len(Pgen_pred)])  # X axis limits
    axes.set_ylim([-0.1, y_max])             # Y axis limits
    if HyF_Parameters['Config']['ID Arbitrage']:
        plt.bar(ticks_x + 0.25, Pgen_pred, color='b', width=0.25, label='Previous forecast', edgecolor='black')
        plt.bar(ticks_x + 0.50, Pgen_real, color='g', width=0.25, label='Current forecast', edgecolor='black')
        plt.bar(ticks_x + 0.75, Pgen_sold, color='r', width=0.25, label='Sold to ID', edgecolor='black')
    if not HyF_Parameters['Config']['ID Arbitrage']:
        plt.bar(ticks_x + 0.25, Pgen_pred, color='b', width=0.25, label='Previous forecast', edgecolor='black')
        plt.bar(ticks_x + 0.50, Pgen_real, color='g', width=0.25, label='Current forecast', edgecolor='black')
    plt.legend()
    plt.ylabel('Gen (MW)')
    plt.grid()

    # ESS Powers
    ESS_plot = fig.add_subplot(5, 1, 3)  # Creating subplot
    ticks_x = np.arange(hour_i, len(Pgen_pred), 1)  # Vertical grid spacing
    plt.xticks(np.arange(hour_i, len(Pgen_pred), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([hour_i, len(Pgen_pred)])  # X axis limits
    axes.set_ylim([y_min, y_max])             # Y axis limits
    if HyF_Parameters['Config']['ID Arbitrage']:
        plt.bar(ticks_x + 0.19, ESS_C, color='b', width=0.2, label='Charge', edgecolor='black')
        plt.bar(ticks_x + 0.39, P_SOCdump, color='g', width=0.2, label='Dumped', edgecolor='black')
        plt.bar(ticks_x + 0.59, ESS_D, color='r', width=0.2, label='Discharged', edgecolor='black')
        plt.bar(ticks_x + 0.79, ESS_S, color='y', width=0.2, label='Sold', edgecolor='black')
    if not HyF_Parameters['Config']['ID Arbitrage']:
        plt.bar(ticks_x + 0.25, ESS_C, color='b', width=0.25, label='Charge', edgecolor='black')
        plt.bar(ticks_x + 0.50, P_SOCdump, color='g', width=0.25, label='Dumped', edgecolor='black')
        plt.bar(ticks_x + 0.75, ESS_D, color='r', width=0.25, label='Discharged', edgecolor='black')
    plt.legend()
    plt.ylabel('ESS (MW)')
    plt.grid()

    # SOC
    SOC_plot = fig.add_subplot(5, 1, 4)  # Creating subplot
    ticks_x = np.arange(hour_i, len(Pgen_pred), 1)  # Vertical grid spacing
    plt.xticks(np.arange(hour_i, len(Pgen_pred), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([hour_i, len(Pgen_pred)])  # X axis limits
    axes.set_ylim([0, 110])  # X axis limits
    plt.plot(SOC)
    plt.ylabel('SOC (%)')
    plt.grid()

    # PCC Powers
    PCC_plot = fig.add_subplot(5, 1, 5)  # Creating subplot
    ticks_x = np.arange(hour_i, len(Pgen_pred), 1)  # Vertical grid spacing
    plt.xticks(np.arange(hour_i, len(Pgen_pred), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([hour_i, len(Pgen_pred)])  # X axis limits
    axes.set_ylim([y_min, y_max])             # Y axis limits
    plt.bar(ticks_x, P_PCC_toDel, align='edge', width=1,color='b', edgecolor='black')
    plt.ylabel('Commited power (MW)')
    plt.grid()

    # Saving the plot
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_folder + '/' + figurename + '.png')
    # Closing plot
    plt.close()

#%% Wind and power forecasts
def windspe_pred_plot(day, Global_dataset, output_folder):
    Predictions = Global_dataset[day]
    fig = plt.figure('Wind & power for {}'.format(day))
    plt.suptitle('Wind & power for {}'.format(day))
    hour_ticks = hourly_xticks(Predictions['hours'][0])
    hours = [-12, -4, 1, 4, 8, 12]
    labels = ['DM', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6']
    # Wind speed forecasting subplot
    windspe_plot= fig.add_subplot(2, 1, 1)
    ticks_x = np.arange(0, len(hour_ticks), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(hour_ticks), 1), '', rotation=45)
    for i,hour in enumerate(Predictions['hours']):
        windspe_pred_list = Predictions['windspe_pred_{}'.format(labels[i])].tolist()
        while len(windspe_pred_list) != len(hour_ticks):
            windspe_pred_list.insert(0,None)            # Filling with nones to the left to adjust plot length
        plt.plot(windspe_pred_list, label=f'{labels[i]}')
    plt.plot(Predictions['windspe_real'], '--', label='Observation')
    plt.ylabel('Wind speed (m/s)')
    plt.legend()
    plt.grid()
    # Generated power subplot
    pgen_plot= fig.add_subplot(2, 1, 2)
    ticks_x = np.arange(0, len(hour_ticks), 1)
    plt.xticks(np.arange(0, len(hour_ticks), 1), hour_ticks, rotation=45)
    for i, hour in enumerate(Predictions['hours']):
        Pgen_pred_list = Predictions['Pgen_pred_{}'.format(labels[i])]
        while len(Pgen_pred_list) != len(hour_ticks):
            Pgen_pred_list.insert(0,None)               # Filling with nones to the left to adjust plot length
        plt.plot(Pgen_pred_list, label=f'{labels[i]}')
    plt.plot(Predictions['Pgen_real'], '--', label='Observation')
    plt.ylabel('Generated power (MW)')
    plt.legend()
    plt.grid()
    # Saving the plot
    plt.savefig(output_folder + '/Wind & Power.png')
    # Closing plot
    plt.close()
#%% Price plotting
def price_pred_plot(day, Global_dataset, output_folder):
    Predictions = Global_dataset[day]
    fig = plt.figure('Prices for {}'.format(day))
    plt.suptitle('Prices for {}'.format(day))
    hour_ticks = hourly_xticks(Predictions['hours'][0])
    hours = [-12, -4, 1, 4, 8, 12]
    labels = ['pred_DM', 'real_ID2', 'real_ID3', 'real_ID4', 'real_ID5', 'real_ID6']
    for i, hour in enumerate(Predictions['hours']):
        price_pred_list = Predictions[f'Price_{labels[i]}']
        if type(price_pred_list) != list:
            price_pred_list = price_pred_list.tolist()
        while len(price_pred_list) != len(hour_ticks):
            price_pred_list.insert(0, None)  # Filling with nones to the left to adjust plot length
        plt.plot(price_pred_list, label=f'{labels[i]}')
    plt.ylabel('Price (€/MWh)')
    ticks_x = np.arange(0, len(hour_ticks), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(hour_ticks), 1), hour_ticks, rotation=45)
    plt.legend()
    plt.grid()
    # Saving the plot
    plt.savefig(output_folder + '/Prices.png')
    # Closing plot
    plt.close()

#%% Real-time operation
def rt_plot(P_PCC_toDel, P_PCC_Real, P_gen, P_ESS_Sch, P_ESS_Real, Purch_IDs, SOC, figurename, output_folder):
    dates_label = []
    x = np.arange(len(P_PCC_toDel))
    PPC_Del = [a + b for a,b in zip(P_PCC_Real, Purch_IDs)]
    for i in range(len(P_PCC_toDel)):
        dates_label.append('{}:00'.format(i))
    max_y = max(max(P_PCC_toDel), max(P_PCC_Real), max(P_gen), max(P_ESS_Sch), max(PPC_Del)) + 1
    min_y = min(min(P_PCC_toDel), min(P_PCC_Real), min(P_gen), min(P_ESS_Sch), min(PPC_Del)) - 1

    # Initializing figure
    fig = plt.figure('Real time operation')
    fig.suptitle(output_folder)

    # PPC Schedule vs Delivery
    fig.add_subplot(3, 1, 1)
    # prin(len(P_PCC_toDel))
    plt.bar(x - 0.20*1.5, P_PCC_toDel, width=0.20, label='Scheduled', edgecolor='black')
    plt.bar(x - 0.20*0.5, P_PCC_Real, width=0.20, label='Delivered by plant', edgecolor='black')
    plt.bar(x + 0.20*0.5, Purch_IDs, width=0.20, label='Covered with IDs', edgecolor='black')
    plt.bar(x + 0.20*1.5, PPC_Del, width=0.20, label='Delivered + IDs', edgecolor='black')
    plt.ylim([min_y, max_y])
    plt.xticks(x, dates_label, rotation=45)
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid()
    # ESS & WTG Delivery
    fig.add_subplot(3, 1, 2)
    plt.bar(x - 0.4/2, P_gen, width=0.4, label='Turbine', edgecolor='black')
    plt.bar(x + 0.4/2, P_ESS_Sch, width=0.4, label='ESS (Sch)', edgecolor='black')
    plt.bar(x + 0.4/2, P_ESS_Real, width=0.4, label='ESS (Real)', edgecolor='black')
    plt.ylim([min_y, max_y])
    plt.xticks(x, dates_label, rotation=45)
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid()
    # SOC
    fig.add_subplot(3, 1, 3)
    dates_label = ['0:00']
    x = np.arange(24)
    for i in range(1,24):
        dates_label.append('{}:00'.format(i))
    plt.plot(SOC)
    plt.xticks(x, dates_label, rotation=45)
    plt.ylabel('SOC (%)')
    plt.ylim([-1, 110])
    plt.grid()
    # Launching the plot
    plt.tight_layout()
    # Saving the plot
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_folder + '/Real-time operation.png')
    # Closing plot
    plt.close()

#%% Plotting case comparison
def plot_case_comparison(Cases_array, Cases_Results, measure, plot_ylabel, sim_folder):
    measures_values = []
    for case_name in Cases_Results.keys():
        measures_values.append(measure_accumulator(Cases_Results, case_name, measure))

    fig = plt.figure(f'{measure}')
    x = np.arange(len(Cases_array))
    plt.xticks(x, Cases_array)
    plt.bar(x, measures_values,  label='Ideal', edgecolor='black', zorder=3)
    plt.grid(zorder=0)
    plt.ylabel(f'{plot_ylabel}')
    plt.savefig(sim_folder + f'/{measure}')
    # Closing plot
    plt.close()


