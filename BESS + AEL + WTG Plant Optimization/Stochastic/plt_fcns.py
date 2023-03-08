#%% Load python libraries
from matplotlib import pyplot as plt
import numpy as np

#%% Force tight layout
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True

# Disabling showing figures
plt.ioff()
#%% Plot daily programs
def daily_powers_plot(day, Results):
    # Generate dates label
    dates_label = []  # X axis dates label
    for i in range(len(Results['P_AEL'])):  # Filling X axis dates label
        dates_label.append('{}:00'.format(i))
    x = np.arange(len(Results['P_AEL']))

    # Initialize RHU powers plot
    fig = plt.figure(f'RHU powers for {day}')
    plt.suptitle(f'RHU powers for {day}')

    # WTG Powers
    WTG_plot = fig.add_subplot(4, 1, 1)  # Creating subplot
    ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
    axes.set_axisbelow(True)
    plt.bar(x + 0.25, Results['P_WTG_Grid'], color='orange',  width=0.25, label='To grid', edgecolor='black')
    plt.bar(x + 0.50, Results['P_WTG_BESS'], color='purple',  width=0.25, label='To BESS', edgecolor='black')
    plt.bar(x + 0.75, Results['P_WTG_AEL'], color='cyan', width=0.25, label='To AEL', edgecolor='black')
    plt.legend()
    plt.ylabel('WTG (MW)')
    plt.grid()

    # AEL Powers
    AEL_plot = fig.add_subplot(4, 1, 2)  # Creating subplot
    ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
    axes.set_axisbelow(True)
    plt.bar(x + 0.25, Results['P_on_AEL'], color= 'blue' ,width=0.25, label='On', edgecolor='black')
    plt.bar(x + 0.50, Results['P_idle_AEL'], color = 'purple', width=0.25, label='Idle', edgecolor='black')
    plt.bar(x + 0.75, Results['P_off_AEL'], color='red', width=0.25, label='Off', edgecolor='black')
    plt.legend()
    plt.ylabel('AEL (MW)')
    plt.grid()

    # BESS Powers
    BESS_plot = fig.add_subplot(4, 1, 3)  # Creating subplot
    ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(Results['P_AEL']), 1), [], rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
    axes.set_axisbelow(True)
    plt.bar(ticks_x + 0.19, [-a for a in Results['P_Grid_BESS']], color='orange',  width=0.2, edgecolor='black')
    plt.bar(ticks_x + 0.39, [-a for a in Results['P_WTG_BESS']], color='green',  width=0.2, label='From WTG', edgecolor='black')
    plt.bar(ticks_x + 0.59, Results['P_BESS_Grid'], color='orange',  width=0.2, label='To/from grid', edgecolor='black')
    plt.bar(ticks_x + 0.79, Results['P_BESS_AEL'], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
    plt.legend()
    plt.ylabel('BESS (MW)')
    plt.grid()

    # Grid Powers
    Grid_plot = fig.add_subplot(4, 1, 4)  # Creating subplot
    ticks_x = np.arange(0, len(Results['P_AEL']), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(Results['P_AEL']), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(Results['P_AEL'])])  # X axis limits
    axes.set_axisbelow(True)
    plt.bar(ticks_x + 0.19, Results['P_WTG_Grid'], color='green', width=0.2, label='From WTG', edgecolor='black')
    plt.bar(ticks_x + 0.39, Results['P_BESS_Grid'], color='purple', width=0.2, label='To/from BESS', edgecolor='black')
    plt.bar(ticks_x + 0.59, [-a for a in Results['P_Grid_BESS']], color='purple',  width=0.2, edgecolor='black')
    plt.bar(ticks_x + 0.79, [-a for a in Results['P_Grid_AEL']], color='cyan',  width=0.2, label='To AEL', edgecolor='black')
    plt.legend()
    plt.ylabel('Grid (MW)')
    plt.grid()

    # Launching the plot
    plt.show()
    plt.ioff()

#%% Plot daily states
def daily_states(day, P_Gen, Price_El, RHU_Parameters, Results, cases, mode):
    # Generate dates label
    dates_label = []  # X axis dates label
    if mode == 'single':
        for i in range(len(Results['P_AEL'])):  # Filling X axis dates label
            dates_label.append('{}:00'.format(i))
        x = np.arange(len(Results['P_AEL']))
        dates_label_plus1 = [] # Dates label for states plot which has 25 hours length
        for i in range(len(Results['state_AEL'])+1):  # Filling X axis dates label
            dates_label_plus1.append('{}:00'.format(i))
        x_plus1 = np.arange(len(Results['state_AEL'])+1)
    if mode == 'multiple':
        for i in range(len(Results[0]['P_AEL'])):  # Filling X axis dates label
            dates_label.append('{}:00'.format(i))
        x = np.arange(len(Results[0]['P_AEL']))
        dates_label_plus1 = [] # Dates label for states plot which has 25 hours length
        for i in range(len(Results[0]['state_AEL'])+1):  # Filling X axis dates label
            dates_label_plus1.append('{}:00'.format(i))
        x_plus1 = np.arange(len(Results[0]['state_AEL'])+1)
    # Initialize RHU powers plot
    fig = plt.figure(f'General states for {day}')

    # Generated power
    Pgen_plot = fig.add_subplot(3, 2, 1)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(P_Gen)
    plt.ylabel('Generation (MW)')
    plt.grid()

    # Electricity price
    E_price_plot = fig.add_subplot(3, 2, 2)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(Price_El)
    plt.ylabel('Energy price (€/MWh)')
    plt.grid()

    # AEL H2 in O2
    AEL_plot = fig.add_subplot(3, 2, 3)  # Creating subplot
    ticks_x = np.arange(0, len(x_plus1), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x_plus1), 1), dates_label_plus1, rotation=45)
    axes = plt.gca()
    axes.set_axisbelow(True)
    if mode == 'single':
        impurity = Results['imp_AEL']
        impurity.insert(0, impurity[0])
        plt.plot(Results['imp_AEL'])
    if mode == 'multiple':
        for i in range(len(cases)):
            impurity = Results[i]['imp_AEL']
            impurity.insert(0, impurity[0])
            plt.plot(impurity, label=cases[i])
        plt.legend()
    plt.ylabel('H2 in O2 (%)')
    plt.grid()

    # BESS SOC
    SOC_plot = fig.add_subplot(3, 2, 4)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    if mode == 'single':
        plt.plot(Results['SOC'])
    if mode == 'multiple':
        for i in range(len(cases)):
            plt.plot(Results[i]['SOC'], label=cases[i])
        plt.legend()
    plt.ylabel('SOC(%)')
    plt.grid()

    # H
    H_acc = fig.add_subplot(3, 2, 5)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    if mode == 'single':
        plt.plot(Results['H_AEL'])
    if mode == 'multiple':
        for i in range(len(cases)):
            plt.plot(Results[i]['H_AEL'], label=cases[i])
        plt.legend()
    plt.ylabel('Generated H2 (kg)')
    plt.grid()

    # Exchanged energy
    WTG_plot = fig.add_subplot(3, 2, 6)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    if mode == 'single':
        plt.plot(Results['E_acc'])
    if mode == 'multiple':
        for i in range(len(cases)):
            plt.plot(Results[i]['E_acc'], label=cases[i])
        plt.legend()
    plt.ylabel('Net energy exchange (MWh)')
    plt.grid()

    plt.show()

#%% Compare selected DM schedule vs real operation
def daily_dm_vs_real(day, P_Gen_Pred, Price_El_Pred, P_Gen_real, Price_El_real,
                     RHU_Parameters, Results_DM, Results_RT):
    # Generate dates label
    dates_label = []  # X axis dates label
    for i in range(len(Results_DM['P_AEL'])):  # Filling X axis dates label
        dates_label.append('{}:00'.format(i))
    x = np.arange(len(Results_DM['P_AEL']))
    # Initialize RHU powers plot
    fig = plt.figure(f'General states for {day}')

    # Generated power comparison
    Pgen_plot = fig.add_subplot(3, 2, 1)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(P_Gen_Pred, label='Predicted')
    plt.plot(P_Gen_real, label='Real')
    plt.ylabel('Generation (MW)')
    plt.legend()
    plt.grid()

    # Electricity prices comparison
    E_price_plot = fig.add_subplot(3, 2, 2)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(Price_El_Pred, label='Predicted')
    plt.plot(Price_El_real, label='Real')
    plt.ylabel('Energy price (€/MWh)')
    plt.legend()
    plt.grid()

    # Generated power sent to grid comparison
    WTG_Grid_plot = fig.add_subplot(3, 2, 3)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_axisbelow(True)
    plt.plot(Results_DM['P_WTG_Grid'], label='Expected')
    plt.plot(Results_RT['P_WTG_Grid'], label='Real')
    plt.legend()
    plt.ylabel('Pgen to grid (MW)')
    plt.grid()

    # BESS power sent to grid comparison
    SOC_plot = fig.add_subplot(3, 2, 4)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(Results_DM['P_BESS_Grid'], label='Expected')
    plt.plot(Results_RT['P_BESS_Grid'], label='Real')
    plt.legend()
    plt.ylabel('BESS power to grid (MW)')
    plt.grid()

    # Generated hydrogen comparison
    H_acc = fig.add_subplot(3, 2, 5)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(Results_DM['H_AEL'], label='Expected')
    plt.plot(Results_RT['H_AEL'], label='Real')
    plt.legend()
    plt.ylabel('Generated H2 (kg)')
    plt.grid()

    # SOC comparison
    WTG_plot = fig.add_subplot(3, 2, 6)  # Creating subplot
    ticks_x = np.arange(0, len(x), 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, len(x), 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, len(x)])  # X axis limits
    axes.set_axisbelow(True)
    plt.plot(Results_DM['SOC'], label='Expected')
    plt.plot(Results_RT['SOC'], label='Real')
    plt.legend()
    plt.ylabel('SOC (%)')
    plt.grid()

    plt.show()



