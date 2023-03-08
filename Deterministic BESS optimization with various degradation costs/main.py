#%% Importing libraries
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import math
#%% Manipulating pyplot default parameters
# Forcing tight layout
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Disabling showing figures
plt.ioff()
#%% Importing external files
from arb_fcns import *
from aux_fcns import *
#%% Importing dataset
Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()
#%% Simulation parameters
day = '2018-04-08'
initial_SOC = 0
batt_capacity = 10
batt_maxpower = 10/2
batt_efficiency = 0.9

#%% Importing energy prices
Global_dataset = np.load('Dataset.npy', allow_pickle=True).item()
Dataset = Global_dataset[day]
energy_price = Dataset['Price_pred_DM']

#%% Launching simulations
P_0, SOC_0 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 0*1000*batt_capacity)
P_10, SOC_10 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 10*1000*batt_capacity)
P_20, SOC_20 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 20*1000*batt_capacity)
P_50, SOC_50 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 50*1000*batt_capacity)
P_200, SOC_200 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 200*1000*batt_capacity)
P_300, SOC_300 = arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, 300*1000*batt_capacity)

#%% Plotting operation
dates_label = []
for i in range(24):                     # Filling X axis dates label
    dates_label.append('{}:00'.format(i))

fig = plt.figure("Program results")  # Creating the figure
# Energy price
price_plot = fig.add_subplot(2, 1, 1)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([min(energy_price) * 0.9, max(energy_price) * 1.1])  # X axis limits
# Inyecting the data
plt.bar(ticks_x, energy_price, align='edge', width=1, edgecolor='black', color='r')
# Adding labels
plt.ylabel('Price (€/MWh)')
plt.grid()
# SOC
SOC_plot = fig.add_subplot(2, 1, 2)  # Creating subplot
# Setting the grid
ticks_x = np.arange(0, 50, 1)  # Vertical grid spacing
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
# Setting the axes
axes = plt.gca()
axes.set_xlim([0, 24])  # X axis limits
axes.set_ylim([0, 110])  # X axis limits
# Inyecting the data
plt.plot(SOC_0, label='0 €/kWh')
plt.plot(SOC_10, label='10 €/kWh')
plt.plot(SOC_20, label='20 €/kWh')
plt.plot(SOC_50, label='50 €/kWh')
plt.plot(SOC_200, label='200 €/kWh')
plt.plot(SOC_300, label='300 €/kWh')
# Adding labels
plt.ylabel('SOC (%)')
plt.xlabel('Time (Hours)')
plt.legend()
plt.grid()

# Launching & saving the plot
plt.show()

#%% Plotting benefits and degradation
Brute_0, Net_0, Deg_0 = scen_eval(P_0, energy_price, SOC_0, 0, batt_capacity)
Brute_10, Net_10, Deg_10 = scen_eval(P_10, energy_price, SOC_10, 10*1000*batt_capacity, batt_capacity)
Brute_20, Net_20, Deg_20 = scen_eval(P_20, energy_price, SOC_20, 20*1000*batt_capacity, batt_capacity)
Brute_50, Net_50, Deg_50 = scen_eval(P_50, energy_price, SOC_50, 50*1000*batt_capacity, batt_capacity)
Brute_200, Net_200, Deg_200 = scen_eval(P_200, energy_price, SOC_200, 200*1000*batt_capacity, batt_capacity)
Brute_300, Net_300, Deg_300 = scen_eval(P_300, energy_price, SOC_300, 300*1000*batt_capacity, batt_capacity)

cases_list = ['0 €/kWh', '10 €/kWh', '20 €/kWh', '50 €/kWh', '200 €/kWh','300 €/kWh']
brutes = [Brute_0, Brute_10, Brute_20, Brute_50, Brute_200, Brute_300]
nets = [Net_0, Net_10, Net_20, Net_50, Net_200, Net_300]
degs = [Deg_0, Deg_10, Deg_20, Deg_50, Deg_200, Deg_300]

# Gross benefits vs net benefits
fig = plt.figure("gross_vs_net_results")  # Creating the figure
# plt.xticks(cases_list)
plt.scatter(Brute_0, Net_0, color='r')
plt.scatter(Brute_10, Net_10, color='r')
plt.scatter(Brute_20, Net_20, color='r')
plt.scatter(Brute_50, Net_50, color='r')
plt.scatter(Brute_200, Net_200, color='r')
plt.scatter(Brute_300, Net_300, color='r')
for i, txt in enumerate(cases_list[:5]):
    plt.annotate(txt, (brutes[i]+1, nets[i]+1))
plt.annotate('300 €/kWh', (Brute_300-10, Net_300+2))
plt.xlim([-10, 80])
plt.grid()
plt.xlabel('Gross benefits (€)')
plt.ylabel('Net benefits (€)')
plt.show()

# Gross benefits vs degradation
fig = plt.figure("gross_vs_deg_results")  # Creating the figure
# plt.xticks(cases_list)
plt.scatter(Brute_0, Deg_0, color='r')
plt.scatter(Brute_10, Deg_10, color='r')
plt.scatter(Brute_20, Deg_20, color='r')
plt.scatter(Brute_50, Deg_50, color='r')
plt.scatter(Brute_200, Deg_200, color='r')
plt.scatter(Brute_300, Deg_300, color='r')
for i, txt in enumerate(cases_list[:5]):
    plt.annotate(txt, (brutes[i]+1, degs[i]))
plt.annotate('300 €/kWh', (Brute_300-10, Deg_300+0.00001))
plt.xlim([-10, 80])
plt.grid()
plt.xlabel('Gross benefits (€)')
plt.ylabel('Short-time degradation (%)')
plt.show()