import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timezone
#%% Load datasets
prices_df = pd.read_csv('Data/Prices_DK1.csv', sep=';', usecols=["Price","datetime"], parse_dates=['datetime'],
                        index_col="datetime")
winds_df = pd.read_csv('Data/Brande_Dataset.csv', sep=';',parse_dates=['Date'], index_col="Date")
# Removing nans
winds_df = winds_df.fillna(method='ffill')
#%% Slicing data
df_day_start = '2020-01-01'
df_day_end = '2020-12-31'
df_day_start_ts = pd.Timestamp(df_day_start)
df_day_end_ts = pd.Timestamp(df_day_end) + pd.Timedelta('23h')
Prices_list = prices_df.loc[df_day_start_ts.replace(tzinfo=timezone.utc):df_day_end_ts.replace(tzinfo=timezone.utc)]['Price']
Winds_list = winds_df.loc[df_day_start_ts:df_day_end_ts]['Speed'].values

prices = Prices_list.values
winds = Winds_list[:len(prices)]
winds = [float(a) for a in winds]
#%% Calculate correlation coefficient
# from scipy.stats import pearsonr
# corr, _ = pearsonr(x, y)
# print('Pearsons correlation: %.3f' % corr)
# #% Plot scatter & linear correlation
# plt.scatter(x, y, s=0.5)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
# plt.xlabel('Electricity price')
# plt.ylabel('Wind speed')
# plt.title(f'Correlation between {df_day_start} and {df_day_end}; r={np.round(corr,3)}')
# plt.show()
#%% Study hydrogen generation profitability
price_H2 = 2
def evaluate_h2_price(prices, price_H2):
    Profitability = []  # List of hours when it is profitable to generate H2 (0:No/1:Yes)
    Power_to_H2 = 0.052  # [MWh] Energy required to generate 1kg of H2
    for price in prices:
        if price * Power_to_H2 < price_H2:
            Profitability.append(1)
        else:
            Profitability.append(0)
    profitability_perc = sum(Profitability) * 100 / len(Profitability)
    print(f'At {price_H2} €/kg it is profitable {np.round(profitability_perc,2)} % of the time')

    return profitability_perc

profitability_percs = []
prices_H2 = np.arange(0,5.25,0.25)
for price_H2 in prices_H2:
    profitability_perc = evaluate_h2_price(prices, price_H2)
    profitability_percs.append(profitability_perc)

figure = plt.figure('H2 prices profitability')
plt.plot(prices_H2, profitability_percs)
plt.xlabel('H2 price (€/kg)')
plt.ylabel('% of hours it is profitable')
# plt.xticks(prices_H2)
plt.grid()
plt.show()


