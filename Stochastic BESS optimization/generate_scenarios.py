# Importing libraries
import pandas as pd
import statsmodels.api as sm
import os, datetime
from datetime import timezone
import warnings
import concurrent.futures
from matplotlib import pyplot as plt
def scenario_price_filename(day):
    return day.strftime("%Y_%m_%d")+'_price_scenarios.csv'

def save_scenarios(day, save_folder, overwrite, n_scenarios, prices_df, model_order,
                   model_seasonal_order, train_length):
    # Filename
    filename = os.path.join(save_folder, scenario_price_filename(day))
    # Proceed only if file does not exist or if it should be overwritten
    if not os.path.isfile(filename) or overwrite:
        # Generating training set
        day = pd.Timestamp(day).replace(tzinfo=timezone.utc)
        train_end = day - pd.Timedelta('1h')
        train_start = train_end - pd.Timedelta('{}d 24h'.format(train_length))
        train_set = prices_df[train_start:train_end-pd.Timedelta('1h')]
        # Defining test set
        test_start = day - pd.Timedelta('1h')
        test_end = test_start + pd.Timedelta('23h')
        test_set = prices_df[test_start:test_end]
        # Generating SARIMA model from doi 10.1109/SSCI44817.2019.9002930
        model = sm.tsa.SARIMAX(train_set, order=model_order, seasonal_order=model_seasonal_order,
                               initialization='approximate_diffuse')
        model_fit = model.fit(disp=False)
        # Generating scenarios
        new_scenarios = model_fit.simulate(nsimulations = len(test_set), repetitions = n_scenarios, anchor = 'end')
        new_scenarios['Price'].to_csv(filename)
        print('written:',filename)
    else:
        print('skipped:',filename)

if __name__ == '__main__':
    
    # Parameters
    firstday = '2020-01-01 00:00:00'      # First day to process
    lastday = '2020-12-31 00:00:00'       # Last day to process
    train_length = 100                    # Training set length (days)
    n_scenarios = 1000                    # Number of price scenarios created per day
    model_order = (2, 1, 3)               # SARIMA order
    model_seasonal_order = (1, 0, 1, 24)  # SARIMA seasonal order
    save_folder = 'scenarios'             # Save folder
    overwrite = False                     # Wether existing files are overwritten

    # Disabling Statsmodels warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', UserWarning)

    timing = datetime.datetime.now()

    # Importing price data from csv
    prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","datetime"], parse_dates=['datetime'],
                            index_col="datetime")
    prices_df = prices_df.asfreq('h')

    # Generate scenarios
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for day in pd.date_range(firstday, lastday, freq='D'):
            executor.submit(save_scenarios, day, save_folder, overwrite, n_scenarios, prices_df, model_order,
                            model_seasonal_order, train_length)

    print('Total time:', datetime.datetime.now() - timing)



