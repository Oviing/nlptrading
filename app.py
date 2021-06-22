# !pip install streamlit
# !pip install ta
# !pip install yfinance
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # https://pypi.org/project/yfinance/
from fbprophet import Prophet
import plotly.graph_objects as go




import datetime

start = pd.to_datetime('2010-01-01')
#end = date.today()
end = pd.to_datetime('2021-06-22')

start_pred = pd.to_datetime('2021-06-14')
end_pred = pd.to_datetime('2021-06-24')



start_date = st.sidebar.date_input('Start date', start)
end_date = st.sidebar.date_input('End date', end)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')


start_date2 = st.sidebar.date_input('Prediction period start date', start_pred)
end_date2 = st.sidebar.date_input('Prediction period end date', end_pred)
if start_date2 < end_date2:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date2, end_date2))
else:
    st.sidebar.error('Error: End date must fall after start date.')

day_d = start_date2 - end_date2
dd = str(day_d)

day_diff = day_d

st.write(dd)


st.sidebar.write('Please insert the start and ending date. The model then takes n time steps to forecast n days')
time_span = st.sidebar.number_input('Insert a Number')




##############
# Stock data #
##############


tickers_to_download = ['EURUSD=X', 'GBPUSD=X', 'USDCHF=X', 'USDJPY=X', 'EURJPY=X', 'USDCAD=X', 'AUDUSD=X', 'EURCHF=X', 
                       'YM=F', 'NQ=F', '^STOXX50E', '^FTSE', '^N225', '^DJI', '^VIX', 'CL=F', 'SI=F', 'GC=F', '^TNX',
                       '^HSI', 'ZF=F', 'HG=F', '^IXIC', '^GDAXI', '^FCHI']


forex_df_list = []

for ticker in tickers_to_download:
  ts = yf.download(ticker, start_date, end_date, progress=False)['Close']
  ts = ts.rename(ticker.lower().split('=', 1)[0] + '_close')
  forex_df_list.append(ts)

forex_df = pd.concat(forex_df_list, axis=1)

####################
#

###################
# Set up main app #
###################

st.title('Forex EUR/USD prediction')



############################# NEW STUFF ########


st.dataframe(forex_df.tail(10))
st.write(' ')
st.line_chart(forex_df['eurusd_close'].tail(730))


##################################################### DATA PREP.

forex_df['ds'] = forex_df.index
forex_df['y'] = forex_df['eurusd_close'].values

# calc test window size

#end_string = str(end[0:9])
#new_window = int(end_string[8:-10]) + int(time_span)

#st.write(time_span)
#st.write(end_string)
#st.write(new_window)

train_window =  ['2010-01-03','2021-06-13']
test_window =  ['2021-06-14','2021-06-27']
#train_window =  ['2010-01-03', str(new_window)]
#train_window =  ['2010-01-03',start_pred]
#test_window =  [start_pred, end_pred]

forex_df = forex_df.dropna()

train = forex_df[train_window[0]:train_window[1]]
test = forex_df[test_window[0]:test_window[1]]


##### PROPHET ################

st.header("Prediction")

# define Prophet model with hyperparameters
m_2 = Prophet(growth='linear',n_changepoints = 9,changepoint_prior_scale=0.05)
# add custom seasonality
m_2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m_2.add_seasonality(name='weekly', period=7, fourier_order=15)
# add extra regressor
extra_regressor_2 = ['cl_close', '^dji_close', '^n225_close', '^ftse_close', '^stoxx50e_close',
                    'nq_close', 'ym_close', 'si_close', 'gc_close', '^tnx_close', '^hsi_close', 'zf_close', 'hg_close', '^ixic_close',
                   '^gdaxi_close', '^fchi_close', 'gbpusd_close', 'usdchf_close', 'usdjpy_close', 'eurjpy_close', 'usdcad_close',
                   'audusd_close', 'eurchf_close']
for reg in extra_regressor_2:
  m_2.add_regressor(name=reg, mode='additive')


m_2.fit(train)

future_2 = m_2.make_future_dataframe(10, freq='B')
future_2 = future_2.merge(forex_df[extra_regressor_2], left_on='ds', right_index=True, how='left').fillna(method='pad')
forecast_2 = m_2.predict(future_2)

st.dataframe(forecast_2.tail(10))
st.line_chart(forecast_2['yhat'].tail(10))

forecast_2.set_index('ds', inplace=True)
final_df_results_all = pd.merge(test['eurusd_close'], forecast_2[['yhat']], left_index=True, right_index=True)

st.dataframe(final_df_results_all.tail(10))

def calc_errors(test, preds):
    percentiles = [5, 25, 50, 75, 95]
    elementwise_mae = np.absolute(np.subtract(preds, test))
    # Mean mae
    mean_mae = sum(elementwise_mae) / len(test)
    print(f"Mean MAE: {mean_mae:.2f}")
    percent_mae = sum(elementwise_mae) / sum(test)
    print(f"MAE%: {percent_mae * 100:.2f}%")
    # Rmse
    rmse = np.sqrt(np.mean(np.power(np.subtract(preds, test), 2)))
    print(f"RMSE: {rmse:.2f}")
    # Bias
    bias = np.mean(np.subtract(preds, test))
    print(f"Bias: {bias:.2f}\n")
    # Mae distrib
    distr_mae = []
    for perc in percentiles:
        temp_mae = np.percentile(elementwise_mae, perc)
        distr_mae.append(temp_mae)
        print(f"{perc}th percetile MAE: {temp_mae:.2f}")


    return mean_mae, percent_mae, rmse, bias, distr_mae, elementwise_mae


mean_mae, percent_mae, rmse, bias, distr_mae, elementwise_mae = calc_errors(final_df_results_all['eurusd_close'].values, final_df_results_all['yhat'].values)

st.header('Metrics')
st.write("Mean Absolute Error:", mean_mae)
st.write("Mean Absolute Error (%):", percent_mae)
st.write("Root Mean Squared Error:", rmse)
st.write("Bias:", bias)



percentiles = [5, 25, 50, 75, 95]
final_error_plot = plt.figure(figsize=(8, 8))
plt.title('MAE distribution')
plt.xlabel('MAE')
plt.ylabel('Count')
# Plot distr
plt.hist(elementwise_mae, bins=30)
    # plot mean MAE
plt.axvline(x=mean_mae, label='Mean MAE', c='r', linestyle='-')
# plot percentiles
line_types = [':', '-.', '--', '-.', ':']
for xc, lt, p in zip(distr_mae, line_types, percentiles):
    plt.axvline(x=xc, label='{}th percentile MAE'.format(p), c='r', linestyle=lt)
plt.legend()
#final_error_plot = plt.show()

st.pyplot(final_error_plot)





