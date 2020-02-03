import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import tqdm
from os import listdir
from os.path import isfile, join
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller, kpss
from itertools import product
from itertools import repeat
from tqdm import notebook
from pandas.plotting import autocorrelation_plot
from ipywidgets import IntProgress
import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host=host_id,
                             user='rcm_aero',
                             password='2nR>q4.5_a6ApfwV',
                             db='ao_bombar_show')
sql = 'SELECT * FROM ao_bombar_show.agg_loc_set'
p = pd.read_sql_query(sql, connection)
p.head()
p.Date = pd.to_datetime(p.Date)
# create day ,year ,month column
p['Year'] = p['Date'].apply(lambda x: x.year)
p['Month'] = p['Date'].apply(lambda x: x.month)
p['Day'] = p['Date'].apply(lambda x: x.day)

# grouping by year, month
s = p.groupby(['Year', 'Month'])['replace'].sum()
s1 = pd.DataFrame(s)
# to check the stationarity using visulization  of series.
s2 = s1.unstack(level=0)
s2['replace'].plot(figsize=(20, 8))

s1.describe()
# to check the summarization
s1.boxplot()

# decomposition of time Series
decomposition = sm.tsa.seasonal_decompose(s1, model='additive', freq=7)
fig = decomposition.plot()
plt.show()

# rolling method to find stationarity , it is show series is not stationary

s1.rolling(12).mean().plot()
plt.xlabel('Year')

split = round(s1.shape[0] / 2)

X1 = s1[0:int(split)]
X2 = s1[int(split):]

mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# to check the stationarity
result = adfuller(s2['replace'], autolag="AIC")
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

kpsstest = kpss(p1.Sales, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
kpss_output

# This is used to find the value of d
s1 = s1 - s1.shift(1)

# To check the stationarity
result = adfuller(s1.dropna(), autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

kpsstest = kpss(s1.dropna(), regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
kpss_output


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


tsplot(s1['replace'], lags=36)

# SARIMA Modelling

train = s1.iloc[0:108, :]

test = s1.iloc[108:, :]

p = range(0, 3)
d = range(0, 2)
q = range(0, 8)
Ps = range(0, 4)
Qs = range(0, 2)
D = range(0, 1)
s = 12 # season length is still 52

# creating list with all the possible combinations of parameters
parameters = product(p, d, q, Ps, D, Qs)
parameters_list = list(parameters)
len(parameters_list)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    rms = []
    results = []
    best_rm = float("inf")
    rt = pd.Series([float(train.replace.values[i]) for i in range(len(train.replace.values))])
    for param in tqdm.notebook.tqdm(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(rt, order=(param[0], param[1], param[2]),
                                              seasonal_order=(param[3], param[4], param[5], 12)).fit()
            rm = np.sqrt(sum(model.resid ** 2) / len(model.resid))
        except:
            continue
        # aic = model.aic

        # rms.append(rm)

        # saving best model, AIC and parameters
        if rm < best_rm:
            best_model = model
            best_rm = rm
            best_param = param
        results.append([param, rm])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'rmse']

    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='rmse', ascending=True).reset_index(drop=True)
    # result_table['rmse']=rms
    return result_table


result_table = optimizeSARIMA(parameters_list, d, D, s)





tr = pd.Series([float(train.replace.values[i]) for i in range(len(train.replace.values))])

rm = []
for i in range(result_table.shape[0]):
    arim = sm.tsa.statespace.SARIMAX(s,  order=(result_table['parameters'][i][0], result_table['parameters'][i][1], result_table['parameters'][i][2]), seasonal_order=(result_table['parameters'][i][3], result_table['parameters'][i][4], result_table['parameters'][i][5], 12)).fit()
    p = arim.forecast(12)
    rm.append(rmse(test.Sales.values, p))
result_table['test_rmse'] = rm

dfe = result_table['rmse']-result_table['test_rmse']
result_table['dfe'] = dfe
result_table[['rmse', 'test_rmse']].plot()

arim = sm.tsa.statespace.SARIMAX(s, order=(2, 0, 7), seasonal_order=(3, 0, 0, 12))
a = arim.fit()
a.summary()

arima_output = pd.DataFrame(a.forecast(12))
arim_out = pd.DataFrame(arima_output[0])
#arima_output.to_csv('C:/Users/amans.AMAN/Downloads/arima_output.csv')

arim_out.columns = ['forecast']
arim_out.to_csv('C:/Users/amans.AMAN/OneDrive/Desktop/New folder (2)/result.csv')

# To Check the quality of model i.e. anything remain or not in residual
pd.Series(a.resid).describe()
statsmodels.stats.diagnostic.acorr_ljungbox(a.resid,24)

# some of the p value is less then 0.05 that is autocorrelation is present. we can modify the sarima more.

autocorrelation_plot(a.resid)
sns.distplot(a.resid)
