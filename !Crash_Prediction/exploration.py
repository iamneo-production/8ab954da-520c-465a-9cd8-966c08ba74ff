# %% [markdown]
# ## Data Exploration
# Datasets: S&P500 (USA), Dow Jones (USA), Nasdaq (USA), Nikkei225 (Japan), SSE (Shanghai/China), HSI (Hong Kong), 
# BSESN (India), DAX (Europe), SMI (Switzerland), MXX (Mexico), BVSP (Brazil)
# 1. Find the least correlated datasets
# 2. Distribution of prices, daily returns and drawdowns
# 3. Identification and definition of crashes

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import seaborn as sns
from pylab import rcParams
from collections import defaultdict
from scipy.optimize import curve_fit
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# %% [markdown]
# #### 1. Find least correlated datasets
# In this section, I select the datasets that will be further explored and will later be used to develop an algorithm that predicts crashes. To avoid overfitting on certain patterns and biased test sets, the datasets used should not have a strong cross correlation. 

# %%
os.chdir('/home/roman/Documents/Projects/Bubbles/data')
datasets_original_test = ['^GSPC.csv', '^DJI.csv', '^NDX.csv', '^N225.csv', 'SSE.csv',\
'^HSI.csv', '^BSESN.csv', '^GDAXI.csv', '^SSMI.csv', '^MXX.csv', \
                     '^BVSP.csv']
dataset_names_test = ['S&P 500', 'DJ', 'NDX', 'N225', 'SSE', 'HSI', 'BSESN', 'DAX', \
                 'SMI', 'MXX', 'BVSP']
datasets_test = []
for d in datasets_original_test:
    data_original = pd.read_csv(d, index_col = 'Date')
    data_original.index = pd.to_datetime(data_original.index, format='%Y/%m/%d')
    data_ch = data_original['Close'].pct_change()
    datasets_test.append(data_ch)
df_returns = pd.concat(datasets_test, axis=1, join_axes=[datasets_test[0].index])
df_returns.columns = dataset_names_test
corr = df_returns.corr()
print('Correlations of daily returns between datasets:')
rcParams['figure.figsize'] = 10, 6
ax = sns.heatmap(corr, annot=True, cmap='rocket_r')

# %% [markdown]
# The Correlation matrix shows that the three US indices (S&P, DJ, NDX), the DAX and SMI and the MXX and BVSP are highly correlated. To avoid overfitting when training prediction models, a correlation of > 0.5 for any two datasets should be avoided. Therefore, the SJ, NDX, DAX and MXX will be excluded for further analysis

# %%
datasets_original = ['^GSPC.csv', '^N225.csv', 'SSE.csv','^HSI.csv', '^BSESN.csv', \
                     '^SSMI.csv', '^BVSP.csv']
dataset_names = ['S&P 500', 'N225', 'SSE', 'HSI', 'BSESN', 'SMI', 'BVSP']
datasets = []
for d in datasets_original:
    data_original = pd.read_csv(d, index_col = 'Date')
    data_original.index = pd.to_datetime(data_original.index, format='%Y/%m/%d')
    data_norm = data_original['Close'] / data_original['Close'][-1]
    data_ch = data_original['Close'].pct_change()
    window = 10
    data_vol = data_original['Close'].pct_change().rolling(window).std()
    data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol], axis=1).dropna()
    data.columns = ['price', 'norm', 'ch', 'vol']
    datasets.append(data)
datasets[5] = datasets[5].loc['1990-11-09':,:]  #<-- SMI has much missing data before 11/9/90

df_ch = [d['ch'] for d in datasets]
df_returns = pd.concat(df_ch, axis=1, join_axes=[datasets[0].index])
df_returns.columns = dataset_names
corr = df_returns.corr()
print('Correlations of daily returns between datasets (non-correlated datasets):')
ax = sns.heatmap(corr, annot=True, cmap='rocket_r')

# %% [markdown]
# The correlation matrix with the remaining datasets shows no correlations among any two datasets of > 0.5

# %% [markdown]
# #### 2. Distribution of prices, daily returns, drawdowns

# %%
##### Plot price over time
rcParams['figure.figsize'] = 10, 3
plt_titles = ['S&P since 1950', 'N225 since 1965', 'SSE since 1996', 'HSI since 1987', \
              'BSESN since 1997', 'SMI since 1990', 'BVSP since 2002']
for ds, t in zip(datasets, plt_titles):
    plt.plot(ds['price'], color='blue', linewidth=0.7)
    plt.grid()
    plt.legend(['Price'])
    plt.title(t + ' - Price')
    plt.show()

# %% [markdown]
# The time series plots give an impression of the performance of the different markets over the past 50-20 years.

# %%
##### Plot daily return over time
for ds, t in zip(datasets, plt_titles):
    plt.plot(ds['ch'], color='blue', linewidth=0.7)
    plt.ylim(-0.2, 0.2)
    plt.grid()
    plt.legend(['Return'])
    plt.title(t + ' - Daily returns')
    plt.show()

# %% [markdown]
# The amplitude of daily returns over time for all datasets give an impression of the volatility in the different markets with the Brazilian market showing the larges daily gains/losses.

# %%
##### Autocorrelation
corr_ds = []
rcParams['figure.figsize'] = 10, 5
for ds, t in zip(datasets, plt_titles):
    corr = [1]
    for i in range(1, 7):
        corr.append(np.corrcoef(ds['ch'][i:], ds['ch'][:-i])[0, 1])
    plt.plot(corr)
plt.title('All data sets - Correlation of daily returns')
plt.legend(dataset_names)
plt.xlabel('Lag (days)')
plt.ylabel('Correlation')
plt.grid()
plt.show()

# %% [markdown]
# The autocorrelation of daily returns is close to zero for a lag > 1 day, indicating that the daily return is not a strong predictor for the price change of the following day.

# %%
##### Plot distribution of daily returns
rcParams['figure.figsize'] = 10, 3
for ds, t in zip(datasets, plt_titles):
    plt.hist(ds['ch'], bins=200, rwidth=1, alpha=0.75)
    plt.xlim(-0.2, 0.2)
    plt.title(t + ' - all daily returns')
    plt.grid()
    plt.show()

# %% [markdown]
# The histograms for daily return distributions show that the vast majority of returns lies between -0.05 and 0.05 for all datasets. The absolute values of extreme daily gains or losses are larger than 0.1. A visual comparison between the datasets shows that the SSE and BVSP have "fat tails" indicating a realtively high volatility with a large amount of high one day gains/large one day losses.

# %%
##### Plot log-distribution of daily returns
for ds, t in zip(datasets, plt_titles):
    max_return = max(abs(ds['ch']))
    m = round(max_return+0.01,2)
    bins = np.linspace(-m, m, 2000)
    d = {}
    for i in range(1, len(bins)+1):
        d[i] = bins[i-1]
    disc = np.digitize(x=ds['ch'], bins=bins)
    d1 = defaultdict(int)
    for i in disc:
        d1[d[i]] += 1
    df = pd.DataFrame(list(d1.items()))
    df.columns = ['return', 'n']
    df_neg = df[df['return']<0]
    df_neg = df_neg.sort_values(by='return', ascending=True).reset_index(drop=True)
    plt.scatter(df_neg['return'], df_neg['n'], s=30, color='red')
    plt.yscale('log')
    df_neg_reg = df_neg[df_neg['return']>-0.05]
    m, c = np.polyfit(df_neg_reg['return'], np.log(df_neg_reg['n']), 1)
    y_fit = np.exp(m*df_neg['return'] + c)
    plt.ylim(bottom=10**0)
    df_pos = df[df['return']>0]
    df_pos = df_pos.sort_values(by='return', ascending=False).reset_index(drop=True)
    plt.scatter(df_pos['return'], df_pos['n'], s=20, color='green')
    plt.yscale('log')
    df_pos_reg = df_pos[df_pos['return']<0.05]
    m, c = np.polyfit(df_pos_reg['return'], np.log(df_pos_reg['n']), 1)
    y_fit = np.exp(m*df_pos['return'] + c)
    plt.ylim(bottom=10**-0.1)
    plt.xlim(-0.3, 0.3)
    plt.title(t + ' - distribution of daily returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency (log)')
    plt.grid()
    plt.show()
    plt.show()

# %% [markdown]
# The frequency distribution plots show that extreme positive (red) and negtive (green) returns occur on rare instances. Extreme negative daily returns of ~0.1 and more likely contribute to a crash.
# 
# ##### Drawdowns
# To detect crashes, the drawdowns are calculated. A drawdown is a total loss over consequtive days from the last maximum to the nex minimum of the price. A drawdown occuring over n days (the period from t_1 to t_n) is described as d = (p_max - p_min)/pmax, with p_max = p(t_1) > p(t_2) > ... > p(t_n) = p_min.

# %%
##### Drawdowns
dd_df = []
for ds in datasets:
    pmin_pmax = (ds['price'].diff(-1) > 0).astype(int).diff() #<- -1 indicates pmin, +1 indicates pmax
    pmax = pmin_pmax[pmin_pmax == 1]
    pmin = pmin_pmax[pmin_pmax == -1]
    if pmin.index[0] < pmax.index[0]:
        pmin = pmin.drop(pmin.index[0])
    if pmin.index[-1] < pmax.index[-1]:
        pmax = pmax.drop(pmax.index[-1])
    dd = (np.array(ds['price'][pmin.index]) - np.array(ds['price'][pmax.index])) \
        / np.array(ds['price'][pmax.index])
    dur = [np.busday_count(p1.date(), p2.date()) for p1, p2 in zip(pmax.index, pmin.index)]
    d = {'Date':pmax.index, 'drawdown':dd, 'd_start': pmax.index, 'd_end': pmin.index, \
         'duration': dur}    
    df_d = pd.DataFrame(d).set_index('Date')
    df_d.index = pd.to_datetime(df_d.index, format='%Y/%m/%d')
    df_d = df_d.sort_values(by='drawdown')
    df_d['rank'] = list(range(1,df_d.shape[0]+1))
    dd_df.append(df_d)

# Plot duration of drawdowns
l_dict_dd = []
for dd, t in zip(dd_df, plt_titles):
    max_dd = max(abs(dd['drawdown']))
    m = round(max_dd+0.01,2)
    bins = np.linspace(-m, m, 800)
    d = {}
    for i in range(1, len(bins)+1):
        d[i] = bins[i-1]
    disc = np.digitize(x=dd['drawdown'], bins=bins)
    d1 = defaultdict(int)
    for i in disc:
        d1[d[i]] += 1
    l_dict_dd.append(d1)
    plt.bar(x=dd['duration'].value_counts().index, height=dd['duration'].\
        value_counts()/dd['duration'].shape[0], color='red', alpha=0.6)
    plt.xticks(dd['duration'].value_counts().index)
    plt.title(t + ' - Duration of drawdowns')
    plt.xlabel('Duration (number of days)')
    plt.grid()
    plt.show()

# %% [markdown]
# The duration of drawdown histograms show how long drawdowns typically last. For all datasets ~50% of all drawdowns last only one day, meaning that a price decrease from the previous to the current day is followed by a price increase on the next following day. This confirms the low autocorrelation identified earlier. The longest drawdowns last around 10-12 business days. However, the longest drawdwons are not necessasary responsible for the highest losses which will be apparent when we identify crashes.

# %%
##### Plot frequency oof drawdowns
rcParams['figure.figsize'] = 10, 4.5
for d1, t in zip(l_dict_dd, plt_titles):
    df_d_bins = pd.DataFrame(list(d1.items()))
    df_d_bins.columns = ['drawdown', 'n']
    plt.scatter(df_d_bins['drawdown'], df_d_bins['n'], s=40, color='red', alpha=0.6)
    plt.yscale('log')
    df_d_bins_reg = df_d_bins[df_d_bins['drawdown']>-0.08]
    m, c = np.polyfit(df_d_bins_reg['drawdown'], np.log(df_d_bins_reg['n']), 1)
    y_fit = np.exp(m*df_d_bins['drawdown'] + c)
    plt.ylim(bottom=10**-.1)
    #plt.plot(df_d_bins['drawdown'], y_fit, color='black', ls='dashed')
    plt.title(t + ' - Frequency all drawdowns')
    plt.xlabel('Drawdown loss')
    plt.ylabel('Freqyency (log)')
    plt.grid()
    plt.show()

# %% [markdown]
# The frequency distribution plot of drawdowns shows that extreme drawdowns (> ~15%) occur on rare instances. While such large drawdowns only occured two times over nearly 70 years in the S&P, they occur much more frequently in the Indian (BSESN) or Brazilian (BVSP) market. These extreme events  are associated with crashes. Under 3. we will introduce two different methods for rules to set a threshold for drawdowns that identifies crashes.

# %%
##### Drawdown by rank
for dd, t in zip(dd_df, plt_titles):
    plt.scatter(dd['rank'], abs(dd['drawdown']), s=10*dd['duration'], alpha=0.5,\
                color='red')
    plt.xscale('log')
    plt.title(t + ' - Rank ordering of all drawdowns')
    plt.xlabel('Drawdown rank (log)')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

# %% [markdown]
# For rank ordering plots above, the drawdwons have been ranked from 1 (largest drawdown in dataset) to n. The size of each bubble corresponds to the duration of each drawdown and show that the largest drawdowns are not necessarily the longest ones. These plots provide further visual evidence of the existence of outliers as drawdowns that are larger than expected.

# %%
##### Fit Weibull exponential function to drawdowns by rank
def weibull(x, chi, z):
    return np.exp(-abs(x/chi)**z)

for dd, t in zip(dd_df, plt_titles):
    x = dd['drawdown']
    y = dd['rank']/dd['rank'].max()
    init_vals = [0.9, 0.015]  # for [z, chi]
    best_vals, covar = curve_fit(weibull, abs(x), y, p0=init_vals)
    chi = best_vals[0]
    z = best_vals[1]
    plt.scatter(abs(x), y, s=10*dd['duration'], alpha=0.5, color='red')
    y_fit = [weibull(abs(xi), chi, z) for xi in x]
    plt.plot(abs(x), y_fit, color='black', ls='dashed')
    plt.yscale('log')
    plt.ylim(bottom=10**-4)
    plt.legend(['Drawdowns', 'Weibull fit'])
    plt.title(t + ' - fit Weibull distribution to drawdowns')
    plt.xlabel('Drawdown loss')
    plt.ylabel('Drawdown rank (log, normalized)')
    plt.grid()
    plt.show()

# %% [markdown]
# The Weibull exponential model: y ~ exp(-abs(x/chi)^z) has been used by Johansen and Sornette (2001) to fit the distributions of drawdowns by rank. As we weill discuss below, large deviations from the Weibull distribution are considered to be crashes.

# %% [markdown]
# ### 3. Identify Crashes
# - First methodology: crashes as the 99.5% empirical quantile of the drawdowns (as suggested by Jacobsson, E., Stockholm University, in 'How to predict crashes in financial markets with the Log-Periodic Power Law', 2009).
# - Second methodology: Crashes as outliers of the fitted Weibull exponential model(as suggested by Johansen, A. and Sornette, D. in 'Large Stock Market Price Drawdowns Are Outliers', 2001). This methodology requires manual identification of crashes based on the Weibull plots.

# %% [markdown]
# #### 3.1 Crashes according to Jaccobsen

# %%
##### 2.1 Emilie Jacobsen, Stockholm University: empirical quantile: 99.5%
crash_thresholds = []
for dd in dd_df:
    ct = dd['drawdown'].iloc[round(dd.shape[0] * .005)]
    crash_thresholds.append(ct)

crashes = []
for df, dd, ct in zip(datasets, dd_df, crash_thresholds):
    df_d = dd.reindex(df.index).fillna(0)
    df_d = df_d.sort_values(by='Date')
    df_c = df_d[df_d['drawdown'] < ct]
    df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'duration', 'rank']
    crashes.append(df_c)
df_combined = []  
for i in range(len(datasets)):
    df_combined.append(pd.concat([datasets[i], dd_df[i]], axis=1).fillna(0))

for c, t in zip(crashes, plt_titles):
    c['crash_st'] = c['crash_st'].dt.date
    c['crash_end'] = c['crash_end'].dt.date
    c['duration'] = c['duration'].astype(int)
    c['rank'] = c['rank'].astype(int)
    print(t + ' - all crashes (99.5% drawdown quantile):')
    display(c)
    print('\n')

# %%
##### 2.1 Plot crashes in time series
rcParams['figure.figsize'] = 10, 6
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1]) 
for i in range(len(df_combined)):
    plt.subplot(gs[0])
    plt.plot(df_combined[i]['norm'], color='blue')
    [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crashes[i]['crash_st'], \
         crashes[i]['crash_end'])]
    plt.plot(df_combined[i]['drawdown'], color='red', marker='v',linestyle='')
    plt.title(plt_titles[i] + ' - crashes: 99.5% drawdown quantile')
    plt.grid()
    plt.xticks([])
    plt.legend(['Price', 'Drawdown'])
    plt.subplot(gs[1])
    plt.plot(df_combined[i]['vol'])
    [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crashes[i]['crash_st'], \
         crashes[i]['crash_end'])]
    plt.legend(['Volatility'])
    plt.grid()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# The plots above show the price (upper plots) and price return volatility with a lag of ten days (lower plots) along with the identified crases (vertical red lines) and the magnitude of the drawdowns (red pointers). Through the method of defining a crash as a drawdown in the 99.5th precentile, the number of crashes in each dataset is related to overall time period of each dataset. This leads to an identified crash occuring on average once 2-3 years. By identifying crashes through this methodology, the drawdown threshold for identifying a crash varies strongly with markets and we do not account for the fact that some markets have much more extreme large drawdowns than others.

# %% [markdown]
# #### 2.2 Crashes according to Johansen and Sornette

# %%
n_crashes = [3, 6, 3, 5, 8, 4, 11]  # <-- number of crashes manually identified based on outliers in Weibul plots below
rcParams['figure.figsize'] = 10, 4.5
for dd, t, n in zip(dd_df, plt_titles, n_crashes):
    x = dd['drawdown']
    y = dd['rank']/dd['rank'].max()
    init_vals = [0.9, 0.015]  # for [z, chi]
    best_vals, covar = curve_fit(weibull, abs(x), y, p0=init_vals)
    chi = best_vals[0]
    z = best_vals[1]
    plt.scatter(abs(x[n:]), y[n:], s=10*dd['duration'][n:], alpha=0.5, color='red')
    plt.scatter(abs(x[:n]), y[:n], s=10*dd['duration'][:n], alpha=0.5, color='black', marker='x')
    y_fit = [weibull(abs(xi), chi, z) for xi in x]
    plt.plot(abs(x), y_fit, color='black', ls='dashed')
    plt.yscale('log')
    plt.ylim(bottom=10**-4)
    plt.legend(['Drawdowns', 'Weibull fit', 'Outliers'])
    plt.title(t + ' - fit Weibull distribution to drawdowns')
    plt.xlabel('Drawdown loss')
    plt.ylabel('Drawdown rank (log, normalized)')
    plt.grid()
    plt.show()

# %% [markdown]
# The Weibull fit plots above are the same ones as shown under 2. but this time with the "x"s identifying outliers that cannot be explained by the Weibull distribution. As Johansen and Sornette do not give a specific threshold deviation from the distribution that identifies a crash, the identification of crashes above has been conducted based on visual interpretation.

# %%
crashes = []
for df, dd, r in zip(datasets, dd_df, n_crashes):
    df_c = dd[dd['rank'] <= r]
    df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'duration', 'rank']
    crashes.append(df_c)

for c, t in zip(crashes, plt_titles):
    c['crash_st'] = c['crash_st'].dt.date
    c['crash_end'] = c['crash_end'].dt.date
    c['duration'] = c['duration'].astype(int)
    c['rank'] = c['rank'].astype(int)
    print(t + ' - all crashes (Weibull outliers):')
    display(c)
    print('\n')

# %%
##### 2.2 Plot crashes in time series
rcParams['figure.figsize'] = 10, 6
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1]) 
for i in range(len(df_combined)):
    plt.subplot(gs[0])
    plt.plot(df_combined[i]['norm'], color='blue')
    [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crashes[i]['crash_st'], \
         crashes[i]['crash_end'])]
    plt.plot(df_combined[i]['drawdown'], color='red', marker='v',linestyle='')
    plt.title(plt_titles[i] + ' - crashes: Weibull outliers')
    plt.grid()
    plt.xticks([])
    plt.legend(['Price', 'Drawdown'])
    plt.subplot(gs[1])
    plt.plot(df_combined[i]['vol'])
    [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crashes[i]['crash_st'], \
         crashes[i]['crash_end'])]
    plt.legend(['Volatility'])
    plt.grid()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Identifying crashes based on drawdown outliers results in a number of crahes that doesn't necessarily correspond to the total number of drawdowns in a dataset. There are for example only three identified crashes over 68 years in the S&P whereas there are 11 crashes in just 16 years of BVSP.

# %% [markdown]
# #### Conclusion
# Since there is no consensus on the exact definition of a financial crash, both, the method introduced by Jacobsson (99.5% quantile of drawdowns) and the method introduced by Johansen and Sornette (outliers identified with the Weibull exponential model) can be used as an approach to identify crashes. For this project, I will be using the quantile method as it is unambiguous (no manual interpretation of outliers required) and leads to a number of crashes which proportional to the length of each dataset and therefore reduces the risk of overfitting crash patterns in certain datasets.


