#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns



# Get the list of FTSE 100 constituent stocks
ftse100_url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
ftse100_tables = pd.read_html(ftse100_url)
ftse100_components = ftse100_tables[4]  # The fifth table contains information on the constituent stocks



stocks = ftse100_components["Ticker"].tolist()



# Download data on FTSE 100 stocks
start_date = '2023-05-01'
end_date = '2024-05-01'
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']



# Remove failed download companies
data = data.dropna(axis=1)




#data




# Calculate the daily return for each stock
returns = data.pct_change().dropna()




# Calculate the correlation coefficient between all stock pairs
correlations = returns.corr()

# update stocks
stocks = data.columns.tolist()

# Find the most correlated stock pairs
max_correlation = 0
selected_pair = ()
for i in range(len(stocks)):
    for j in range(i+1, len(stocks)):
        pair_correlation = correlations.iloc[i, j]
        if pair_correlation > max_correlation:
            max_correlation = pair_correlation
            selected_pair = (stocks[i], stocks[j])

print("The stock pairs with the highest correlation are:", selected_pair, "with correlation coefficient:", max_correlation)


# - CPG: Compass Group, Support services
# - MRO: Melrose Industries, Aerospace & defence



sns.heatmap(correlations)
plt.savefig('heatmap.png')




select=dict()
for i in range(len(correlations)):
    for j in range(len(correlations.columns)):
        if (correlations.iloc[i,j]>.6 and correlations.iloc[i,j]!=1 
            and "{}-{}".format(correlations.index[i],correlations.columns[j]) not in select.keys() 
            and "{}-{}".format(correlations.columns[j],correlations.index[i]) not in select.keys()):
#             print(correlations.index[i],'\t',correlations.columns[j],'\t',correlations.iloc[i,j],)
            select["{}-{}".format(correlations.index[i],correlations.columns[j])]=correlations.iloc[i,j]
select


# SHEL: Shell plc, Oil & gas producers



# Calculate returns and return ratios
shel_returns = []
mro_returns = []
shel_ratios = []
mro_ratios = []

for i in range(len(data)):
    shel_return = data['SHEL'][i] - data['SHEL'][0]
    mro_return = data['MRO'][i] - data['MRO'][0]
    shel_returns.append(shel_return)
    mro_returns.append(mro_return)
    
    shel_ratio = shel_return/data['SHEL'][0]
    mro_ratio = mro_return/data['MRO'][0]
    shel_ratios.append(shel_ratio)
    mro_ratios.append(mro_ratio)

data["shel_returns"] = shel_returns
data["mro_returns"] = mro_returns
data["shel_ratios"] = shel_ratios
data["mro_ratios"] = mro_ratios




# plot the returns including the ratio between them

fig, axs = plt.subplots(1, 2, figsize=(11, 4))


axs[0].plot(data["shel_returns"], label="SHEL")
axs[0].plot(data["mro_returns"], label="MRO")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Returns (actual profit)")
axs[0].set_title("Returns of SHEL and MRO")
axs[0].legend()

axs[1].plot(data["shel_ratios"], label="SHEL")
axs[1].plot(data["mro_ratios"], label="MRO")
#axs[1].plot(data["shel_ratios"]/data["mro_ratios"], label="ratio between them", color="red")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Return Ratios")
axs[1].set_title("Return Ratios of SHEL and MRO")
axs[1].legend()
plt.tight_layout()

plt.savefig("returns_ratios_shel_mro.png")
plt.show()




ratio = data['SHEL']/data['MRO']
# Calculate the mean of the ratio time series
mean_ratio = np.mean(ratio)
print("The mean of price ratio is: ", mean_ratio)

plt.figure(figsize=(8, 4))
plt.plot(ratio, label='Price Ratio (SHEL/MRO)')
plt.axhline(mean_ratio, color='red', label="Mean")
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.title("Price Ratio between SHEL and MRO")
plt.legend()

plt.savefig("price_ratio_shel_mro.png")
plt.show()


# # Strategy 1: Mean Reversion



# Perform the Augmented Dickey-Fuller test
result = adfuller(ratio)

# print results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))



# Calculate the Z-score of the ratio
data["std_ratio"] = np.std(ratio)
data["mean_ratio"] = mean_ratio
z_scores = (ratio - data["mean_ratio"]) / data["std_ratio"]

# Plotting the Z-score of a ratio
plt.plot(z_scores, label="Z-score")
plt.axhline(0, color='black')
plt.axhline(1, color='red')
plt.axhline(1.25, color='red')
plt.axhline(-1, color='green')
plt.axhline(-1.25, color='green')
plt.xlabel('Date')
plt.ylabel('Z-scores')
plt.title('Z-score of Ratio of SHEL and MRO')
plt.legend()

plt.savefig("zscore_price_ratio_shel_mro.png")
plt.show()



# Strategies
buy_slot =[]
sell_slot = []
buy_date = []
sell_date = []

for i in range(len(z_scores)):
    if z_scores[i] < -1.25:
        buy_date.append(z_scores.index[i].strftime('%Y-%m-%d %H:%M:%S').split(" ")[0])
    elif 1.25 > z_scores[i] > -1.25:
        if len(buy_date) != 0:
            buy_slot.append([buy_date[0], buy_date[-1]])
            buy_date.clear()
        if len(sell_date) != 0:
            sell_slot.append([sell_date[0], sell_date[-1]])
            sell_date.clear()
    else:
        sell_date.append(z_scores.index[i].strftime('%Y-%m-%d %H:%M:%S').split(" ")[0])


print(" Mean Reversion Strategy: \n -----------------------------------------------")
for i in range(len(buy_slot)):
    print("Buy (take long position) SHEL and Sell (take short position) MRO during", buy_slot[i][0], "and", buy_slot[i][1])
for i in range(len(sell_slot)):
    print("Sell (take short position) SHEL and Buy (take long position) MRO during", sell_slot[i][0], "and", sell_slot[i][1])




# Assume buy stocks when t=0, sell stocks when period ends, each time buy/sell 1 unit.
# there is only two valid trading points during the period: 
# 1. Buy (take long position) SHEL and Sell (take short position) MRO on 2023-07-27
# 2. Sell (take short position) SHEL and Buy (take long position) MRO on 2023-12-12

shel_sell = data["SHEL"].loc["2023-12-12 00:00:00"]
shel_profit = shel_sell - data["SHEL"][0]
print("SHEL has profit:", shel_profit)
shel_profit_rate = shel_profit/data["SHEL"][0]
print("SHEL has profit rate: ", shel_profit_rate*100, "%")

mro_profit = data["MRO"].loc["2023-07-27 00:00:00"] - data["MRO"][0] + data["MRO"][-1] - data["MRO"].loc["2023-12-12 00:00:00"]
print("MRO has profit:", mro_profit)
mro_profit_rate = mro_profit/data["MRO"][0]
print("MRO has profit rate: ", mro_profit_rate*100, "%")

print("Total profit is: ", shel_profit+mro_profit)
print("Total profit rate is: ", ((shel_profit+mro_profit)/(data["SHEL"][0]+data["MRO"][0]))*100, "%")


# # Strategy 2: Momentum/trend following



# Define long-term and short-term time periods
long_term_period = 28 # a month/4 wekks
short_term_period = 7 # a week

# Calculate long term MA
long_term_ma = ratio.rolling(window=long_term_period).mean()

# Calculate short term MA
short_term_ma = ratio.rolling(window=short_term_period).mean()

# print long and short term MA
#print("Long-term Moving Average:\n", long_term_ma)
#print("\nShort-term Moving Average:\n", short_term_ma)



#plt.figure(figsize=(8,5))
plt.plot(ratio, label='Price Ratio (SHEL/MRO)')
plt.axhline(mean_ratio, color='black', label="Mean")
plt.plot(long_term_ma, label='Long-term MA', color='red')
plt.plot(short_term_ma, label='Short-term MA', color='green', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.title("Momentum with Long and Short Term MA")
plt.legend()

plt.savefig("momentum.png")
plt.show()



# Strategies
num_buy = 0
num_sell = 0
buy_shel = []
sell_shel = []
buy_mro = []
sell_mro = []

print("Momentum Strategy: \n -----------------------------------------------")

for i in range(len(ratio)):
    if short_term_ma[i] > long_term_ma[i]:
        if num_buy == 0:
            print("Buy (take long position) SHEL and Sell (take short position) MRO on", 
              short_term_ma.index[i].strftime('%Y-%m-%d %H:%M:%S').split(" ")[0],
              ", with SHEL=", data["SHEL"][i].round(2), ", MRO=", data["MRO"][i].round(2))
            buy_shel.append(data["SHEL"][i])
            sell_mro.append(data["MRO"][i])
            num_buy += 1
            num_sell = 0
        else:
            continue
    elif short_term_ma[i] < long_term_ma[i]:
        if num_sell == 0:
            print("Sell (take short position) SHEL and Buy (take long position) MRO on", 
              short_term_ma.index[i].strftime('%Y-%m-%d %H:%M:%S').split(" ")[0], 
              ", with SHEL=", data["SHEL"][i].round(2), ", MRO=", data["MRO"][i].round(2) )
            buy_mro.append(data["MRO"][i])
            sell_shel.append(data["SHEL"][i])
            num_sell += 1
            num_buy = 0
    else:
        continue



# The first print of strategy is not valid because we only have 1 year data, and there is no cross point on 2023-06-08
# Remove first item of sell_shel and buy_mro forever.

sell_shel.pop(0)
buy_mro.pop(0)



# Assume buy stocks when t=0, sell stocks when period ends, each time buy/sell 1 unit.

shel_profit2 = data["SHEL"][-1] + sum(sell_shel) - sum(buy_shel[1:]) - data["SHEL"][0]
print("SHEL has profit:", shel_profit2)
shel_profit_rate2 = shel_profit2/data["SHEL"][0]
print("SHEL has profit rate:", shel_profit_rate2*100, "%")

mro_profit2 = sum(sell_mro) - sum(buy_mro) - data["MRO"][0]
print("MRO has profit:", mro_profit2)
mro_profit_rate2 = mro_profit2/data["MRO"][0]
print("MRO has profit rate:", mro_profit_rate2*100, "%")

print("Total profit is: ", shel_profit2+mro_profit2)
print("Total profit rate is: ", ((shel_profit2+mro_profit2)/(data["SHEL"][0]+data["MRO"][0]))*100, "%")


# # Combine the two strategies

# Because Strategy 1 performs well on MRO, Srategy 2 performs well on SHEL,
# but MRO has more returns rate than SHEL, so Strategy 1 get more total profit.


# Use Strategy 1 for MRO, Strategy 2 for SHEL 
print("Combined Strategy: \n -----------------------------------------------")
print("Total profit is: ", shel_profit2+mro_profit)
print("Total profit rate is: ", ((shel_profit2+mro_profit)/(data["SHEL"][0]+data["MRO"][0]))*100, "%")

