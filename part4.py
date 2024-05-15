# %%
# %pip install yfinance

# %%
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from statsmodels.graphics.tsaplots import plot_acf


# %%
# Get stock data (for example, select a stock in the FTSE 100)
stock = 'AAPL' 
data = yf.download(stock, start='2023-01-01', end='2024-01-01')

# Calculate daily return
data['Return'] = data['Adj Close'].pct_change().dropna()

# Check for missing values and remove
missing_values = data['Return'].isna().sum()
print(f"Number of missing values in returns: {missing_values}")
data = data.dropna(subset=['Return'])


# %% [markdown]
# Heavy tail analysis

# %%
# Plot return distribution histograms and KDE curves
sns.histplot(data['Return'], kde=True)
plt.title('Distribution of Returns')
plt.xlabel('Return')
plt.ylabel('Count')
plt.show()

# Calculate and print kurtosis
kurt_value = kurtosis(data['Return'], fisher=False)
print("Kurtosis of Returns:", kurt_value)


# %% [markdown]
# Autocorrelation Analysis

# %%
# Plot the autocorrelation function
plot_acf(data['Return'], lags=30)
plt.title('Autocorrelation of Returns')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()


# %% [markdown]
# Volatility Clustering

# %%
# Plot a time series plot of returns and squared returns
data['Squared Return'] = data['Return'] ** 2
plt.figure(figsize=(10, 6))
plt.plot(data['Return'], label='Returns')
plt.plot(data['Squared Return'], label='Squared Returns')
plt.legend()
plt.title('Returns and Squared Returns')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()





