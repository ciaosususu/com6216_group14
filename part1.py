# -*- coding: utf-8 -*-
"""
# Part 1: Time Series Analysis

2.1 ARIMA part 1 (max. 3 marks)
You are given the time series in Table 2. Apply the ARIMA(2,1,1) model. For the coefficients, use the last 3 digits of one of the group’s student numbers. In the report, clearly state which digits you used for which component. Com- pute yˆ7 and yˆ8. Explain your approach in your report. In particular, explain the challenges/assumptions you have made for computing yˆ8. Make sure you write your own code, and provide the code as part of your submission including instructions how to run it.

非常高的 AR 系数
𝜙
1
=
5
ϕ
1
​
 =5 可能并不典型，并且可能导致实际场景中的不稳定或不收敛。给定的 MA 系数也异常高，这可能会对噪声项过度拟合或反应过度。误差对于动态调整ARIMA中的预测至关重要。在这里，由于缺乏实际
𝑡
=
6
t=6 数据，对
𝑦
7
y
7
​
   
𝑦
8
y
8
​
  模型参数的预测越来越依赖于模型的参数和先前计算的误差，这些误差可能无法准确反映现实世界的变化。
"""

import numpy as np
import pandas as pd
# Given data
t = np.array([1, 2, 3, 4, 5, 6]) # The value of Yt
yt = np.array([15, 10, 12, 17, 25, 23])

# Differencing the series
y_diff = np.diff(yt) # Calculate the difference of yt
# [10-15, 12-10, 17-12 ,25-17 ,23-25]
print(y_diff)

"""Since ARIMA(2,1,1), so the formula should be:
yt_prime = phi1 * yt-1_prime + phi2 * yt-2_prime + theta * epsilont-1 + epsilon t
"""

# ARIMA(2,1,1) model parameters
# Initialize the parameter by using last 3 digits (503)
phi = [5,0] # AR
theta = [3] # MA

# Initialize the (error) epsilon_t = 0
y_diff_hat = np.zeros(2)
epsilon = np.zeros(2)

# Forecasting y_diff_7
y_diff_hat[0] = phi[0]* y_diff[-1] + phi[1]*y_diff[-2] + theta[0]* epsilon[0]

# Forecasting y_diff_8
y_diff_hat[1] = phi[0]* y_diff_hat[0] + phi[1] * y_diff[-1] + theta[0] * epsilon[0]

# Since yt_diff = yt - yt-1, yt = yt_diff + yt-1
y_hat_7 = y_diff_hat[0] + yt[-1]
y_hat_8 = y_diff_hat[1] + y_hat_7

print(f"y_hat_7: {y_hat_7}")
print(f"y_hat_8: {y_hat_8}")

"""2.2 ARIMA part 2 (max. 4 marks)
Obtain daily prices from one or more random companies listed on the FTSE 100, e.g. through Yahoo! finance. Obtain daily data of stock prices for the past year at least. Split the data into a training and test set as appropriate. Only use the test set for final validation and not for parameter tuning.
Using the implementation from part 1, now apply this to the dataset. Use a simple approach to tune the weights such as random search, hill climbing, or gradient descent. Tune the hyperparameters of the ARIMA model using an appropriate approach. Evaluate the final result using appropriate metrics. Describe your approach and motivate your choices.

# Grid Search
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Step 1: Obtain daily prices for a random company listed on the FTSE 100
ticker = 'AAL'
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

# Step 2: Preprocess the data
data = data['Close'].dropna()
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Step 3: Define a function to apply ARIMA(2,1,1) model
def arima_forecast(train, p, d, q):
    # Differencing
    diff = np.diff(train, n=d)

    # Initialize
    y_diff_hat = np.zeros(len(test))
    epsilon = np.zeros(len(test))

    # AR and MA coefficients (random initialization)
    phi = np.random.randn(p)
    theta = np.random.randn(q)

    # Forecasting
    for t in range(len(test)):
        y_diff_hat[t] = np.dot(phi, diff[-p:]) + np.dot(theta, epsilon[-q:])
        if t > 0:
            epsilon[t] = test.iloc[t] - y_diff_hat[t-1]

    # Reconstruct the forecasted values
    y_hat = np.zeros(len(test))
    y_hat[0] = y_diff_hat[0] + train.iloc[-1]
    for t in range(1, len(test)):
        y_hat[t] = y_diff_hat[t] + y_hat[t-1]

    return y_hat

# Step 4: Tuning the hyperparameters
def tune_arima(train, test, p_values, d_values, q_values):
    best_rmse = float('inf')
    best_params = None
    best_forecast = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    forecast = arima_forecast(train, p, d, q)
                    rmse = np.sqrt(mean_squared_error(test, forecast))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (p, d, q)
                        best_forecast = forecast
                except Exception as e:
                    continue

    return best_params, best_rmse, best_forecast

# Step 5: Define the hyperparameter ranges and tune the model
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_params, best_rmse, best_forecast = tune_arima(train, test, p_values, d_values, q_values)

# Step 6: Evaluate the final result
print(f"Best Parameters: p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
print(f"Best RMSE: {best_rmse}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, best_forecast, label='Forecasted')
plt.legend()
plt.title(f'ARIMA{best_params} Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

"""# Random Search"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random

# Step 1: Obtain daily prices for a random company listed on the FTSE 100
ticker = 'VOD'  # Vodafone Group, as an example
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

# Step 2: Preprocess the data
data = data['Close'].dropna()
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Step 3: Define a function to apply ARIMA model
def arima_forecast(train, test, p, d, q):
    # Differencing
    diff = np.diff(train, n=d)

    # Initialize
    y_diff_hat = np.zeros(len(test))
    epsilon = np.zeros(len(test))

    # AR and MA coefficients (random initialization)
    phi = np.random.randn(p)
    theta = np.random.randn(q)

    # Forecasting
    for t in range(len(test)):
        if p > 0 and q > 0:
            y_diff_hat[t] = np.dot(phi, diff[-p:]) + np.dot(theta, epsilon[-q:])
        elif p > 0:
            y_diff_hat[t] = np.dot(phi, diff[-p:])
        elif q > 0:
            y_diff_hat[t] = np.dot(theta, epsilon[-q:])

        if t > 0:
            epsilon[t] = test.iloc[t] - y_diff_hat[t-1]

    # Reconstruct the forecasted values
    y_hat = np.zeros(len(test))
    y_hat[0] = y_diff_hat[0] + train.iloc[-1]
    for t in range(1, len(test)):
        y_hat[t] = y_diff_hat[t] + y_hat[t-1]

    return y_hat

# Step 4: Random Search for hyperparameter tuning
def random_search(train, test, p_values, d_values, q_values, iterations=100):
    best_rmse = float('inf')
    best_params = None
    best_forecast = None

    for _ in range(iterations):
        p = random.choice(p_values)
        d = random.choice(d_values)
        q = random.choice(q_values)

        try:
            forecast = arima_forecast(train, test, p, d, q)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (p, d, q)
                best_forecast = forecast
        except Exception as e:
            continue

    return best_params, best_rmse, best_forecast

# Define the hyperparameter ranges
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_params_random, best_rmse_random, best_forecast_random = random_search(train, test, p_values, d_values, q_values, iterations=100)

# Evaluate the final result
print(f"Best Parameters (Random Search): p={best_params_random[0]}, d={best_params_random[1]}, q={best_params_random[2]}")
print(f"Best RMSE (Random Search): {best_rmse_random}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, best_forecast_random, label='Forecasted')
plt.legend()
plt.title(f'ARIMA{best_params_random} Forecast vs Actual (Random Search)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

"""# Hill Climbing"""

# Step 4: Hill Climbing for hyperparameter tuning
def hill_climbing(train, test, p_values, d_values, q_values):
    best_rmse = float('inf')
    best_params = (random.choice(p_values), random.choice(d_values), random.choice(q_values))
    best_forecast = None

    improved = True
    while improved:
        improved = False
        for dp in [-1, 0, 1]:
            for dd in [-1, 0, 1]:
                for dq in [-1, 0, 1]:
                    if dp == 0 and dd == 0 and dq == 0:
                        continue
                    p = best_params[0] + dp
                    d = best_params[1] + dd
                    q = best_params[2] + dq
                    if p in p_values and d in d_values and q in q_values:
                        try:
                            forecast = arima_forecast(train, test, p, d, q)
                            rmse = np.sqrt(mean_squared_error(test, forecast))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params = (p, d, q)
                                best_forecast = forecast
                                improved = True
                        except Exception as e:
                            continue

    return best_params, best_rmse, best_forecast

best_params_hill, best_rmse_hill, best_forecast_hill = hill_climbing(train, test, p_values, d_values, q_values)

# Evaluate the final result
print(f"Best Parameters (Hill Climbing): p={best_params_hill[0]}, d={best_params_hill[1]}, q={best_params_hill[2]}")
print(f"Best RMSE (Hill Climbing): {best_rmse_hill}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, best_forecast_hill, label='Forecasted')
plt.legend()
plt.title(f'ARIMA{best_params_hill} Forecast vs Actual (Hill Climbing)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

"""# Gradient Descent"""

def gradient_descent(train, test, p_values, d_values, q_values, learning_rate=0.01, iterations=100):
    best_params = None
    best_rmse = float('inf')
    best_forecast = None

    for _ in range(10):  # Run multiple times to mitigate initial parameter effects
        p = random.choice(p_values)
        d = random.choice(d_values)
        q = random.choice(q_values)

        current_rmse = float('inf')
        current_forecast = None

        for _ in range(iterations):
            try:
                forecast = arima_forecast(train, test, p, d, q)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                if rmse < current_rmse:
                    current_rmse = rmse
                    current_forecast = forecast

                # Gradient approximation
                p = max(min(p + int(learning_rate * (np.random.randn() - 0.5)), max(p_values)), min(p_values))
                d = max(min(d + int(learning_rate * (np.random.randn() - 0.5)), max(d_values)), min(d_values))
                q = max(min(q + int(learning_rate * (np.random.randn() - 0.5)), max(q_values)), min(q_values))
            except Exception as e:
                continue

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = (p, d, q)
            best_forecast = current_forecast

    return best_params, best_rmse, best_forecast

# Define the hyperparameter ranges
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_params_gd, best_rmse_gd, best_forecast_gd = gradient_descent(train, test, p_values, d_values, q_values)

# Evaluate the final result
print(f"Best Parameters (Gradient Descent): p={best_params_gd[0]}, d={best_params_gd[1]}, q={best_params_gd[2]}")
print(f"Best RMSE (Gradient Descent): {best_rmse_gd}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, best_forecast_gd, label='Forecasted')
plt.legend()
plt.title(f'ARIMA{best_params_gd} Forecast vs Actual (Gradient Descent)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
