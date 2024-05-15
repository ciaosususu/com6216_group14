{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%%**********************]  3 of 3 completed"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ticker             AZN.L        BP.L      TSCO.L\n",
      "Date                                            \n",
      "2023-01-03  11123.242188  456.687164  220.133682\n",
      "2023-01-04  11188.820312  440.152527  225.901382\n",
      "2023-01-05  11292.974609  445.727081  229.265869\n",
      "2023-01-06  11362.411133  450.734680  232.053589\n",
      "2023-01-09  11318.047852  452.860565  233.783890\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "\n",
    "# Define the stock symbols\n",
    "symbols = ['AZN.L', 'BP.L', 'TSCO.L']\n",
    "\n",
    "# Fetch historical data from Yahoo Finance\n",
    "data = yf.download(symbols, start=\"2023-01-01\", end=\"2024-01-01\")['Adj Close']\n",
    "\n",
    "print(data.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data\n",
    "split_index = len(data) // 2\n",
    "train_data = data[:split_index]\n",
    "test_data = data[split_index:]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ticker\n",
      "AZN.L    -0.000644\n",
      "BP.L      0.000150\n",
      "TSCO.L    0.001036\n",
      "dtype: float64\n",
      "Ticker         AZN.L      BP.L        TSCO.L\n",
      "Ticker                                      \n",
      "AZN.L   1.903355e-04  0.000009  4.879081e-07\n",
      "BP.L    9.435338e-06  0.000386  5.029067e-05\n",
      "TSCO.L  4.879081e-07  0.000050  1.130869e-04\n"
     ]
    }
   ],
   "source": [
    "# Calculate daily returns\n",
    "daily_returns = train_data.pct_change().dropna()\n",
    "\n",
    "# Estimate returns and covariance matrix\n",
    "average_returns = daily_returns.mean()\n",
    "covariance_matrix = daily_returns.cov()\n",
    "\n",
    "print(average_returns)\n",
    "print(covariance_matrix)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Weights: [0. 0. 1.]\n",
      "Equal Weights: [0.33333333 0.33333333 0.33333333]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Brute force search for efficient portfolio\n",
    "weights = np.linspace(0, 1, 101)\n",
    "best_sharpe = -np.inf\n",
    "best_weights = None\n",
    "\n",
    "for w1 in weights:\n",
    "    for w2 in weights[weights <= 1 - w1]:\n",
    "        w3 = 1 - w1 - w2\n",
    "        portfolio_weights = np.array([w1, w2, w3])\n",
    "        expected_return = np.dot(portfolio_weights, average_returns)\n",
    "        expected_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(covariance_matrix, portfolio_weights)))\n",
    "        sharpe_ratio = expected_return / expected_volatility\n",
    "        if sharpe_ratio > best_sharpe:\n",
    "            best_sharpe = sharpe_ratio\n",
    "            best_weights = portfolio_weights\n",
    "\n",
    "# 1/n portfolio\n",
    "equal_weights = np.array([1/3, 1/3, 1/3])\n",
    "\n",
    "print(\"Best Weights:\", best_weights)\n",
    "print(\"Equal Weights:\", equal_weights)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Efficient Portfolio Test Return: 0.17948429258385212\n",
      "1/n Portfolio Test Return: 0.07571731308172806\n"
     ]
    }
   ],
   "source": [
    "# Calculate test returns\n",
    "test_returns = test_data.pct_change().dropna()\n",
    "\n",
    "# Performance evaluation\n",
    "efficient_return = (test_returns * best_weights).sum(axis=1)\n",
    "equal_return = (test_returns * equal_weights).sum(axis=1)\n",
    "\n",
    "print(\"Efficient Portfolio Test Return:\", efficient_return.sum())\n",
    "print(\"1/n Portfolio Test Return:\", equal_return.sum())\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
