import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Fetch Apple stock data from Yahoo Finance
ticker = 'GLD'
data = yf.download(ticker, start='2015-01-01', end='2023-06-15')

# Use the 'Close' price for analysis
df = data[['Close']].reset_index()
df.columns = ['date', 'value']

# Split data into training and testing sets
split_date = '2022-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Plotting the density
plt.figure(figsize=(12, 6))

# Training data density
sns.kdeplot(train['value'], label='Training Data', fill=True)

# Testing data density
sns.kdeplot(test['value'], label='Testing Data', fill=True)

plt.title('Density Plot of Apple Stock Training and Testing Data')
plt.xlabel('Close Price')
plt.ylabel('Density')
plt.legend()
plt.show()
