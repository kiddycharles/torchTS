import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Download historical data for NVDA and AAPL
nvda = yf.download("NVDA", start="2018-01-01", end="2023-01-01")
aapl = yf.download("AAPL", start="2018-01-01", end="2023-01-01")

# Select 'Close' prices
nvda_close = nvda['Close']
aapl_close = aapl['Close']

# Split data into training and test sets (80-20 split)
nvda_train = nvda_close[:int(0.8*len(nvda_close))]
nvda_test = nvda_close[int(0.8*len(nvda_close)):]
# Assuming 'train_data' is your training time series dataset, which is a 2D array (or DataFrame)
nvda_scaler = StandardScaler()
nvda_train_data_scaled = nvda_scaler.fit_transform(nvda_train.to_frame())
nvda_test_data_scaled = nvda_scaler.transform(nvda_test.to_frame())


aapl_train = aapl_close[:int(0.8*len(aapl_close))]
aapl_test = aapl_close[int(0.8*len(aapl_close)):]
aapl_scaler = StandardScaler()
aapl_train_data_scaled = aapl_scaler.fit_transform(aapl_train.to_frame())
aapl_test_data_scaled = aapl_scaler.transform(aapl_test.to_frame())

# Plot density distributions
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
sns.kdeplot(nvda_train, label='NVDA Train', fill=True)
sns.kdeplot(nvda_test, label='NVDA Test', fill=True)
plt.title('Density Distribution of NVDA Train and Test Data')
plt.xlabel('Close Price')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(aapl_train, label='AAPL Train', fill=True)
sns.kdeplot(aapl_test, label='AAPL Test', fill=True)
plt.title('Density Distribution of AAPL Train and Test Data')
plt.xlabel('Close Price')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
