import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def plot_time_series(data, volume, train_start=None, train_end=None, val_start=None, val_end=None, test_start=None,
                     test_end=None, title=None, ylabel=None):
    """
    Plots a time series data with separate colors for training, validation, and testing sets.
    Additionally, it plots the trading volume as a subplot, and MACD as another subplot.

    Args:
        data (pd.Series or pd.DataFrame): Time series data.
        volume (pd.Series): Trading volume data.
        train_start (str or int, optional): Start index for the training set.
        train_end (str or int, optional): End index for the training set.
        val_start (str or int, optional): Start index for the validation set.
        val_end (str or int, optional): End index for the validation set.
        test_start (str or int, optional): Start index for the testing set.
        test_end (str or int, optional): End index for the testing set.
        title (str, optional): Title for the plot.
        ylabel (str, optional): Label for the y-axis.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot price data with Moving Averages
    if train_start is not None and train_end is not None:
        train_data = data.loc[train_start:train_end]
        ax1.plot(train_data.index, train_data.values, label='Training', color='green')

    if val_start is not None and val_end is not None:
        val_data = data.loc[val_start:val_end]
        ax1.plot(val_data.index, val_data.values, label='Validation', color='orange')

    if test_start is not None and test_end is not None:
        test_data = data.loc[test_start:test_end]
        ax1.plot(test_data.index, test_data.values, label='Testing', color='red')

    MA5 = data.rolling(window=5).mean().dropna()
    MA10 = data.rolling(window=10).mean().dropna()
    MA20 = data.rolling(window=20).mean().dropna()

    ax1.plot(data.index[4:], MA5, label='MA5', color='purple')
    ax1.plot(data.index[9:], MA10, label='MA10', color='brown')
    ax1.plot(data.index[19:], MA20, label='MA20', color='gray')

    ax1.set_xlabel('Date')
    if ylabel:
        ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.set_title(title)

    # Plot trading volume
    ax2.plot(volume.index, volume.values, color='black')
    ax2.bar(volume.index, volume.values, color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')

    # Plot MACD
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    ax3.plot(macd.index, macd.values, label='MACD', color='blue')
    ax3.plot(signal.index, signal.values, label='Signal', color='red')
    ax3.bar(macd.index, macd - signal, label='Histogram', color='gray')
    ax3.axhline(0, color='black', lw=1)
    ax3.fill_between(macd.index, macd.values, 0, where=(macd > 0), color='lightgreen', alpha=0.5)
    ax3.fill_between(macd.index, macd.values, 0, where=(macd < 0), color='lightcoral', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('MACD')
    ax3.legend()

    plt.show()


def plot_stock_data(ticker, start_date, end_date, train_ratio=0.7, val_ratio=0.2, title=None):
    """
    Plots stock data for a given ticker and date range, with separate colors for training, validation, and testing sets.
    Additionally, it plots the trading volume as a subplot.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for the data (YYYY-MM-DD format).
        end_date (str): End date for the data (YYYY-MM-DD format).
        train_ratio (float, optional): Ratio of data to be used for training.
        val_ratio (float, optional): Ratio of data to be used for validation.
        title (str, optional): Title for the plot.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Adj Close']
    volume = data['Volume']

    train_end = int(len(prices) * train_ratio)
    val_end = int(len(prices) * (train_ratio + val_ratio))

    plot_time_series(prices, volume, train_start=prices.index[0], train_end=prices.index[train_end - 1],
                     val_start=prices.index[train_end],
                     val_end=prices.index[val_end - 1], test_start=prices.index[val_end], test_end=prices.index[-1],
                     title=f"{ticker} Stock Prices" if title is None else title, ylabel='Price (USD)')

plot_stock_data('AAPL', '2020-01-01', '2024-04-05', title='Apple Stock Prices')
plot_stock_data('NVDA', '2020-01-01', '2024-04-05', title='NVDA Stock Prices')
plot_stock_data('TSLA', '2020-01-01', '2024-04-05', title='TSLA Stock Prices')
plot_stock_data('BTC-USD', '2020-01-01', '2024-04-05', title='BTC-USD Prices')