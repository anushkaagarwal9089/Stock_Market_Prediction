import yfinance as yf
import pandas as pd

# Defining the stock symbol and date range
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetching the historical stock price data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculating the daily returns
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()

# Drop the first row with NaN values
stock_data.dropna(inplace=True)

# Example: Moving Averages
stock_data['5_day_MA'] = stock_data['Adj Close'].rolling(window=5).mean()
stock_data['20_day_MA'] = stock_data['Adj Close'].rolling(window=20).mean()

# Droppikng the unnecessary columns
columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
stock_data.drop(columns=columns_to_drop, inplace=True)

# Save the preprocessed data to a CSV file
preprocessed_data_path = 'preprocessed_data.csv'
stock_data.to_csv(preprocessed_data_path, index=False)
