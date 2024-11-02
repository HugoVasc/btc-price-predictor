import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance

# GET BTC - USD OHLCV DATA
btc = yfinance.Ticker('BTC-USD')
btc_hist = btc.history(start='2022-01-01')

# Adding moving averages 7 days, 25 days and 99 days
btc_hist['m_avg_7'] = btc_hist['Close'].rolling(window=7).mean()
btc_hist['m_avg_25'] = btc_hist['Close'].rolling(window=25).mean()
btc_hist['m_avg_99'] = btc_hist['Close'].rolling(window=99).mean()

# add column with pct_change
btc_hist['close_diff'] = btc_hist['Close'].pct_change()
btc_hist['m_avg_7_diff'] = btc_hist['m_avg_7'].pct_change()
btc_hist['m_avg_25_diff'] = btc_hist['m_avg_25'].pct_change()
btc_hist['m_avg_99_diff'] = btc_hist['m_avg_99'].pct_change()

btc_hist.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
btc_hist.dropna(inplace=True)

btc_hist.to_parquet('./btc_hist.parquet')