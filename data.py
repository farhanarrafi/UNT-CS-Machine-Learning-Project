# Data to be worked on is comprised of seven main categories

# Date: time period we are working on 
# High: highest price that a stock reached
# Low: lowest price that a stock reached 
# Open: price of stock soon as a market opens on a particular day
# Volume: number of shares traded 
# Closing Price: cost of shares agreed on by traders at the end of a particular day  
# Adjusted Close: cost of share at the end of a day, taking into consediration stock splits, new stock offerings and dividens 

# For this study, the critical datapoint we are most interested in and the pivot to this study is the Closing Price

# importing all necessary libraries and APIs

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

import datetime

import keras

from keras import Sequential

from keras import Model, Sequential 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 4)

# to ensure somewhat of more consistent results when training the model, we have to set a randomm seed.
# without setting a random seed, training the same model twice
# the number 42 was chosen as the random seed, as we have seen it set as the random seed in many previous studies

tf.random.set_seed(42)
np.random.seed(42)

# loading the data

dataset_stock_sp500_A = pd.read_csv('kaggle_dataset/stock_market_data/sp500/csv/A.csv')

# to check the first few rows (earliest data recorded)
print(dataset_stock_sp500_A.head())

# to check the last few rows (latest data recorded)
print(dataset_stock_sp500_A.tail())

# by analyzing the dates above, we can see when the data collection started and where it ended

# to check for missing data 
print(dataset_stock_sp500_A.isna().sum())


dataset_stock_sp500_AAL = pd.read_csv('kaggle_dataset/stock_market_data/sp500/csv/AAL.csv')
dataset_stock_sp500_AAP = pd.read_csv('kaggle_dataset/stock_market_data/sp500/csv/AAP.csv')
dataset_stock_sp500_AAPL = pd.read_csv('kaggle_dataset/stock_market_data/sp500/csv/AAPL.csv')
dataset_stock_sp500_ABBV = pd.read_csv('kaggle_dataset/stock_market_data/sp500/csv/ABBV.csv')

# since the data in the "A" contains the last 2 months of the year 1999, but all 12 months of all the years onwards, we are going to drop data from the year 1999

# converting date to datetime data type
dataset_stock_sp500_A['Date'] = pd.to_datetime(dataset_stock_sp500_A['Date'])

# dropping the rows where the year is 1999
dataset_stock_sp500_A = dataset_stock_sp500_A[dataset_stock_sp500_A['Date'].dt.year !=1999]

# to reset the DataFrame indeex
dataset_stock_sp500_A.reset_index(inplace=True, drop=True)

# to confirm that the year 1999 has been dropped
print(dataset_stock_sp500_A.head(5))

# to plot data from the first year (there are 250-260 working days in a year- we will take the higher limit)

fig, ax = plt.subplots(figsize=(10,6))

# defining x, y axises and plotting the closing price

ax.plot(dataset_stock_sp500_A['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')

# setting the range for x-values
ax.set_xlim(0,260)

fig.autofmt_xdate()
plt.tight_layout()

# to show the plot
plt.show()

# since we made adjustments to the original data file, we are goint to save the new data file, dropping the data from 1999
dataset_stock_sp500_A.to_csv('kaggle_dataset/stock_market_data/sp500/csv/A_new.csv', header=True, index=False)

# stopped at feature enineering 