import numpy as np
import pandas as pd
import yfinance as yf
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_stock(start, end, stock):
    '''
    Return the data as Numpy arrays and incrementing indexes for each value for the time series.

    Parameters:
    -- start: start date "Year-Month-Day"
    -- end: end date "Year-Month-Day"
    -- stock: The label of the stock to download from yahoo stocks

    Returns:
    -- df: 1d Numpy array containing all the closing prices from the start to end data except for the last 100 days
    -- index: incrementing indexes for the closing prices
    '''

    yfd = yf.download(stock, start=start, end=end)
    df = np.array(yfd['Close']).reshape(-1, 1)
    index = np.linspace(1, df.shape[0], df.shape[0])

    return df, index

def bin_data(df, bin_size=365):
    '''
    Returns the data split into bins of size <bin_size>. The bins are used to predict the day on index <bin_size> + 1.
    
    Parameters:
    -- df: dataset containing the closing prices for the stock
    -- bin_size: size of bins for the data

    Returns:
    -- bins: Numpy array size (len(df) - bin_size, bin_size), bins created based on a sliding window incrementing by 1
    '''
    
    bins = np.array([df[i:i + bin_size].T.reshape(-1) for i in range(0, len(df) - bin_size - 1, 1)])
    return bins 

def split_data(bins, df, index, bin_size):
    '''
    Splits data into training and testing sets, maintaining the relationship between bins and targets.

    Parameters:
    -- bins: Numpy array of binned data (features).
    -- df: Original dataset (target values).
    -- index: Corresponding indices of the dataset.
    -- bin_size: Size of the bins (used to offset the target values).
    '''
    targets = df[bin_size:bin_size + len(bins)].T.reshape(-1)
    indices = index[bin_size:bin_size + len(bins)]

    # Use sklearn's train_test_split
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        bins, targets, indices, test_size=0.2, random_state=42, shuffle=False
    )

    return X_train, y_train, X_test, y_test, train_index, test_index, len(X_train), len(X_test)
    

def test_stock(stock, restrict=0, bin_size=365):

    # Get the stock data
    df, index = get_stock(stock[1], stock[2], stock[0])

    # Create bins
    bins = bin_data(df, bin_size)

    # Split the data
    X_train, y_train, X_test, y_test, train_index, test_index, train_size, test_size = split_data(bins, df, index, bin_size)

    # Run the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = np.floor(model.predict(X_test))
    y_test = np.floor(y_test)
    
    
    plt.plot(test_index[-100:], y_test[-100:], label="actual")
    plt.plot(test_index[-100:], y_pred[-100:], color="red", label="predictions")
    plt.title("Test data")
    plt.ylabel("Price")
    plt.xlabel("Days")
    plt.legend()
    plt.show()