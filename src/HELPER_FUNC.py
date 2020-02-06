#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:17:30 2019

@author: csy
"""

import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf


"""
This file contains all the helper functions
used for training the sequential LSTM model
"""


"""
Returns Tensorflow LSTM layer
"""


def lstm(u, rs):
    return tf.keras.layers.LSTM(units=u, return_sequences=rs)


"""
Returns Tensorflow Dropout layer
"""


def dropout(r):
    return tf.keras.layers.Dropout(rate=r)


"""
Returns Tensorflow Dense layer
"""


def dense(u):
    return tf.keras.layers.Dense(units=u)


"""
Read csv file and return the data
"""


def get_data(filename):
    return pd.read_csv(filename)


"""
Normalize the dataset in the range of 0 to 1
"""


def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


"""
Convert the dataset back to the original range
"""


def denormalize(x, X):
    return x * (np.max(X) - np.min(X)) + np.min(X)


"""
This function creates and returns feature set and target set from the dataset,
and reshapes the training set for model's input format
"""


def get_past_n_xy(X, past_n):
    x = []
    y = []
    for i in range(past_n, len(X)):
        x.append(X[i - past_n : i])
        y.append(X[i])
    x = np.array(x)
    y = np.array(y)
    return x.reshape((x.shape[0], x.shape[1], 1)), y


"""
This function separates training set and test sets
"""


def get_normalized_training_test_set(dataset, past_n, test_size_):
    normalized = normalize(dataset)
    data_X, data_y = get_past_n_xy(normalized, past_n)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data_X, data_y, test_size=test_size_, random_state=0, shuffle=False
    )
    return X_train, X_test, y_train, y_test


"""
This function separates returns 
the numerical data to array from the entire dataset
"""


def separate_numerical_data(data):
    open_ = np.array(data.iloc[:, 1]).T
    high_ = np.array(data.iloc[:, 2]).T
    low_ = np.array(data.iloc[:, 3]).T
    close_ = np.array(data.iloc[:, 4]).T
    adj_close_ = np.array(data.iloc[:, 5]).T
    volume_ = np.array(data.iloc[:, 6]).T
    return open_, high_, low_, close_, adj_close_, volume_
