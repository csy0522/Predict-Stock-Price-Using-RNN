#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:18:50 2019

@author: csy
"""

from RNN import RNN
import HELPER_FUNC


if __name__ == "__main__":

    """
    Dataset
    """
    data = HELPER_FUNC.get_data("AAPL.csv")

    """
    Each Numerical Label from the Dataset
    """
    open_, high_, low_, close_, adj_close_, volume_ = HELPER_FUNC.separate_numerical_data(
        data
    )

    """
    Hyper Parameters
    """
    past_n = [30, 30, 60, 60]
    batch_size_ = [30, 60, 30, 60]
    learning_rate_ = 0.001
    epochs_ = 10
    test_size = 0.2

    """
    Base Model
        &
    Staked Model
    """
    sequential = [
        [HELPER_FUNC.lstm(30, False), HELPER_FUNC.dropout(0.15), HELPER_FUNC.dense(1)],
        [HELPER_FUNC.lstm(30, False), HELPER_FUNC.dropout(0.15), HELPER_FUNC.dense(1)],
        [
            HELPER_FUNC.lstm(40, True),
            HELPER_FUNC.dropout(0.2),
            HELPER_FUNC.lstm(20, False),
            HELPER_FUNC.dropout(0.1),
            HELPER_FUNC.dense(1),
        ],  # Stacked Model
        [
            HELPER_FUNC.lstm(40, True),
            HELPER_FUNC.dropout(0.2),
            HELPER_FUNC.lstm(20, False),
            HELPER_FUNC.dropout(0.1),
            HELPER_FUNC.dense(1),
        ],  # Stacked Model
    ]

    """
    This section creates a RNN class for each iteration,
    builds training set and test set based on the class variables,
    and trains the model.
    It iterates a total of 4 timse (creates 4 different classes)
    """
    for i in range(4):

        rnn = RNN(
            past_n=past_n[i],
            learning_rate=learning_rate_,
            batch_size=batch_size_[i],
            epochs=epochs_,
            sequential=sequential[i],
        )

        """ Format Training and Testing Set """
        open_train_X, open_test_X, open_train_y, open_test_y = HELPER_FUNC.get_normalized_training_test_set(
            open_, rnn.past_n_, test_size
        )
        high_train_X, high_test_X, high_train_y, high_test_y = HELPER_FUNC.get_normalized_training_test_set(
            high_, rnn.past_n_, test_size
        )
        low_train_X, low_test_X, low_train_y, low_test_y = HELPER_FUNC.get_normalized_training_test_set(
            low_, rnn.past_n_, test_size
        )
        close_train_X, close_test_X, close_train_y, close_test_y = HELPER_FUNC.get_normalized_training_test_set(
            close_, rnn.past_n_, test_size
        )

        """ Train """
        rnn.__train__(open_train_X, open_train_y)
        rnn.__train__(high_train_X, high_train_y)
        rnn.__train__(low_train_X, low_train_y)
        rnn.__train__(close_train_X, close_train_y)

        """ Plot The total Loss """
        rnn.__plot_total_loss__()

        """ Plot Predicted & Actual Data """
        rnn.__test__(open_test_X)
        rnn.__plot_prediction__(open_test_X, open_test_y, open_, "OPEN")

        rnn.__test__(high_test_X)
        rnn.__plot_prediction__(high_test_X, high_test_y, high_, "HIGH")

        rnn.__test__(low_test_X)
        rnn.__plot_prediction__(low_test_X, low_test_y, low_, "LOW")

        rnn.__test__(close_test_X)
        rnn.__plot_prediction__(close_test_X, close_test_y, close_, "CLOSE")
