#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:12:46 2019

@author: csy
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import HELPER_FUNC

"""
This class creates a Recurrent Neural Network model
that is specifically a Long Short Term Memory type
"""


class RNN:
    def __init__(self, past_n, learning_rate, batch_size, epochs, sequential):
        self.past_n_ = past_n
        self.learning_rate_ = learning_rate
        self.batch_size_ = batch_size
        self.epochs_ = epochs
        self.model_ = self.__build_model__(sequential)
        self.total_epochs_ = 0
        self.total_error_ = np.array([])

    """
    This function builds a sequential LSTM model
    based on the sequential parameter initialized in the __init__ function
    """

    def __build_model__(self, sequential):
        model_ = tf.keras.models.Sequential(sequential)
        model_.compile(optimizer="SGD", loss="mse")
        return model_

    """
    This function trains the model mentioned above
    It also stored the total epochs and the total errors in lists
    """

    def __train__(self, X, y):
        self.history_ = self.model_.fit(
            X, y, batch_size=self.batch_size_, epochs=self.epochs_, verbose=1
        )
        self.total_epochs_ += self.epochs_
        self.total_error_ = np.concatenate(
            (self.total_error_, self.history_.history["loss"])
        )

    """
    This function tests the accuracy of the trained model
    using the test set
    """

    def __test__(self, test_X):
        self.predictions_ = []
        self.predictions_.append(self.model_.predict(test_X))
        self.predictions_ = np.array(self.predictions_).reshape((len(test_X)))

    """
    This function plots the total loss graphs in total_epochs time
    """

    def __plot_total_loss__(self):
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(self.total_epochs_), self.total_error_, "ro")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

    """
    This function plots the predictions the model made with the testing set
    along witht the actual data
    """

    def __plot_prediction__(self, test_X, test_y, origin_data, label):
        loss = self.model_.evaluate(test_X, test_y)
        test_y = HELPER_FUNC.denormalize(test_y, origin_data)
        self.predictions_ = HELPER_FUNC.denormalize(self.predictions_, origin_data)

        plt.figure(figsize=(10, 8))
        plt.plot(test_y, color="blue", label="Actual")
        plt.plot(self.predictions_, color="red", label="prediction")
        plt.title("Apple Stock Price {}".format(label))
        plt.xlabel("Date")
        plt.ylabel("Stock Pricec")
        plt.legend()
        plt.figtext(
            0.27, 0.85, "Training Loss: {}".format(self.history_.history["loss"][-1])
        )
        plt.figtext(0.27, 0.82, "Test Loss: {}".format(loss))

        plt.show()
