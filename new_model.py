import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error

#keras
from keras import layers, Input, Model
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, BatchNormalization, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.experimental.preprocessing import Normalization


#others
import os
import json
from pathlib import Path
import joblib
import folium
import geojson
import geopandas as gpd
from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBRegressor, plot_importance, plot_tree
from pandas.plotting import lag_plot
from keras.models import load_model
from utils import get_closing_df, grab_ticker

class PreProcessed_Model:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.n_input = 300
        self.n_feature = 1
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same', input_shape=(self.n_input, self.n_feature))) #(300, 1)
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=2))
        
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=self.n_feature)) #1
        self.model.compile(optimizer='adam', loss='mse')

        #stochastic gradient descent sgd, mse is best for outlier detection
        return self.model
    
    def preprocessed_data(self):
        
        df = get_closing_df(self.ticker)
        self.ticker_train = df[:int(len(df) * 0.7)]
        self.ticker_test = df[int(len(df) * 0.7):]
        self.scaler.fit(self.ticker_train)
        scaled_train_ticker = self.scaler.transform(self.ticker_train)
        scaled_test_ticker = self.scaler.transform(self.ticker_test)
        
        #creating the generator for training and testing
        train_generator_ticker = TimeseriesGenerator(scaled_train_ticker, scaled_train_ticker, length=self.n_input, batch_size=1)
        test_generator_ticker = TimeseriesGenerator(scaled_test_ticker, scaled_test_ticker, length=self.n_input, batch_size=1)
        self.model.fit(train_generator_ticker, epochs=5, batch_size=50, verbose=2)
        
        
        test_predictions = []
        train_predictions = []

        #how far into the future will I forecast?
        #loop through test window
        for x,_ in test_generator_ticker:
            
            # One timestep ahead of historical 300 points
            current_pred = self.model.predict(x)
            
            #store that prediction
            test_predictions.append(current_pred[0])

        #train window
        for x,_ in train_generator_ticker:
            
            # One timestep ahead of historical 300 points
            current_pred_train = self.model.predict(x)
            
            #store that prediction
            train_predictions.append(current_pred_train[0])


        #reverse the scaled into normal prices
        self.train_predictions = self.scaler.inverse_transform(train_predictions)
        self.true_predictions = self.scaler.inverse_transform(test_predictions)
    
    def get_results(self):
        self.test_result = self.ticker_test[300:]
        self.test_result['predictions'] = self.true_predictions
        return self.test_result
        
    def plot_test_result_graph(self):
        self.test_result[['Close', 'predictions']].plot(figsize=(10,8))
        sns.scatterplot(x=self.test_result.index, y=self.test_result.Close).set(title='{} Result Predictions'.format(self.ticker))
        sns.scatterplot(x=self.test_result.index, y=self.test_result.predictions)
        
    def plot_train_result_graph(self):
        self.train_result = self.ticker_train[300:]
        self.train_result['predictions'] = self.train_predictions
        self.train_result.plot(figsize=(12,8))
        plt.title('{} Training Predictions'.format(self.ticker))
    
    def predict(self, target):
        df = get_closing_df(target)
        self.scaler.fit(df)
        scaled_test_df = self.scaler.transform(df)
        n_input = 300 #window size of 300
        n_feature = 1 #univariate


        #creating the generator for training and testing
        test_generator_df = TimeseriesGenerator(scaled_test_df, scaled_test_df, length=n_input, batch_size=1)

        test_predictions = []

        #how far into the future will I forecast?
        #loop through test window
        for x,_ in test_generator_df:
            
            # One timestep ahead of historical 300 points
            current_pred = self.model.predict(x)
            
            #store that prediction
            test_predictions.append(current_pred[0])



        #reverse the scaled into normal prices
        self.test_predictions = self.scaler.inverse_transform(test_predictions)
        
        test_result = df[300:]
        test_result['predictions'] = self.test_predictions
        print(test_result)
        
        mse = mean_squared_error(y_true=test_result['Close'], y_pred=test_result['predictions'])
        
        test_result[['Close', 'predictions']].plot(figsize=(10,8))
        sns.scatterplot(x=test_result.index, y=test_result.Close).set(title='{} Results. MSE is {}'.format(target, mse))
        sns.scatterplot(x=test_result.index, y=test_result.predictions)

        print('{} MSE is: {}'.format(target, mse))
        
        plt.savefig(os.path.join(os.getcwd(), 'results', '{}_results.png'.format(target)))