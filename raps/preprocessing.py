import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
import time
#import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import random
from scipy.fft import fft, ifft, fftfreq
import copy
import scipy
from sklearn.model_selection import train_test_split

class preprocessor():
  def __init__(self, df):
    self.df = pd.DataFrame(df)
    print("printing dataframe from preprocessor")
    print(self.df)


  def pad_sequences(self, data, target_len, dtype=np.float32):
    padded_data = np.zeros((len(data), target_len), dtype=dtype)
    for i, seq in enumerate(data):
      if len(seq) < target_len:
        mean_val = np.mean(seq)
        padded_data[i, :len(seq)] = seq
        padded_data[i, len(seq):] = mean_val
      else:
        padded_data[i] = seq[:target_len]
    return padded_data
  # Process nodes (list of node IDs)
  def process_nodes(self, nodes_list):
    if not isinstance(nodes_list, list) or len(nodes_list) == 0:
      return 0, 0, 0
    nodes = [int(n) for n in nodes_list if str(n).isdigit()]
    return len(nodes), min(nodes), max(nodes)
  def process_cores_nodes_alloc(self, df, features):
    # Process cores_alloc_layout (dict of node: core_list)
    df['node_count'] = df['nodes'].apply(lambda x: self.process_nodes(x)[0])
    # Process req_nodes (mostly empty)
    # Cleanup original columns
    features.append('node_count')
    return df, features


  def preprocess_data_disjoint_v1(self, features, target, input_seq_len, output_seq_len):
    """
    Preprocess the data: fill NaN values, drop columns with all zeros, pad or trim sequences.

    :param df: pd.DataFrame, the input dataframe
    :param features: list of str, the feature columns to use
    :param target: str, the target column to predict
    :param input_seq_len: int, the number of time steps in the input sequence
    :param output_seq_len: int, the number of future values to predict
    :return: tuple of processed train and test dataframes, scaler, X_train, y_train, X_test, y_test
    """
    # Fill NaN values with the mean of each column
    for feature in features:
      print(f"the feature is {feature}")
      mean_val = self.df[feature].mean()
      self.df[feature].fillna(mean_val, inplace=True)

    # Replace NaN values in the target column with zeros before padding
    self.df[target] = self.df[target].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # Pad or trim the target sequences
    max_len = self.df[target].apply(len).max()
    print(max_len)
    
    def pad_or_trim(x):
      if len(x) < max_len:
        mean_val = np.nanmean(x)
        x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=mean_val if not np.isnan(mean_val) else 0)
      else:
        x = x[:max_len]
      return x

    self.df[target] = self.df[target].apply(lambda x: pad_or_trim(x))

    # Min-max scale the feature columns
    scaler = MinMaxScaler()
    self.df[features] = scaler.fit_transform(self.df[features])

    # Split the data into training and testing sets
    train_size = int(0.8 * len(self.df))
    df_train = self.df.iloc[:train_size]
    df_test = self.df.iloc[train_size:]

    def create_disjoint_sequences(df, target, input_seq_len, output_seq_len):
      X, y = [], []
      for row in df.itertuples():
        time_series = getattr(row, target)
        for i in range(0, len(time_series) - input_seq_len - output_seq_len + 1, input_seq_len):
          X.append(time_series[i:i+input_seq_len])
          y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
      return np.array(X), np.array(y)

    X_train, y_train = create_disjoint_sequences(df_train, target, input_seq_len, output_seq_len)
    X_test, y_test = create_disjoint_sequences(df_test, target, input_seq_len, output_seq_len)

    # Ensure the numerical features are in the same shape
    X_train_numerical = np.repeat(df_train[features].values, input_seq_len, axis=0)
    X_test_numerical = np.repeat(df_test[features].values, input_seq_len, axis=0)

    return df_train, df_test, scaler, X_train_numerical, y_train, X_test_numerical, y_test

  def preprocess_data_disjoint2(self, features, target, input_seq_len, output_seq_len):
    """
    Preprocess the data: fill NaN values, drop columns with all zeros, pad or trim sequences.

    :param df: pd.DataFrame, the input dataframe
    :param features: list of str, the feature columns to use
    :param target: str, the target column to predict
    :param input_seq_len: int, the number of time steps in the input sequence
    :param output_seq_len: int, the number of future values to predict
    :return: tuple of processed train and test dataframes, scaler, X_train_numerical, y_train, X_test_numerical, y_test
    """
    # Fill NaN values with the mean of each column
    for feature in features:
      mean_val = self.df[feature].mean()
      self.df[feature].fillna(mean_val, inplace=True)

    # Replace NaN values in the target column with zeros before padding
    self.df[target] = self.df[target].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # Pad or trim the target sequences
    max_len = self.df[target].apply(len).max()

    def pad_or_trim(x):
      if len(x) < max_len:
        mean_val = np.nanmean(x)
        x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=mean_val if not np.isnan(mean_val) else 0)
      else:
        x = x[:max_len]
      return x

    self.df[target] = self.df[target].apply(lambda x: pad_or_trim(x))

    # Min-max scale the feature columns
    scaler = MinMaxScaler()
    self.df[features] = scaler.fit_transform(self.df[features])

    # Split the data into training and testing sets
    train_size = int(0.8 * len(self.df))
    df_train = self.df.sample(frac=0.8, random_state=42)
    df_test = self.df.drop(df_train.index)

    def create_disjoint_sequences(df, target, input_seq_len, output_seq_len):
      X, y = [], []
      for row in df.itertuples():
        time_series = getattr(row, target)
        for i in range(0, len(time_series) - input_seq_len - output_seq_len + 1, input_seq_len + output_seq_len):
          X.append(time_series[i:i+input_seq_len])
          y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
      X = pad_sequences(X, input_seq_len)
      y = pad_sequences(y, output_seq_len)
      return np.array(X), np.array(y)

    X_train, y_train = create_disjoint_sequences(df_train, target, input_seq_len, output_seq_len)
    X_test, y_test = create_disjoint_sequences(df_test, target, input_seq_len, output_seq_len)

    def create_disjoint_numerical_sequences(df, features, input_seq_len):
      X = []
      for row in df.itertuples():
        for i in range(0, len(features) - input_seq_len + 1, input_seq_len):
          X.append([getattr(row, feature) for feature in features])
      return np.array(X)

    X_train_numerical = create_disjoint_numerical_sequences(df_train, features, input_seq_len)
    X_test_numerical = create_disjoint_numerical_sequences(df_test, features, input_seq_len)

    return df_train, df_test, scaler, X_train_numerical, y_train, X_test_numerical, y_test

  def preprocess_data_overlapping(self, features, target, input_seq_len, output_seq_len):
    """
    Preprocess the data: fill NaN values, drop columns with all zeros, pad or trim sequences.

    :param df: pd.DataFrame, the input dataframe
    :param features: list of str, the feature columns to use
    :param target: str, the target column to predict
    :param input_seq_len: int, the number of time steps in the input sequence
    :param output_seq_len: int, the number of future values to predict
    :return: tuple of processed train and test dataframes, scaler, X_train, y_train, X_test, y_test
    """
    # Fill NaN values with the mean of each column
    for feature in features:
        mean_val = self.df[feature].mean()
        self.df[feature].fillna(mean_val, inplace=True)

    # Replace NaN values in the target column with zeros before padding
    self.df[target] = self.df[target].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # Pad or trim the target sequences
    max_len = self.df[target].apply(len).max()

    def pad_or_trim(x):
        if len(x) < max_len:
            mean_val = np.nanmean(x)
            x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=mean_val if not np.isnan(mean_val) else 0)
        else:
            x = x[:max_len]
        return x

    self.df[target] = self.df[target].apply(lambda x: pad_or_trim(x))

    # Min-max scale the feature columns
    scaler = MinMaxScaler()
    self.df[features] = scaler.fit_transform(self.df[features])

    # Split the data into training and testing sets
    train_size = int(0.8 * len(self.df))
    df_train = self.df.iloc[:train_size]
    df_test = self.df.iloc[train_size:]

    def create_sequences(df, target, input_seq_len, output_seq_len):
        X, y = [], []
        for row in df.itertuples():
            time_series = getattr(row, target)
            for i in range(len(time_series) - input_seq_len - output_seq_len + 1):
                X.append(time_series[i:i+input_seq_len])
                y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
        X = pad_sequences(X, input_seq_len)
        y = pad_sequences(y, output_seq_len)
        return X, y

    X_train, y_train = create_sequences(df_train, target, input_seq_len, output_seq_len)
    X_test, y_test = create_sequences(df_test, target, input_seq_len, output_seq_len)

    return df_train, df_test, scaler, X_train, y_train, X_test, y_test

  def preprocess_data_disjoint_testing(self, features, target, input_seq_len, output_seq_len):
    # Fill NaN values with the mean of each column
    ###jaya's functions
    self.df, features = self.process_cores_nodes_alloc(self.df, features)

    for feature in features:
      print(f"the feature is {feature} {self.df[feature]}")
      mean_val = np.nanmean(self.df[feature])
      self.df[feature].fillna(mean_val, inplace=True)

    # Replace NaN values in the target column with zeros before padding
    #self.df[target] = self.df[target].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # Pad or trim the target sequences
    max_len = self.df[target].apply(len).max()
    print(f"Max length of target sequences: {max_len}")
    print("df_Inside_Preprocessing")
    print(self.df[features])
    def pad_or_trim(x):
      if len(x) < max_len:
        mean_val = np.nanmean(x)
        x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=mean_val if not np.isnan(mean_val) else 0)
      else:
        x = x[:max_len]
      return x
    def create_disjoint_sequences2(df, target, input_seq_len, output_seq_len):
      X, y = [], []
      for row in df.itertuples():
        time_series = getattr(row, target)
        if len(time_series) >= input_seq_len + output_seq_len:
          num_samples = (len(time_series) - input_seq_len - output_seq_len + 1)
          for i in range(0, num_samples, input_seq_len + output_seq_len):
            X.append(time_series[i:i+input_seq_len])
            y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
        else:
          # Pad sequences if they are shorter than required
          padded_ts = pad_sequences([time_series], input_seq_len + output_seq_len)[0]
          X.append(padded_ts[:input_seq_len])
          y.append(padded_ts[input_seq_len:input_seq_len+output_seq_len])
      print(f"create_disjoint_sequences >> X.shape {np.array(X).shape} Y.shape {np.array(y).shape}")
      return np.array(X)[:, :, np.newaxis], np.array(y)  # Add a new axis for features


    #self.df[target] = self.df[target].apply(lambda x: pad_or_trim(x))

    # Min-max scale the feature columns
    scaler = MinMaxScaler()
    self.df[features] = scaler.fit_transform(self.df[features])
    print("After scaling the features")
    print(self.df[features])
    self.df.dropna(axis=1, inplace=True)
    print("After dropping nan values")
    print(self.df)
    # Randomly split the data into training and testing sets
    #df_train = self.df.sample(frac=0.8, random_state=42)
    df_test = self.df#.drop(df_train.index)

    #X_train, y_train = create_disjoint_sequences2(df_train, target, input_seq_len, output_seq_len)
    #X_test, y_test = create_disjoint_sequences2(df_test, target, input_seq_len, output_seq_len)
    #X_train = df_train[features]
    X_test = df_test[features]
    # Ensure the numerical features are in the same shape
    """
    print(f"df_train[features] {df_train[features]}")
    print(f"X_train.shape[0] {X_train.shape[0]}")
    print(f"df_train.shape[0] {df_train.shape[0]}")
    X_train_numerical = np.repeat(df_train[features].values, X_train.shape[0] // df_train.shape[0], axis=0)
    print(f"X_train_numerical {X_train_numerical}")
    """
    X_test_numerical = df_test[features].values #np.repeat(df_test[features].values, X_test.shape[0] // df_test.shape[0], axis=0)
    
    return df_test, scaler, X_test_numerical


  def preprocess_data_disjoint3(self, features, target, input_seq_len, output_seq_len):
    # Fill NaN values with the mean of each column
    for feature in features:
      print(f"the feature is {feature} {self.df[feature]}")
      mean_val = self.df[feature].mean()
      self.df[feature].fillna(mean_val, inplace=True)

    # Replace NaN values in the target column with zeros before padding
    self.df[target] = self.df[target].apply(lambda x: np.nan_to_num(x, nan=0.0))

    # Pad or trim the target sequences
    max_len = self.df[target].apply(len).max()
    print(f"Max length of target sequences: {max_len}")

    def pad_or_trim(x):
      if len(x) < max_len:
        mean_val = np.nanmean(x)
        x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=mean_val if not np.isnan(mean_val) else 0)
      else:
        x = x[:max_len]
      return x

    def create_disjoint_sequences2(df, target, input_seq_len, output_seq_len):
      X, y = [], []
      for row in df.itertuples():
        time_series = getattr(row, target)
        if len(time_series) >= input_seq_len + output_seq_len:
          num_samples = (len(time_series) - input_seq_len - output_seq_len + 1)
          for i in range(0, num_samples, input_seq_len + output_seq_len):
            X.append(time_series[i:i+input_seq_len])
            y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
        else:
          # Pad sequences if they are shorter than required
          padded_ts = self.pad_sequences([time_series], input_seq_len + output_seq_len)[0]
          X.append(padded_ts[:input_seq_len])
          y.append(padded_ts[input_seq_len:input_seq_len+output_seq_len])
      print(f"create_disjoint_sequences >> X.shape {np.array(X).shape} Y.shape {np.array(y).shape}")
      return np.array(X)[:, :, np.newaxis], np.array(y)  # Add a new axis for features


    self.df[target] = self.df[target].apply(lambda x: pad_or_trim(x))

    # Min-max scale the feature columns
    scaler = MinMaxScaler()
    self.df[features] = scaler.fit_transform(self.df[features])

    # Randomly split the data into training and testing sets
    df_train = self.df.sample(frac=0.8, random_state=42)
    df_test = self.df.drop(df_train.index)

    X_train, y_train = create_disjoint_sequences2(df_train, target, input_seq_len, output_seq_len)
    X_test, y_test = create_disjoint_sequences2(df_test, target, input_seq_len, output_seq_len)

    # Ensure the numerical features are in the same shape
    print(f"df_train[features] {df_train[features]}")
    print(f"X_train.shape[0] {X_train.shape[0]}")
    print(f"df_train.shape[0] {df_train.shape[0]}")
    X_train_numerical = np.repeat(df_train[features].values, X_train.shape[0] // df_train.shape[0], axis=0)
    print(f"X_train_numerical {X_train_numerical}")
    X_test_numerical = np.repeat(df_test[features].values, X_test.shape[0] // df_test.shape[0], axis=0)

    return df_train, df_test, scaler, X_train_numerical, y_train, X_test_numerical, y_test

  def create_disjoint_sequences3(self, df, target, input_seq_len, output_seq_len):
    X, y = [], []
    print("inside create disjoint sequences3")
    print(df)
    itr =0 
    for row in df.itertuples():
      time_series = getattr(row, target)
      if len(time_series) >= input_seq_len + output_seq_len:
        num_samples = (len(time_series) - input_seq_len - output_seq_len + 1)
        for i in range(0, num_samples, input_seq_len + output_seq_len):
          X.append(time_series[i:i+input_seq_len])
          y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
      else:
        padded_ts = self.pad_sequences([time_series], input_seq_len + output_seq_len)[0]
        X.append(padded_ts[:input_seq_len])
        y.append(padded_ts[input_seq_len:input_seq_len+output_seq_len])
      itr = itr + 1
    print(f"itr {itr}")
    print(f"create_disjoint_sequences >> X.shape {np.array(X).shape} Y.shape {np.array(y).shape}")
    return np.array(X)[:, :, np.newaxis], np.array(y)  # Add a new axis for features


  def create_disjoint_sequences_testing(self, df, target, input_seq_len, output_seq_len):
    X, y = [], []
    print("inside create disjoint sequences3")
    print(df)
    itr =0
    for row in df.itertuples():
      time_series = getattr(row, target)
      """
      if len(time_series) >= input_seq_len + output_seq_len:
        num_samples = (len(time_series) - input_seq_len - output_seq_len + 1)
        for i in range(0, num_samples, input_seq_len + output_seq_len):
          X.append(time_series[i:i+input_seq_len])
          y.append(time_series[i+input_seq_len:i+input_seq_len+output_seq_len])
      else:
      """
      padded_ts = self.pad_sequences([time_series], input_seq_len + output_seq_len)[0]
      X.append(padded_ts[:input_seq_len])
      y.append(padded_ts[input_seq_len:input_seq_len+output_seq_len])
      itr = itr + 1
    print(f"itr {itr}")
    print(f"create_disjoint_sequences >> X.shape {np.array(X).shape} Y.shape {np.array(y).shape}")
    return np.array(X)[:, :, np.newaxis], np.array(y)  # Add a new axis for features



  
