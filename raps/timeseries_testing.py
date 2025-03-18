import yaml
#import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Input, Dense, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import preprocessing
#import dataloader
#from preprocessing import preprocessor
#from dataloader import dataLoader
import sys
import os
import os.path
import csv
from .util import *
import re
#import TimeSeriesPredictor
from .TimeSeriesPredictor import TimeSeriesPredictor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

class timeseries_testing():
  def __init__(self, transfer_technique):
    self.transfer_technique = transfer_technique
  def __call__(self, X_test_numerical):
    """
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    """
    vers_dict = {}
    all_centroids = load_variables(f"/work2/08389/hcs77/ls6/application-fingerprinting/Models/preprocessed_data_centroids.pkl" )
    centroids = all_centroids["Centroids"]
    print(centroids)
    models = []
    for i in range(5):
        num_features = 12
        hidden_size = 16
        time_steps = 100
        model = TimeSeriesPredictor(num_features, hidden_size, time_steps)
        # Load the saved state dictionary into the model
        model.load_state_dict(torch.load(f"/work2/08389/hcs77/ls6/models/TimeSeries/TimeSeriesModel-{i}.pth" )) #"/work/08389/hcs77/ls6/application-fingerprinting/Models/rnn_model.pth"))
        model = model#.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        models.append(model)

    X_static_tensor = torch.tensor(X_test_numerical, dtype=torch.float32)
    #Y_series_tensor = torch.tensor(X_time_series_test, dtype=torch.float32)
    #print("loaded_df column names")
    #print(loaded_df.columns)
    print("X Tensor shape")
    print(X_static_tensor.shape)
    #print("Y Tensor Shape")
    #print(Y_series_tensor.shape)
    test_labels = find_nearest_centroids(X_static_tensor, torch.tensor(centroids, dtype=torch.float32))
    print(test_labels)
    predictions =[]
    for i in range(len(X_static_tensor)):
      selected_model = models[test_labels[i]]
      y_pred = selected_model(X_static_tensor[i])
      print("X_static tensor")
      print(X_static_tensor[i])
      print("Predictions")
      print(y_pred)
      predictions.append(y_pred)
      #os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}.csv"), exist_ok=True)
     
      
      #os.makedirs(os.path.dirname(os.getcwd()+global_config['time_series_model']+f"TimeSeriesModel-{rank}.json"), exist_ok=True)
    print(predictions)
    return predictions
