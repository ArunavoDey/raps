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
import sys
import os
import os.path
import csv
#import util
import re
#import Autoencoder
from .Autoencoder import ImprovedAutoencoder
import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
#import rnn_predictor
from .rnn_predictor import rnn_predictor
import torch.nn as nn
import torch.optim as optim
#import intel_extension_for_pytorch as ipex
#import torchsummary
#from torchsummary import summary
class rnn_testing():
  def __init__(self, transfer_technique="Default"):
    self.transfer_technique = transfer_technique
  def __call__(self, X_test_numerical, X_time_series_test):
    """
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    use_case_specific_config = global_config["rnn_training"]
    """
    ##load the encoded versions of data
    autoencoder_model = ImprovedAutoencoder(input_dim=12, encoding_dim=64, attention_dim=32)
    autoencoder_model.encoder.load_state_dict(torch.load("/work2/08389/hcs77/ls6/application-fingerprinting/Models/encoder_model.pth"))
    autoencoder_model.decoder.load_state_dict(torch.load("/work2/08389/hcs77/ls6/application-fingerprinting/Models/decoder_model.pth"))

    # Encode the numerical features
    X_test_numerical_encoded = autoencoder_model.encoder(torch.tensor(X_test_numerical, dtype=torch.float32)).detach().numpy()

    # Verify shapes after encoding
    print(f"Shape of X_test_numerical_encoded after encoding: {X_test_numerical_encoded.shape}")
    print(f"Shape of X_time_series_test: {X_time_series_test.shape}")
    #print(f"Shape of y_test_sequences: {y_test_sequences.shape}")
      
    #Ensure that the first dimension (number of samples) matches
    #assert X_test_numerical_encoded.shape[0] == X_time_series_test.shape[0] == y_test_sequences.shape[0], "Mismatch in the number of testing samples"
     

    # Train RNN
    hidden_dim = 64
    num_layers = 2  # Number of RNN layers
    output_dim = 10  # Number of future time steps to predict 

    loaded_rnn_model = rnn_predictor(input_dim=1, input_dim2=32, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    
    # Load the saved state dictionary into the model
    loaded_rnn_model.load_state_dict(torch.load("/work2/08389/hcs77/ls6/application-fingerprinting/Models/rnn_model.pth",map_location=torch.device('cpu') ))
    loaded_rnn_model = loaded_rnn_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    loaded_rnn_model.eval()
    print("Model loaded successfully.")
    epch = 0
    with torch.no_grad():
        #X_test_numerical_encoded = autoencoder_model.encoder(torch.tensor(X_test_numerical, dtype=torch.float32)).detach().numpy()
        #X_test_time_series = np.array(X_test_numerical[:, :, np.newaxis])
        y_test_pred = []
        for i in range(len(X_time_series_test)):
            #print(f"X_num {X_num}")
            #print(f"X_ts {X_ts}")
            X_ts = torch.tensor(X_time_series_test[i:i+1],dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            X_num = torch.tensor(X_test_numerical_encoded[i:i+1],dtype=torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            print(f"X ts {X_ts}")
            print(f"X num {X_num}")
            y_pred = loaded_rnn_model(X_num, X_ts).cpu().numpy()  # Pass both X_num and X_ts
            #print(f"X_num {X_num}")
            #print(f"X_ts {X_ts}")
            print("Predictions")
            print(y_pred)
            print("Test")
            #print(y_test[i])
            print("Test Sequences")
            #print(y_test_sequences[i])
            y_test_pred.append(y_pred)
            epch = epch + 1
        #y_test_pred = np.vstack(y_test_pred)
    Y_t = []
    for i in range(len(y_test_pred)):
        y_test_pred[i]= y_test_pred[i].ravel()
        Y_t.append(y_test_pred[i].tolist())
    print("printing y_test_pred")
    print(y_test_pred)
    y_pred_mean = np.mean(y_test_pred, axis=1)
    print("printing y_pred_mean")
    print(y_pred_mean)
    #print(y_test_pred.shape)
    #print(y_pred_mean.shape)
    jobrank_array = np.argsort(y_pred_mean)
    high_priority = len(y_pred_mean)
    priority_array = []
    for i in range(high_priority):
        priority_array.append(0)
    print("printing jobrank array")
    print(jobrank_array)
    for i in range(high_priority):
        priority_array[int(jobrank_array[i])] = int(i)

    #jobs_ranked_in_priority = np.argsort(y_pred_mean)
    return priority_array, y_test_pred, Y_t
    """ 
    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epch + 1) , (y_test_pred-y_test), 'bo', label='actual')
    #plt.ylim(-500,500)
    #plt.plot(range(1, epch + 1) , y_test, 'b')
    plt.plot(range(1, epch + 1) , y_test_pred, 'r', alpha=0.1)
    plt.plot(range(1, epch + 1) , y_test, 'b', alpha=0.05)
    plt.title('Actual and Prediction')
    plt.xlabel('Sample Number')
    plt.ylabel('Actual and Predictions')
    plt.legend()
    plt.savefig(global_config["ActndPred"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/ActndPred.pdf")
    

    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epch + 1) , (y_test_pred-y_test), 'bo', label='actual')
    plt.ylim(top=35000)
    plt.plot(range(1, epch + 1) , y_test, 'b')
    #plt.plot(range(1, epch + 1) , y_test_pred, 'r', label='prediction')
    plt.title('Actual')
    plt.xlabel('Sample Number')
    plt.ylabel('Actual Values')
    plt.legend()
    plt.savefig(global_config["ActualValues"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/ActualValues.pdf")



    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epch + 1) , (y_test_pred-y_test), 'bo', label='actual')
    plt.ylim(top = 35000)
    #plt.plot(range(1, epch + 1) , y_test, 'b')
    plt.plot(range(1, epch + 1) , y_test_pred, 'r')
    plt.title('Predictions')
    plt.xlabel('Sample Number')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(global_config["PredictedValues"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/PredictedValues.pdf")
    
    
    newDF = pd.DataFrame()
    newDF["Epcs"]= range(1,epch+1)
    newDF["Actual"]= list(y_test)
    newDF["Prediction"]=list(y_test_pred)
    yt = list(y_test)
    yp = list(y_test_pred)
    newDF["Difference"]= [yp[i]-yt[i] for i in range(len(yt))] #list(y_test_pred)-list(y_test)
    yd = [yp[i]-yt[i] for i in range(len(yt))]
    std_nums =[]
    stds = []
    for std_num in range(40):
        std_nums.append(std_num+1)
        stds.append(0)
    mn_num = np.mean(yt)
    for i in range(len(yd)):
        st_num = 5 #np.std(yt[i])
        print(f"st_num for {i} is {st_num}")
            stt_num = int(np.abs(yd[i][j])/np.abs(st_num))
            if stt_num <= 39:
                stds[stt_num] = stds[stt_num]+1

   
    newDF.to_csv(os.getcwd()+global_config["result_dataframe"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/NewDF.csv")
    util.visualize_box_plot(newDF, os.getcwd()+global_config["box_plot"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/BoxPlot.pdf")
    util.visualize_bar_plot(newDF, os.getcwd()+global_config["bar_plot"]) #"/work/08389/hcs77/ls6/application-fingerprinting/fig/BarPlot.pdf")
    util.visualize_bar_plot2(std_nums,stds,os.getcwd()+global_config["bar_plot2"])#"/work/08389/hcs77/ls6/application-fingerprinting/fig/BarPlotStd.pdf")
    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epch + 1) , (y_test_pred-y_test), 'bo', label='actual')
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.plot(y_test, y_test_pred, 'bo', label='prediction')
    plt.title('Actual vs Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig(os.getcwd()+global_config["ActvsPred"])#"/work/08389/hcs77/ls6/application-fingerprinting/fig/ActvsPred.pdf")
    """

    





