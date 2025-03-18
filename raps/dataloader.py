import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
import time
from .util import *
#import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import random
from scipy.fft import fft, ifft, fftfreq
import copy
import scipy
from sklearn.model_selection import train_test_split
class dataLoader():
  def __init__(self, config, dataset_name):
    #self.src_path = src_path
    self.config = config
    self.dataset_name = dataset_name
  def load_data(self):
    if self.dataset_name =="pm100":
        base_path = os.getcwd()+self.config.get('base_paths', {}).get('data')+self.config.get('datasets',{}).get('pm100',{}).get('filename')
        csv_path = os.getcwd()+self.config.get('datasets',{}).get('pm100',{}).get('selected_data')
        indices_path = os.getcwd()+self.config.get('datasets',{}).get('pm100',{}).get('selected_indices')
        df = pd.read_parquet(base_path)
        n_df, _ = split_dataframe(df, [0, 231200], sample_size=10000, csv_filename=csv_path, txt_filename=indices_path)
    return n_df
  def load_saved_vars(self):
      vars = load_variables(os.getcwd()+self.config.get('datasets',{}).get(self.dataset_name, {}).get('variable_path'))
      return vars
  """
  def loadData(self):
    i = 0
    for filename in os.listdir(self.src_path):
      df = pd.read_csv(self.src_path+filename, low_memory=False)
      df = df.dropna()
      df.reset_index(drop=True, inplace=True)
      ff = filename.split(".")
      df["Domain"] = ff[0]
      if i==0:
        self.src_data = df
      else:
        self.src_data = pd.concat([self.src_data, df], axis=0)
      i = i+1
    self.src_data.reset_index(drop=True, inplace=True)
    j = 0
    for filename in os.listdir(self.tar_path):
      df = pd.read_csv(self.tar_path+filename, low_memory=False)
      df = df.dropna()
      df.reset_index(drop=True, inplace=True)
      #df = df[0:10]
      ff = filename.split(".")
      df["Domain"] = ff[0]
      #print(df)
      if j==0:
        self.tar_data = df
      else:
        self.tar_data = pd.concat([self.tar_data, df], axis=0) 
      j = j + 1
    self.tar_data.reset_index(drop=True, inplace=True)
  def getData(self):
    return self.src_data, self.tar_data
  """
  def getXY(self, colName, rowName, targetColumn):
    #if datasetName == "perfvar":
    #  application = rowName
    #  self.src_data = self.src_data.loc[self.src_data['application']!= application]
    #  self.tar_data = self.tar_data.loc[self.tar_data['application']== application]
    s_arr = self.src_data["Domain"].unique()
    t_arr = self.tar_data["Domain"].unique()
    for name in t_arr:
      dat = []
      for i in range(len(self.src_data.columns)):
        dat.append(0)
      dat[-1]=name
      self.src_data.loc[len(self.src_data.index)] = dat
    for name in s_arr:
      dat = []
      for i in range(len(self.tar_data.columns)):
        dat.append(0)
      dat[-1]=name
      self.tar_data.loc[len(self.tar_data.index)] = dat

    ##one hot encoding
    self.src_data = pd.get_dummies(self.src_data, columns=["Domain"], drop_first=False)
    self.tar_data = pd.get_dummies(self.tar_data, columns=["Domain"], drop_first=False)
    ##remove extra inserted rows
    for name in t_arr:
        self.src_data = self.src_data.drop([len(self.src_data.index)-1])
    for name in s_arr:
        self.tar_data = self.tar_data.drop([len(self.tar_data.index)-1])
    ##listing other string valued columns in source
    stringColumns = []
    for i in range(len(self.src_data.columns)):
      if self.src_data[self.src_data.columns[i]].dtypes == "object":
        stringColumns.append(self.src_data.columns[i])
    ##apply one hot encoding to other string columns on source
    print(f"src string columns {stringColumns}")
    print(self.src_data)
    self.src_data = pd.get_dummies(self.src_data, columns = stringColumns, drop_first=False)
    #Columns to be omitted
    omitColumns = []
    omitColumns.append(targetColumn)
    #self.src_ty = self.src_data[[targetColumn]]
    self.src_ty = self.src_data[targetColumn]
    self.src_y = self.src_data.loc[:, targetColumn].values
    #inds = [i for i,f in enumerate(self.src_data.columns) if f not in omitColumns]
    #print(inds)
    #self.src_data = self.src_data.drop([targetColumn], axis=1)
    self.src_data = self.src_data.drop(targetColumn, axis=1)
    self.src_tx = self.src_data
    self.src_x = self.src_data.values

    ##listing other string valued columns in target
    stringColumns = []
    for i in range(len(self.tar_data.columns)):
      if self.tar_data[self.tar_data.columns[i]].dtypes == "object":
        stringColumns.append(self.tar_data.columns[i])
    ##apply one hot encoding to other string columns on source
    self.tar_data = pd.get_dummies(self.tar_data, columns = stringColumns, drop_first=False)
    #Columns to be omitted
    omitColumns = [ ]
    omitColumns.append(targetColumn)
    #self.tar_ty = self.tar_data[[targetColumn]]
    self.tar_ty = self.tar_data[targetColumn]
    self.tar_y = self.tar_data.loc[:, targetColumn].values
    #inds = [i for i,f in enumerate(self.tar_data.columns) if f not in omitColumns]
    #print(inds)
    #self.tar_data = self.tar_data.drop([targetColumn], axis =1)
    self.tar_data = self.tar_data.drop(targetColumn, axis =1)
    self.tar_tx = self.tar_data
    self.tar_x = self.tar_data.values

    #print(omitColumns)
    print("Src Columns########################")
    for name in list(self.src_data.columns):
        print(name)
    print("Tar Columns ****************************")
    for name in list(self.tar_data.columns):
        print(name)
    #print("src_x shape {self.src_x.shape
    print(self.src_x.shape)
    print(self.tar_x.shape)
    return self.src_x, self.src_y, self.tar_x, self.tar_y
  def getSrcXY(self):
    return self.src_tx, self.src_ty
  def getTarXY(self):
    return self.tar_tx, self.tar_ty
