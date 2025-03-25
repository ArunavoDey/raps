import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
#import SimpleNN
from .SimpleNN import SimpleFeedForwardNN, train_model
import math

def inverse_sqrt(x):
    return 1 / math.sqrt(1+x) #np.log(1 + x) #math.sqrt(1+x)


import numpy as np
def compute_job_score(predicted_power, job_duration, electricity_rate,
                      node_count, cores_per_node, total_nodes,
                      running_jobs, job_submission_time, current_time,
                      max_wait_time=24, weights=None):
    """
    Computes the multi-objective job score based on power cost, efficiency, utilization, and wait time.
    Parameters:
    - predicted_power (float): GNN-predicted power consumption for the job.
    - job_duration (float): Estimated job duration.
    - electricity_rate (float): Cost per kWh at the given HPC site and time.
    - node_count (int): Number of nodes requested by the job.
    - cores_per_node (int): Number of cores per node.
    - total_nodes (int): Total available nodes at the HPC site.
    - running_jobs (list of int): List of node counts for currently running jobs at the site.
    - job_submission_time (float): Time when the job was submitted.
    - current_time (float): Current time step.
    - max_wait_time (float): Maximum time before a job must be scheduled (default: 24 hours).
    - weights (dict): Dictionary of weights for each factor {wc, wp, wu, ww}.
    Returns:
    - job_score (float): Computed priority score for scheduling.
    """
    if weights is None:
        weights = {"wc": 1.0, "wp": 1.0, "wu": 1.0, "ww": 1.0}
    # Cost Factor
    C_jkt = 1 / (1 + predicted_power * job_duration * electricity_rate)
    # Power Efficiency Factor
    P_jk = 1 / (1 + predicted_power / (node_count * cores_per_node))
    # Utilization Factor
    U_kt = 1 - (sum(running_jobs) / total_nodes) if total_nodes > 0 else 0
    # Wait Time Factor
    W_jt = min((current_time - job_submission_time) / max_wait_time, 1)
    # Compute Final Score
    job_score = (weights["wc"] * C_jkt +
                 weights["wp"] * P_jk +
                 weights["wu"] * U_kt +
                 weights["ww"] * W_jt)
    return job_score



def scoring_function(df, feature_columns, time_series_column, time_series_stat='mean'):
    """
    Returns the indices that would sort the DataFrame by multiple features,
    including a time-series column, similar to numpy argsort.
    
    Args:
        df (pd.DataFrame): Input DataFrame with tabular data, including a time-series column.
        feature_columns (list of str): List of column names to sort by, in order of priority.
        time_series_column (str): Column name of the time-series feature.
        time_series_stat (str): Statistic to use for the time series ('mean', 'median', 'max', 'min').
        
    Returns:
        np.array: Indices that would sort the DataFrame by the specified columns.
    """
    # Define the data as a list of lists
    t_data = [
    [1, 2, 3, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
    [7, 8, 9, 10, 11, 12],
    [10, 11, 12, 13, 14, 15],
    [13, 14, 15, 16, 17, 18],
    [16, 17, 18, 19, 20, 21],
    [19, 20, 21, 22, 23, 24],
    [22, 23, 24, 25, 25, 27],
    [25, 26, 27, 28, 29, 30],
    [28, 29, 30, 31, 32, 33]
    ]

    # Create the DataFrame
    t_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6']
    t_df = pd.DataFrame(t_data, columns=t_columns)
    # Normalize the data for better learning
    t_df = (t_df - t_df.min()) / (t_df.max() - t_df.min())

    ##trainning
    input_dim = 6  # Number of features per sample
    hidden_dim = 6  # Increased hidden layer size for better learning

    model = SimpleFeedForwardNN(input_dim, hidden_dim)

    train_data = torch.tensor(t_df.to_numpy(), dtype=torch.float32)
    target_scores = torch.linspace(1, 0, steps=len(t_df))  # Assign highest to lowest scores correctly

    train_model(model, train_data, target_scores, epochs=500, lr=0.01)

        
    # Define a function to aggregate the time series based on the specified statistic
    def aggregate_time_series(series, time_series_stat):
        if time_series_stat == 'mean':
            return series.apply(np.mean) #list(np.mean(series, axis=1))  #series.apply(np.mean)
        elif time_series_stat == 'median':
            return series.apply(np.median)
        elif time_series_stat == 'max':
            return series.apply(np.max)
        elif time_series_stat == 'min':
            return series.apply(np.min)
        else:
            raise ValueError("Invalid time_series_stat. Choose from 'mean', 'median', 'max', or 'min'.")
    
    # Create a new DataFrame to store sorting keys
    sort_df = df.copy()
    print("printing sort df")
    print(sort_df)
    # Aggregate the time-series feature and add it as a new column
    sort_df['aggregated_time_series'] = aggregate_time_series(sort_df[time_series_column], 'mean')
    # Construct the sort keys by including the aggregated time-series and other feature columns
    sort_columns = ["num_nodes_req","aggregated_time_series"] #[col if col != time_series_column else 'aggregated_time_series' for col in feature_columns]
    #print("printing sort df")
    #print(sort_df)
    output_scores = []
    
    alpha = 0.0
    beta = 100.0
    gamma = 0.0
    delta = 0.00
    phi = 0.00
    
    for i in range(len(sort_df)):
        elm = sort_df[i:i+1]
        n_nodes = elm["num_nodes_req"]
        power_usage = elm["aggregated_time_series"]
        priority = elm["priority"]
        job_duration = elm["time_limit"]
        gpus = elm["num_gpus_req"]
        #print(f"num nodes {n_nodes}")
        #print(f"power usage {power_usage}")
        print("printing components of scoring function")
        print(f"n_nodes {n_nodes}")
        print(f"term {inverse_sqrt(n_nodes)}")
        print(f"after multiplication {(alpha)*inverse_sqrt(n_nodes)}")
        print(f"power_usage {power_usage}")
        print(f"term {inverse_sqrt(power_usage)}")
        print(f"after multiplication {(beta)*inverse_sqrt(power_usage)}")
        print(f"job_duration {job_duration}")
        print(f"term {inverse_sqrt(job_duration)}")
        print(f"after multiplication {(gamma)*inverse_sqrt(job_duration)}")
        print(f"priority {priority}")
        print(f"term {inverse_sqrt(priority)}")
        print(f"after multiplication {(delta)*inverse_sqrt(priority)}")


        #score = (alpha)*np.exp(-n_nodes)+ beta*np.exp(-power_usage)+gamma*np.exp(-job_duration)+delta*np.exp(priority)+phi*np.exp(-gpus)
        score = (alpha)*inverse_sqrt(n_nodes)+ beta*inverse_sqrt(power_usage)+gamma*inverse_sqrt(job_duration)+delta*inverse_sqrt(priority)+phi*inverse_sqrt(gpus)
        print(f"inside func_loop score {score}")
        if math.isinf(score):
            score = 10.0
        if math.isnan(score):
            score = -0.1
        score = np.float32(score)
        
        if math.isnan(score):
            score = np.float32(-1000.00)
        output_scores.append(score)
    """
    sort_df["r_priority"]= 1000 - sort_df["priority"]
    s_df = sort_df[["num_nodes","aggregated_time_series","cores_per_task","num_gpus_req","r_priority","time_limit"]]
    #Normalize the data for better learning
    s_df = (s_df - s_df.min()) / (s_df.max() - s_df.min())
    s_data = torch.tensor(s_df.to_numpy(), dtype=torch.float32)
    output_scores = model(s_data).squeeze().detach().numpy()
    """  
    # Sort by the specified columns and get the sorted indices
    # sorted_indices = sort_df.sort_values(by=sort_columns).index.to_numpy()
    #print("printing scores from scoring function")
    #print(scores)
    return output_scores #sorted_indices






class TabularTimeSeriesRankingModel(nn.Module):
    def __init__(self, num_features, time_series_length, lstm_hidden_size, dense_hidden_size):
        super(TabularTimeSeriesRankingModel, self).__init__()
        
        # LSTM layer for time series feature processing
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, batch_first=True)
        
        # Dense layers for non-time-series tabular data processing
        self.fc1 = nn.Linear(num_features, dense_hidden_size)
        self.fc2 = nn.Linear(dense_hidden_size, dense_hidden_size)
        
        # Final dense layer to output ranking score
        self.combined_fc = nn.Linear(lstm_hidden_size + dense_hidden_size, 1)  # Single output for ranking score

    def forward(self, x_tabular, x_time_series):
        # Process time-series feature with LSTM
        x_time_series = x_time_series.unsqueeze(-1)  # Add extra dimension for single feature input
        _, (h_n, _) = self.lstm(x_time_series)  # h_n is the LSTM hidden state
        
        # Process non-time-series tabular data with dense layers
        x_tabular = torch.relu(self.fc1(x_tabular))
        x_tabular = torch.relu(self.fc2(x_tabular))
        
        # Concatenate LSTM output with processed tabular features
        combined = torch.cat((h_n[-1], x_tabular), dim=1)
        
        # Final output layer for ranking score
        score = self.combined_fc(combined)
        
        return score  # Output a single ranking score per sample

# Hyperparameters and configuration
#num_features = 10  # Number of non-time-series features in tabular data
#time_series_length = 15  # Length of the time-series data
#lstm_hidden_size = 16
#dense_hidden_size = 32
#margin = 1.0  # Margin for the hinge loss

# Instantiate the model
#model = TabularTimeSeriesRankingModel(num_features, time_series_length, lstm_hidden_size, dense_hidden_size)

# Pairwise Ranking Loss Function based on feature columns
def feature_combination_pairwise_ranking_loss(model, pred, features, margin):
    """
    Pairwise ranking loss that ranks samples based on a combination of feature columns.
    Args:
        pred: Model predictions (B x 1) - batch of ranking scores
        features: Feature values (B x F) - batch of feature values used for ranking
    Returns:
        Loss value
    """
    loss = 0.0
    
    # Compute pairwise loss based on feature combination
    for i in range(pred.size(0)):
        for j in range(i + 1, pred.size(0)):
            # Compare features: (i, j) pair
            features_i, features_j = features[i], features[j]
            
            # Determine which sample should rank higher based on combined feature values
            if torch.all(features_i < features_j):  # i should rank higher than j
                loss += torch.relu(margin + pred[i] - pred[j])
            elif torch.all(features_i > features_j):  # j should rank higher than i
                loss += torch.relu(margin + pred[j] - pred[i])
                
    # Normalize by the number of pairs
    return loss / (pred.size(0) * (pred.size(0) - 1) / 2)

# Optimizer
def ranking_jobs_on_multi_feature_train(model, train_loader):
  """
  # Hyperparameters and configuration
  num_features = 12  # Number of non-time-series features in tabular data
  time_series_length = 10  # Length of the time-series data
  lstm_hidden_size = 16
  dense_hidden_size = 32
  margin = 1.0  # Margin for the hinge loss

  # Instantiate the model
  model = TabularTimeSeriesRankingModel(num_features, time_series_length, lstm_hidden_size, dense_hidden_size)
  """
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  # Training Loop
  for epoch in range(100):
    for x_tabular, x_time_series, target in train_loader:  # Assuming train_loader is defined
      optimizer.zero_grad()
        
      # Forward pass
      pred = model(x_tabular, x_time_series)  
      # Compute pairwise ranking loss based on feature columns
      loss = feature_combination_pairwise_ranking_loss(model, pred, x_tabular, margin) 
      # Backpropagation
      loss.backward()
      optimizer.step()
      print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")
  return model

def get_job_ranking(model, x_test_tabular, x_test_ts):
  # Hyperparameters and configuration
  num_features = 12  # Number of non-time-series features in tabular data
  time_series_length = 10  # Length of the time-series data
  lstm_hidden_size = 16
  dense_hidden_size = 32
  margin = 1.0  # Margin for the hinge loss
  # Instantiate the model
  #model = TabularTimeSeriesRankingModel(num_features, time_series_length, lstm_hidden_size, dense_hidden_size)
  pred = model(x_test_tabular, x_test_ts)
  return pred

