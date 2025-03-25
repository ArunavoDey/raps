"""
    # Reference
    Antici, Francesco, et al. "PM100: A Job Power Consumption Dataset of a
    Large-scale Production HPC System." Proceedings of the SC'23 Workshops
    of The International Conference on High Performance Computing,
    Network, Storage, and Analysis. 2023.

    # get the data
    Download `job_table.parquet` from https://zenodo.org/records/10127767

    # to simulate the dataset
    python main.py -f /path/to/job_table.parquet --system marconi100

    # to replay using differnt schedulers
    python main.py -f /path/to/job_table.parquet --system marconi100 --policy fcfs --backfill easy
    python main.py -f /path/to/job_table.parquet --system marconi100 --policy priority --backfill firstfit

    # to fast-forward 60 days and replay for 1 day
    python main.py -f /path/to/job_table.parquet --system marconi100 -ff 60d -t 1d

    # to analyze dataset
    python -m raps.telemetry -f /path/to/job_table.parquet --system marconi100 -v

"""
import uuid
import random
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
from ..job import job_dict
from ..utils import power_to_utilization, next_arrival
from ..preprocessing import preprocessor
from ..rnn_testing import rnn_testing
from ..timeseries_testing import timeseries_testing
from ..timeseries_test import timeseries_test
from ..TimeSeriesPredictor import TimeSeriesPredictor
from ..ranking import TabularTimeSeriesRankingModel, ranking_jobs_on_multi_feature_train, scoring_function
"""
load_config_variables([
    'CPUS_PER_NODE',
    'GPUS_PER_NODE',
    'NICS_PER_NODE',
    'TRACE_QUANTA',
    'POWER_GPU_IDLE',
    'POWER_GPU_MAX',
    'POWER_CPU_IDLE',
    'POWER_CPU_MAX',
    'POWER_NIC',
    'POWER_NVME', 
    'UI_UPDATE_FREQ'
], globals())
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from ..SimpleNN import SimpleFeedForwardNN, train_model



def load_data(jobs_path, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Parameters
    ----------
    jobs_path : str
        The path to the jobs parquet file.
    Returns
    -------
    list
        The list of parsed jobs.
    """
    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')
    training_data = pd.read_csv("/work2/08389/hcs77/ls6/application-fingerprinting/fig/selected_data.csv" )
    jobs_df =  jobs_df.drop(training_data.index, errors='ignore') #jobs_df[0:500]
    #jobs_df = jobs_df[0:5000]
    #jobs_df = jobs_df.dropna()
    #mean_val = np.nanmean(jobs_df[:])
    #jobs_df = jobs_df.fillna(jobs_df.mean(), inplace=True)
    # Replace NaN values in numerical columns only
    #jobs_df[jobs_df.select_dtypes(include=['number']).columns] = jobs_df.select_dtypes(include=['number']).fillna(jobs_df.mean())
    
    # Identify numeric columns
    numeric_cols = jobs_df.select_dtypes(include=['number']).columns
    print(f"printing numeric cols {numeric_cols}")
    print("printing numeric df")
    print(jobs_df.loc[:, numeric_cols])
    # Fill NaN values in numeric columns with column means
    jobs_df.loc[:, numeric_cols] = jobs_df[numeric_cols].fillna(jobs_df[numeric_cols].mean())




    # Optionally, replace NaN in non-numeric columns with a placeholder (e.g., 'Unknown')
    jobs_df.fillna('Unknown', inplace=True)


    p = preprocessor(jobs_df)
    feature_cols = ['cores_per_task', 'num_cores_req', 'num_cores_alloc', 'num_nodes_req', 'num_nodes_alloc', 'num_tasks', 'priority', 'num_gpus_req', 'num_gpus_alloc', 'mem_req', 'mem_alloc', 'time_limit']
    
    target_col = 'cpu_power_consumption'
    N = 100  # Number of time steps in the input sequence
    M = 10
    df_test, scaler, X_test_numerical = p.preprocess_data_disjoint_testing(feature_cols, target_col, N, M)
    #print(f"X_test_numerical {X_test_numerical}")
    # Create disjoint sequences for the time series data
    #X_time_series_train, y_train_sequences = p.create_disjoint_sequences3(df_train, target_col, N, M)  # CHANGED LINE
    X_time_series_test, y_test_sequences = p.create_disjoint_sequences_testing(df_test, target_col, N, M)  # CHANGED LINE
    #print(f"X_time_series_test {X_time_series_test}")
    """
    model = rnn_testing()
    _ , _, cpu_power_consumption_predictions = model(X_test_numerical, X_time_series_test)
    """
    model = timeseries_test("RNN")
    cpu_power_consumption_predictions = model(df_test)
    cpu_power_consumption_predictions = [t.detach().numpy() for t in cpu_power_consumption_predictions]
    print("cpu_power_consumption_predictions")
    print(cpu_power_consumption_predictions)
    print(f"printing length of cpu_power_consumptions {len(cpu_power_consumption_predictions)}")
    print(f"printing length of priority {len(df_test['priority'])}")
    print(f"printing length of num_gpus_req {len(df_test['num_gpus_req'])}")
    print(f"printing length of cores_per_task {len(df_test['cores_per_task'])}")
    sample_df_data=[]
    random.seed(42)
    for i in range(len(df_test["num_nodes_req"].values)):
        sample_df_data.append(df_test["num_nodes_req"].values[i])#(random.randint(1, 10))
    #print("printing sample data")
    #print(sample_df_data)
    sample_df = pd.DataFrame(columns=["num_nodes"], data = sample_df_data) #df_test["num_nodes_alloc"].values)
    
    sample_df["power_consumptions"] = cpu_power_consumption_predictions#.tolist()
    sample_df["priority"] = df_test["priority"]
    sample_df["time_limit"] = df_test["time_limit"]
    sample_df["cores_per_task"] =  df_test["cores_per_task"]
    sample_df["num_gpus_req"] = df_test["num_gpus_req"]
    sample_df2 = df_test [['num_nodes_req','num_gpus_req','cores_per_task','priority','time_limit']]
    sample_df2["power_consumptions"] = cpu_power_consumption_predictions
    """
    ###ranking
    # Hyperparameters and configuration
    num_features = 12  # Number of non-time-series features in tabular data
    time_series_length = 10  # Length of the time-series data
    lstm_hidden_size = 16
    dense_hidden_size = 32
    margin = 1.0  # Margin for the hinge loss

    # Instantiate the model
    model = TabularTimeSeriesRankingModel(num_features, time_series_length, lstm_hidden_size, dense_hidden_size)
    model = ranking_jobs_on_multi_feature_train(model, train_loader)

    ranking = get_job_ranking(model, x_test_tabular, x_test_ts)
    """
    #print("printing sample_df from ranking_generation")
    #print(sample_df)
    print("printing df_test num_gpus_req column")
    print(df_test["num_gpus_req"])
    print("printing dataframe before scoring function")
    print(sample_df["num_gpus_req"])
    print("printing sample_df2 num_gpus_req")
    print(sample_df2["num_gpus_req"])
    scores = scoring_function(sample_df2, feature_columns=['num_nodes', 'power_consumptions'], 
             time_series_column='power_consumptions', time_series_stat='mean')
    print("printing job scores")
    print(scores)
    """
    high_priority = len(indices)
    priority_array = []
    for i in range(high_priority):
        priority_array.append(0)
    for i in range(high_priority):
        priority_array[int(indices[i])] = int(i)
    print("printing job priorities")
    print(priority_array)
    """
    jobs_df["ml_priority"]= scores #priority_array #jobs_df["priority"]
    jobs_df = jobs_df.sort_values(by='submit_time')
    #jobs_df["num_nodes_req"] = sample_df["num_nodes"]
    #jobs_df["num_nodes_alloc"] = sample_df["num_nodes"]
    #sorted(jobs, key=lambda job: job.submit_time)
    return load_data_from_df(jobs_df, **kwargs)


def load_data_from_df(jobs_df: pd.DataFrame, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    config = kwargs.get('config')
    min_time = kwargs.get('min_time', None)
    arrival = kwargs.get('arrival')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')
    debug = kwargs.get('debug')

    #fastforward = kwargs.get('fastforward')
    #if fastforward:
    #    print(f"fast-forwarding {fastforward} seconds")

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # Dataset has one value from start to finish.
    # Therefore we set telemetry start and end equal to job start and end.
    first_start_timestamp = jobs_df['start_time'].min()
    telemetry_start_timestamp = first_start_timestamp

    last_end_timestamp = jobs_df['end_time'].max()
    telemetry_end_timestamp = last_end_timestamp

    telemetry_start = 0
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end = int(diff.total_seconds())

    num_jobs = len(jobs_df)

    if debug:
        print("num_jobs:", num_jobs)
        print("telemetry_start:", telemetry_start, "simulation_fin", telemetry_end)
        print("telemetry_start_timestamp:", telemetry_start_timestamp, "telemetry_end_timestamp", telemetry_end_timestamp)
        print("first_start_timestamp:",first_start_timestamp, "last start timestamp:", jobs_df['time_start'].max())

    jobs = []

    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):

        account = jobs_df.loc[jidx, 'user_id']  # or 'user_id' ?
        job_id = jobs_df.loc[jidx, 'job_id']
        # allocation_id =
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']
        end_state = jobs_df.loc[jidx, 'job_state']

        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']

        name = str(uuid.uuid4())[:6]  # This generates a random 6 char identifier....

        if validate:
            cpu_power = jobs_df.loc[jidx, 'node_power_consumption'] / jobs_df.loc[jidx, 'num_nodes_alloc']
            cpu_trace = cpu_power
            gpu_trace = cpu_trace

        else:
            cpu_power = jobs_df.loc[jidx, 'cpu_power_consumption']
            cpu_power_array = cpu_power.tolist()
            cpu_min_power = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * config['CPUS_PER_NODE']

            node_power = (jobs_df.loc[jidx, 'node_power_consumption']).tolist()
            mem_power = (jobs_df.loc[jidx, 'mem_power_consumption']).tolist()
            # Find the minimum length among the three lists
            min_length = min(len(node_power), len(cpu_power), len(mem_power))
            # Slice each list to the minimum length
            node_power = node_power[:min_length]
            cpu_power = cpu_power[:min_length]
            mem_power = mem_power[:min_length]

            gpu_power = (node_power - cpu_power - mem_power
                - ([nodes_required * config['NICS_PER_NODE'] * config['POWER_NIC']] * len(node_power))
                - ([nodes_required * config['POWER_NVME']] * len(node_power)))
            gpu_power_array = gpu_power.tolist()
            gpu_min_power = nodes_required * config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
            gpu_max_power = nodes_required * config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * config['GPUS_PER_NODE']

        priority = int(jobs_df.loc[jidx, 'priority'])

        # wall_time = jobs_df.loc[i, 'run_time']
        #wall_time = gpu_trace.size * config['TRACE_QUANTA'] # seconds
        #end_state = jobs_df.loc[jidx, 'job_state']
        #time_start = jobs_df.loc[jidx+1, 'start_time']
        #diff = time_start - time_zero
        ##adding new features here
        #cores_per_task = jobs_df.loc[jidx, 'cores_per_task']
        num_cores_req = jobs_df.loc[jidx, 'num_cores_req']
        num_cores_alloc = jobs_df.loc[jidx,'num_cores_alloc']
        num_nodes_req = jobs_df.loc[jidx, 'num_nodes_req']
        num_nodes_alloc = jobs_df.loc[jidx, 'num_nodes_alloc']
        num_tasks = jobs_df.loc[jidx, 'num_tasks']
        num_gpus_req = jobs_df.loc[jidx, 'num_gpus_req']
        num_gpus_alloc = jobs_df.loc[jidx,'num_gpus_alloc']
        mem_req = jobs_df.loc[jidx, 'mem_req']
        mem_alloc = jobs_df.loc[jidx, 'mem_alloc']
        threads_per_core = jobs_df.loc[jidx, 'threads_per_core']
        time_limit = jobs_df.loc[jidx, 'time_limit']
        cpu_power_consumption = jobs_df.loc[jidx, 'cpu_power_consumption']
        ml_priority = jobs_df.loc[jidx, 'ml_priority']
        ##new features end here
        """
        if jid == '*':
            time_offset = max(diff.total_seconds(), 0)
        else:
            # When extracting out a single job, run one iteration past the end of the job
            time_offset = config['UI_UPDATE_FREQ']
        """
        partition = int(jobs_df.loc[jidx, 'partition'])

        submit_timestamp = jobs_df.loc[jidx, 'submit_time']
        diff = submit_timestamp - telemetry_start_timestamp
        submit_time = int(diff.total_seconds())

        time_limit = jobs_df.loc[jidx, 'time_limit']

        start_timestamp = jobs_df.loc[jidx, 'start_time']
        diff = start_timestamp - telemetry_start_timestamp
        start_time = int(diff.total_seconds())

        end_timestamp = jobs_df.loc[jidx, 'end_time']
        diff = end_timestamp - telemetry_start_timestamp
        end_time = int(diff.total_seconds())

        wall_time = int(jobs_df.loc[jidx, 'run_time'])
        if np.isnan(wall_time):
            wall_time = 0
        if wall_time != (end_time - start_time):
            print("wall_time != (end_time - start_time)")
            print(f"{wall_time} != {(end_time - start_time)}")

        trace_time = gpu_trace.size * config['TRACE_QUANTA'] # seconds
        trace_start_time = 0
        trace_end_time = trace_time
        if wall_time > trace_time:
            missing_trace_time = wall_time - trace_time
            if start_time < 0:
                trace_start_time = missing_trace_time
                trace_end_time = wall_time
            elif end_time > telemetry_end:
                trace_start_time = 0
                trace_end_time = trace_time
            else:
                # Telemetry mission at the end
                trace_start_time = 0
                trace_end_time = trace_time
                trace_missing_values = True

        # What does this do?
        #if jid == '*':
        #    # submit_time = max(submit_time.total_seconds(), 0)
        #    submit_timestamp = jobs_df.loc[jidx, 'submit_time']
        #    diff = submit_timestamp - telemetry_start_timestamp
        #    submit_time = diff.total_seconds()


        #else:
        #    # When extracting out a single job, run one iteration past the end of the job
        #    submit_time = config['UI_UPDATE_FREQ']

        if arrival == 'poisson':  # Modify the arrival times according to Poisson distribution
            scheduled_nodes = None
            time_submit = next_arrival(1/config['JOB_ARRIVAL_TIME'])
            time_start = None
        else:  # Prescribed replay
            scheduled_nodes = (jobs_df.loc[jidx, 'nodes']).tolist()

        if gpu_trace.size > 0 and (jid == job_id or jid == '*'):  # and time_submit >= 0:
            """
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [], [],
                                end_state, scheduled_nodes,
                                job_id, priority, partition,
                                submit_time=submit_time, time_limit=time_limit,
                                start_time=start_time, end_time=end_time,
                                wall_time=wall_time, trace_time=trace_time,
                                trace_start_time=trace_start_time,
                                trace_end_time=trace_end_time,
                                trace_missing_values=trace_missing_values)

            jobs.append(job_info)
            """
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [], [], end_state, scheduled_nodes, job_id, num_cores_req, num_cores_alloc, num_nodes_req, num_nodes_alloc, num_tasks, num_gpus_req, num_gpus_alloc, mem_req, mem_alloc, threads_per_core, ml_priority, priority, partition,
                                submit_time=submit_time, time_limit=time_limit,
                                start_time=start_time, end_time=end_time,
                                wall_time=wall_time, trace_time=trace_time,
                                trace_start_time=trace_start_time,
                                trace_end_time=trace_end_time,
                                trace_missing_values=trace_missing_values)
            jobs.append(job_info)


    return jobs, telemetry_start, telemetry_end


def node_index_to_name(index: int, config: dict):
    """ Converts an index value back to an name string based on system configuration. """
    return f"node{index:04d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    return (0, index) # TODO
