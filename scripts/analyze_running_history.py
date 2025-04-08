import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

paths = []
if len(sys.argv) > 1:
    for arg in range(1, len(sys.argv)):
        paths.append(sys.argv[arg])
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir> ...")
    exit()

SECONDS_IN_DAY = 86400
SECONDS_IN_HOURS = 60 * 60 * 6
run_files = ['running_history.csv']
queue_files = ['queue_history.csv']
full_run_files = [os.path.join(path, file) for path in paths for file in run_files]
full_queue_files = [os.path.join(path, file) for path in paths for file in queue_files]

# Functions for calculations
def calculate_active_nodes(df):
    events = []
    for idx, row in df.iterrows():
        events.append((row['start_time'], row['num_nodes'], 1, row['energy'], row['avg_node_power']))
        events.append((row['end_time'], -row['num_nodes'], -1, -row['energy'], -row['avg_node_power']))

    events.sort()
    active_nodes = active_jobs = active_energy = node_power = 0
    time_series = []
    for event_time, nodes_change, jobs_change, energy_change, node_power_change in events:
        active_nodes += nodes_change
        active_jobs += jobs_change
        active_energy += energy_change
        node_power += node_power_change
        time_series.append((event_time, active_nodes, active_jobs, active_energy, node_power))

    return pd.DataFrame(time_series, columns=['time', 'active_nodes', 'active_jobs', 'active_energy', 'node_power'])

def calculate_queue_length(df):
    events = []
    for idx, row in df.iterrows():
        events.append((row['submit_time'], 1))
        events.append((row['start_time'], -1))

    events.sort()
    queue_length = 0
    queue_series = []
    for event_time, change in events:
        queue_length += change
        queue_series.append((event_time, queue_length))

    return pd.DataFrame(queue_series, columns=['time', 'queue_length'])

# Create subplots
fig = make_subplots(rows=5, cols=1,
                    subplot_titles=('Active Nodes Over Time', 'Active node power over time', 'Active Jobs Over Time', 
                                    'Cumulative Energy Over Time', 'Queue Length Over Time'))

# Assign colors to policies
policy_colors = {}
dark_colors = ["#00FF00", "#FF0000", "#36454F", "#FF8C00", "#0000FF", "#556B2F"]


for i, (run_file, queue_file) in enumerate(zip(full_run_files, full_queue_files)):
    dirname = os.path.basename(os.path.dirname(run_file))
    policy_type = dirname.removeprefix("fugaku").split("-")[0]
    if policy_type not in policy_colors:
        policy_colors[policy_type] = dark_colors[len(policy_colors) % len(dark_colors)]

    run_df = pd.read_csv(run_file)
    queue_df = pd.read_csv(queue_file)

    nodes = calculate_active_nodes(run_df)
    queue_length = calculate_queue_length(queue_df)

    nodes['time'] = nodes['time']/SECONDS_IN_DAY
    queue_length['time'] = queue_length['time']/SECONDS_IN_DAY

    fig.add_trace(go.Scatter(x=nodes['time'], y=nodes['active_nodes'], mode='lines', name=f'{policy_type}',
                             legendgroup=policy_type, showlegend=True, line=dict(color=policy_colors[policy_type])), row=1, col=1)
    fig.add_trace(go.Scatter(x=nodes['time'], y=nodes['node_power'], mode='lines', name=f'{policy_type}',
                             legendgroup=policy_type, showlegend=False, line=dict(color=policy_colors[policy_type])), row=2, col=1)
    fig.add_trace(go.Scatter(x=nodes['time'], y=nodes['active_jobs'], mode='lines', name=f'{policy_type}',
                             legendgroup=policy_type, showlegend=False, line=dict(color=policy_colors[policy_type])), row=3, col=1)
    fig.add_trace(go.Scatter(x=nodes['time'], y=nodes['active_energy'], mode='lines', name=f'{policy_type}',
                             legendgroup=policy_type, showlegend=False, line=dict(color=policy_colors[policy_type])), row=4, col=1)
    fig.add_trace(go.Scatter(x=queue_length['time'], y=queue_length['queue_length'], mode='lines', name=f'{policy_type}',
                             legendgroup=policy_type, showlegend=False, line=dict(color=policy_colors[policy_type])), row=5, col=1)

fig.update_layout(height=1200, title_text='Comparative Analysis of Scheduling Policies')
fig.update_xaxes(title_text="Time (s)", row=4, col=1)
fig.update_yaxes(title_text="Number of Active Nodes", row=1, col=1)
fig.update_yaxes(title_text="Cummulative node power", row=1, col=1)
fig.update_yaxes(title_text="Number of Active Jobs", row=3, col=1)
fig.update_yaxes(title_text="Cumulative Energy", row=4, col=1)
fig.update_yaxes(title_text="Queue Length", row=5, col=1)

fig.show()
fig.write_image("Fugaku-Analyse_runtime_history.pdf", width=2000, height=2000)
