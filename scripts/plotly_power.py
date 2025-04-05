import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import sys
import os

# Store command-line arguments (except script name)
paths = []
if len(sys.argv) > 1:
    for arg in range(1, len(sys.argv)):  # Start from index 1 to skip script name
        paths.append(sys.argv[arg])
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir> ...")
    exit()

print("Paths:", paths)

# File to look for
files = ['power_history.parquet']

# Construct full file paths
full_files = [os.path.join(path, file) for path in paths for file in files]

# Define colors for multiple plots
colours = ['blue', 'yellow', 'red']

# Create Plotly figure
fig = go.Figure()

for i, file in enumerate(full_files):
    try:
        df_power = pd.read_parquet(file)

        # Ensure the dataset has at least two columns
        if df_power.shape[1] >= 2:
            df_power.columns = ['time', 'power [kw]']  # Rename first two columns
        else:
            print(f"Warning: {file} has unexpected column format")
            continue

        # Add trace to Plotly figure
        fig.add_trace(go.Scatter(
            x=df_power['time'], 
            y=df_power['power [kw]'],
            mode='lines',
            name=os.path.basename(file),
            line=dict(color=colours[i % len(colours)])
        ))

    except Exception as e:
        print(f"Error reading {file}: {e}")

# Update layout
fig.update_layout(
    title="Power Consumption Over Time",
    xaxis_title="Time [s]",
    yaxis_title="Power [kW]",
    legend_title="Simulation Files",
    template="plotly_dark"
)

# Show plot
fig.show()
