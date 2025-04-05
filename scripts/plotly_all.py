import pandas as pd
import plotly.graph_objects as go
import sys

def iter_to_seconds(i):
    return i * 15

# Check if a path argument is provided
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir>")
    exit()

# Define file paths
files = ['cooling_model.parquet', 'loss_history.parquet', 'power_history.parquet', 'util.parquet']
power_file = f"{path}/{files[2]}"
util_file = f"{path}/{files[3]}"
loss_file = f"{path}/{files[1]}"

# Read and preprocess power data
df_power = pd.read_parquet(power_file)
df_power = df_power.rename(columns={0: 'time', 1: 'power [kW]'})

df_loss = pd.read_parquet(loss_file)
df_loss = df_loss.rename(columns={0:'time',1:'loss [kW]'})


# Read and preprocess utilization data
df_util = pd.read_parquet(util_file)
df_util = df_util.rename(columns={0: 'time', 1: 'utilization [%]'})
df_util['utilization'] = df_util['utilization [%]'] / 100  # Convert to fraction

# Determine Y-axis max value
ymax = max(0, max(df_util['utilization']))

# Create Plotly figure
fig = go.Figure()

# Add Power (kW) - Left Y-axis
fig.add_trace(go.Scatter(x=df_power['time'], y=df_power['power [kW]'], 
                         mode='lines', name='Power [kW]', line=dict(color='black')))

# Add Power (kW) - Left Y-axis
fig.add_trace(go.Scatter(x=df_loss['time'], y=df_loss['loss [kW]'], 
                         mode='lines', name='loss [kW]', line=dict(color='red')))

# Add Utilization (%) - Right Y-axis
fig.add_trace(go.Scatter(x=df_util['time'], y=df_util['utilization'], 
                         mode='lines', name='Utilization [%]', line=dict(color='orange'), yaxis='y2'))

# Layout adjustments
fig.update_layout(
    title=f"Simulation Results: {path}",
    xaxis_title="Time [s]",
    yaxis=dict(title="Power [kW]", side="left"),
    yaxis2=dict(title="Utilization [%]", overlaying='y', side="right", range=[0, ymax * 1.05]),
    legend=dict(x=0.01, y=0.99),
)

# Show Plot
fig.show()
