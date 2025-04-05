import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import plotly.io as pio
pio.kaleido.scope.mathjax = None

# Store command-line arguments (except script name)
paths = []
if len(sys.argv) > 1:
    for arg in range(1, len(sys.argv)):  # Start from index 1 to skip script name
        paths.append(sys.argv[arg])
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir> ...")
    exit()
print("Paths:", paths)
SECONDS_IN_DAY = 86400

# File to look for
files = ['power_history.parquet']
# Construct full file paths
full_files = [os.path.join(path, file) for path in paths for file in files]
# Define colors for multiple plots
#colours = ['black', 'yellow', 'red', 'orange', 'blue']
#colours = ['#b4d2b1', '#568f8b', '#1d4a60', '#a35d6a', '#cd7e59']
dark_colors = [ "#00FF00", "#FF0000", "#36454F", "#FF8C00", "#0000FF", "#556B2F",]
pastel_colors = ["#C4D7A6", "#F4A6A6", "#BDC9D1", "#FFD4A3", "#A3B8E2", "#C4D7A6",]


# Create Plotly figure
fig = go.Figure()
for i, file in enumerate(full_files):
    try:
        dirname = os.path.basename(os.path.dirname(file))
        policy_type = dirname.removeprefix("fugaku").split("-")[0]
        print(policy_type)
        df_power = pd.read_parquet(file)
        print(df_power.columns)
        print(df_power.head())
        # Ensure the dataset has at least two columns
        if df_power.shape[1] >= 2:
            df_power.columns = ['time', 'power [kw]']  # Rename first two columns
        else:
            print(f"Warning: {file} has unexpected column format")
            continue

        # Convert time from seconds to days
        df_power['time'] = df_power['time'] / SECONDS_IN_DAY
        fig.add_trace(go.Scatter(
            x=df_power['time'],
            y=df_power['power [kw]'],
            mode='lines',
            name=policy_type,
            line=dict(color=dark_colors[i % len(dark_colors)])
        ))
    except Exception as e:
        print(f"Error reading {file}: {e}")
# Update layout

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,         # adjust vertical position (negative = below plot)
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=10, r=10, t=20, b=20),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        title="Time [days]",
        #tickfont=dict(color='black'),
        #titlefont=dict(color='black'),
        showline=True,
        linecolor='black',
        showgrid=True,
        gridcolor='lightgray'  # subtle grid color
        #zeroline=False
    ),
    yaxis=dict(
        title="Power [kW]",
        #tickfont=dict(color='black'),
        #titlefont=dict(color='black'),
        showline=True,
        linecolor='black',
        showgrid=True,
        gridcolor='lightgray'
        #zeroline=False
    ),
    font=dict(
        size=20  # Increase font size here
    )
)
# Show plot
fig.write_image("Fugaku-power-vs-time.pdf")