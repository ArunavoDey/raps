import os
import re
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

pio.kaleido.scope.mathjax = None

# === CONFIGURATION ===
plots = ['power', 'util', 'stats']  # You can remove any of these to toggle
plot_files = {
    'power': 'power_history.parquet',
    'util': 'util.parquet',
    'stats': 'stats.out'
}

policy_color_map = {
    "sjf": "#00FF00",
    "fcfs": "skyblue",
    "ljf": "#FF0000",
    "priority": "brown",
    "ml": "#0000FF",
}
r_policy_color_map = {
    "sjf": "#66C2A5",     # soft green → similar to "#00FF00"
    "fcfs": "#A6CEE3",    # soft skyblue → similar to "skyblue"
    "ljf": "#FB9A99",     # soft red → less harsh than "#FF0000"
    "priority": "#B15928",# rich brown → deeper than plain "brown"
    "ml": "#1F78B4"       # academic blue → alternative to "#0000FF"
}


def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


policy_order = list(policy_color_map.keys())
SECONDS_IN_DAY = 86400

display_name_map = {
    "Average Wait time": "Average<br>Wait time",
    "Average Turnaround Time": "Average<br>Turnaround Time",
    "Inverse Throughput": "Inverse<br>Job Throughput",
    "Inverse Total Jobs Completed": "Inverse Total<br>Jobs Completed",
    "Avg EDP^2": "Avg EDP^2",
    "Avg Energy": "Avg Energy",
    "Average Runtime": "Average<br>Runtime",
    "Priority-Weighted Specific Response Time": "Priority-Weighted<br>Specific Response Time",
    "Area-Weighted Avg Response Time": "Area-Weighted Avg<br>Response Time",
    "Avg Aggregate Node Hours": "Avg Aggregate<br>Node Hours",
    "Inverse Avg CPU util": "Inverse Avg<br>CPU util",
    "Inverse Avg GPU util": "Inverse Avg<br>GPU util"
}

# === STAT PROCESSING ===
def extract_stats(file_path):
    stats = {}
    patterns = {
        "Average Wait time": r'"average_wait_time": ([\d\.]+)',
        "Average Turnaround Time": r'"average_turnaround_time": ([\d\.]+)',
        "Avg Aggregate Node Hours": r'"avg_aggregate_node_hours": ([\d\.]+)',
        "Avg EDP^2": r'"avg edp\^2": ([\d\.]+)',
        "Inverse Total Jobs Completed": r'"jobs completed": (\d+)',
        "Inverse Throughput": r'"throughput": "([\d\.]+) jobs/hour"',
        "Average Runtime": r'"average runtime": ([\d\.]+)',
        "Inverse Avg CPU util": r'"avg_cpu_util": ([\d\.]+)',
        "Inverse Avg GPU util": r'"avg_gpu_util": ([\d\.]+)',
        "Priority-Weighted Specific Response Time": r'"priority_weighted_specific_response_time": ([\d\.]+)',
        "Avg Energy": r'"avg energy": ([\d\.]+)',
        "Area-Weighted Avg Response Time": r'"area_weighted_avg_response_time": ([\d\.]+)',
    }

    with open(file_path, 'r') as file:
        content = file.read()
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1)) if key != "Inverse Total Jobs Completed" else int(match.group(1))
                if key == "Inverse Throughput":
                    value = np.exp(-value)
                elif key in ["Inverse Total Jobs Completed", "Inverse Avg CPU util", "Inverse Avg GPU util"]:
                    value = 1 / value if value != 0 else 0
                stats[key] = value
            else:
                stats[key] = None
    return stats

def normalize_stats_minmax(all_stats):
    categories = list(all_stats[next(iter(all_stats))].keys())
    min_vals = {col: min(stats[col] for stats in all_stats.values()) for col in categories}
    max_vals = {col: max(stats[col] for stats in all_stats.values()) for col in categories}
    normalized_stats = {}
    for policy, stats in all_stats.items():
        rescaled_data = {}
        for col in categories:
            min_val, max_val = min_vals[col], max_vals[col]
            if max_val != min_val:
                rescaled_data[col] = (stats[col] - min_val) / (max_val - min_val)
            else:
                rescaled_data[col] = 0.5
        normalized_stats[policy] = rescaled_data
    return normalized_stats

# === INPUT HANDLING ===
weeks = sys.argv[1:]
if not weeks:
    print(f"Usage: python {sys.argv[0]} <week_dir1> <week_dir2> ...")
    sys.exit(1)

week_to_policies = {}
for week in weeks:
    policy_dirs = [os.path.join(week, d) for d in os.listdir(week)
                   if os.path.isdir(os.path.join(week, d))]

    def policy_sort_key(d):
        policy_type = os.path.basename(d).removeprefix("fugaku").split("-")[0]
        return policy_order.index(policy_type) if policy_type in policy_order else len(policy_order)

    policy_dirs.sort(key=policy_sort_key)
    week_to_policies[week] = policy_dirs

num_weeks = len(weeks)


# Dynamically create specs for each row
specs = []
row_heights = []
for plot_type in plots:
    if plot_type == "stats":
        specs.append([{"type": "polar"}] * num_weeks)
        row_heights.append(1.0)  
    else:
        specs.append([{}] * num_weeks)
        row_heights.append(1.0)

# Create subplot grid with correct specs
fig = sp.make_subplots(
    rows=len(plots), cols=num_weeks,
    specs=specs,
    row_heights=row_heights,
    vertical_spacing=0.1,
    horizontal_spacing=0.04,
)

# === PLOT LOOP ===
for i, (week, policy_dirs) in enumerate(week_to_policies.items()):
    for j, plot_type in enumerate(plots):
        row = j + 1
        show_legend = i == 0 and row == 1  # Only show legend on first row/first week

        if plot_type == 'power':
            for policy_dir in policy_dirs:
                dirname = os.path.basename(policy_dir)
                policy_type = dirname.removeprefix("fugaku").split("-")[0]
                file_path = os.path.join(policy_dir, plot_files['power'])

                if not os.path.exists(file_path):
                    continue

                df = pd.read_parquet(file_path)
                df.columns = ['time', 'power [kw]']
                df['time'] /= SECONDS_IN_DAY

                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['power [kw]'],
                    mode='lines',
                    name=policy_type,
                    line=dict(color=policy_color_map[policy_type]),
                    showlegend=show_legend
                ), row=row, col=i+1)

        elif plot_type == 'util':
            for policy_dir in policy_dirs:
                dirname = os.path.basename(policy_dir)
                policy_type = dirname.removeprefix("fugaku").split("-")[0]
                file_path = os.path.join(policy_dir, plot_files['util'])

                if not os.path.exists(file_path):
                    continue

                df = pd.read_parquet(file_path)
                df.columns = ['time', 'utilization [%]']
                df['time'] /= SECONDS_IN_DAY

                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['utilization [%]'],
                    mode='lines',
                    name=policy_type,
                    line=dict(color=policy_color_map[policy_type]),
                    showlegend=show_legend
                ), row=row, col=i+1)

        elif plot_type == 'stats':
            all_stats = {}
            for policy_dir in policy_dirs:
                stats_file = os.path.join(policy_dir, plot_files['stats'])
                if not os.path.exists(stats_file):
                    continue
                dirname = os.path.basename(policy_dir)
                policy_type = dirname.removeprefix("fugaku").split("-")[0]
                stats = extract_stats(stats_file)
                all_stats[policy_type] = stats

            if not all_stats:
                continue

            normalized_stats = normalize_stats_minmax(all_stats)
            for policy, stats in normalized_stats.items():
                categories = list(stats.keys())
                values = list(stats.values())
                wrapped_labels = [display_name_map.get(label, label) for label in categories]

                subplot_ref = "polar" if i == 0 else f"polar{i+1}"

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=wrapped_labels,
                    fill='toself',
                    name=policy,
                    subplot=subplot_ref,
                    line=dict(color=r_policy_color_map[policy], width=3),
                    fillcolor=hex_to_rgba(r_policy_color_map[policy], 0.2),
                    opacity=1.0,
                    showlegend=show_legend
                ),row=row, col=i+1)

                """
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=wrapped_labels,
                    fill='toself',
                    name=policy,
                    subplot=subplot_ref,
                    line=dict(
                        color=policy_color_map[policy],
                        dash='solid',
                        width=1
                    ),
                    opacity=0.2,
                    showlegend=show_legend
                ), row=row, col=i+1)"""

# === AXES AND LAYOUT ===
for col in range(1, num_weeks + 1):
    if 'power' in plots:
        fig.update_yaxes(title_text="Power [kW]" if col == 1 else None, row=1, col=col, showline=True, linecolor='black', showgrid=True, gridcolor='lightgray')
    if 'util' in plots:
        fig.update_yaxes(title_text="Utilization [%]" if col == 1 else None, row=2, col=col, showline=True, linecolor='black', showgrid=True, gridcolor='lightgray')

    for row in range(1, len(plots) + 1):
        fig.update_xaxes(
            title_text="Time [days]" if plots[row - 1] in ['power', 'util'] else "",
            row=row, col=col, showline=True,
        linecolor='black',
        showgrid=True,
        gridcolor='lightgray'  # subtle grid color
        )
    

if 'stats' in plots:
    for col in range(1, num_weeks + 1):
        polar_id = "" if col == 1 else str(col)
        fig.update_layout({
            f'polar{polar_id}': dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',   # light gray, semi-transparent
                    linecolor='rgba(128,128,128,0.3)',   # same for axis line
                    #gridcolor='grey',
                    #linecolor='grey',
                    #visible=True,
                    #tickvals=[0, 0.5, 1.0],          
                    ticktext=["0", "0.5", "1.0"],
                    range=[0, 1]
                ),
                angularaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',   # light gray, semi-transparent
                    linecolor='rgba(128,128,128,0.3)',
                    #gridcolor='grey',
                    #linecolor='grey',
                    #visible=True
                )
            )
        })

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.05,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=10, r=10, t=30, b=20),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=80),
)

# === SAVE ===
fig.write_image("Fugaku-power-utilization-stats-all.pdf", width=3000 * len(weeks), height= 2000 * len(plots))
print("Plot saved to: Fugaku-power-utilization-stats-all.pdf")
