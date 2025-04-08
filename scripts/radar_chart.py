import os
import re
import sys
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
pio.kaleido.scope.mathjax = None

# Maps long metric names to wrapped versions for plotting
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
        "Inverse Avg CPU util": "Inverse Avg <br> CPU util",
        "Inverse Avg GPU util": "Inverse Avg <br> GPU util"
    }


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
        "Inverse Avg CPU util" : r'"avg_cpu_util": ([\d\.]+)',
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

                # Apply inverse transformations
                if key == "Inverse Throughput":
                    value = np.exp(-value)
                elif key == "Inverse Total Jobs Completed":
                    value = 1 / value if value != 0 else 0
                elif key == "Inverse Avg CPU util":
                    value = 1 / value if value != 0 else 0
                elif key == "Inverse Avg GPU util":
                    value = 1 / value if value != 0 else 0


                stats[key] = value
            else:
                stats[key] = None

    return stats

def normalize_stats_l2(all_stats):
    categories = list(all_stats[next(iter(all_stats))].keys())
    data_matrix = np.array([[stats[col] for col in categories] for stats in all_stats.values()])

    normalized_stats = {}
    for policy, stats in all_stats.items():
        rescaled_data = {}
        for col in categories:
            norm = np.sqrt(np.sum(np.square([s[col] for s in all_stats.values()])))
            rescaled_data[col] = stats[col] / norm if norm != 0 else 0.5
        normalized_stats[policy] = rescaled_data

    return normalized_stats

def normalize_stats_minmax(all_stats):
    categories = list(all_stats[next(iter(all_stats))].keys())

    # Compute min and max for each category
    min_vals = {col: min(stats[col] for stats in all_stats.values()) for col in categories}
    max_vals = {col: max(stats[col] for stats in all_stats.values()) for col in categories}

    normalized_stats = {}

    for policy, stats in all_stats.items():
        rescaled_data = {}
        for col in categories:
            min_val = min_vals[col]
            max_val = max_vals[col]
            # Avoid division by zero if all values are the same
            if max_val != min_val:
                rescaled_data[col] = (stats[col] - min_val) / (max_val - min_val)
            else:
                rescaled_data[col] = 0.5  # Assign midpoint if no variation
        normalized_stats[policy] = rescaled_data

    return normalized_stats


def plot_combined_radar_chart(all_stats):
    normalized_stats = normalize_stats_minmax(all_stats)
    fig = go.Figure()
    colors = ["#C4D7A6", "#F4A6A6", "#BDC9D1", "#FFD4A3", "#A3B8E2", "#C4D7A6"]
    dash_styles = ["solid", "dash", "dash", "dashdot", "longdash", "longdashdot"]

    for i, (policy, stats) in enumerate(normalized_stats.items()):
        categories = list(stats.keys())
        values = list(stats.values())

        # Apply wrapped display names for labels
        wrapped_labels = [display_name_map.get(label, label) for label in categories]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=wrapped_labels,
            fill='toself',
            name=policy,
            line=dict(
                color=colors[i % len(colors)],
                dash=dash_styles[i % len(dash_styles)],
                width=3
            ),
            opacity=1.0,
        ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=18),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                gridcolor='grey',
                linecolor='grey',
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                gridcolor='grey',
                linecolor='grey',
                visible=True
            ),
        ),
    )

    fig.show()
    fig.write_image("Fugaku_scheduling_radarchart.pdf", width=2000, height=2000)


if __name__ == "__main__":
    paths = []

    if len(sys.argv) > 1:
        for arg in range(1, len(sys.argv)):
            paths.append(sys.argv[arg])
    else:
        print(f"Usage: python {sys.argv[0]} <simulation_result/dir> ...")
        exit()

    print("Paths:", paths)
    files = ['stats.out']
    full_files = [os.path.join(path, file) for path in paths for file in files]

    all_stats = {}

    for file_path in full_files:
        dirname = os.path.basename(os.path.dirname(file_path))
        policy_type = dirname.removeprefix("fugaku").split("-")[0]
        print(policy_type)

        extracted_stats = extract_stats(file_path)

        for key, value in extracted_stats.items():
            print(f"{key}: {value}")

        all_stats[policy_type] = extracted_stats

    plot_combined_radar_chart(all_stats)
