import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
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
full_files = [f"{path}/{file}" for path in paths for file in files]

def iter_to_seconds(i):
    return i * 15

fig, ax1 = plt.subplots(figsize=(10, 6))

colours = ['black', 'yellow', 'red', 'blue', 'green', 'orange']

for i, file in enumerate(full_files):
    try:
        dirname = os.path.basename(os.path.dirname(file))
        policy_type = dirname.removeprefix("fugaku").split("-")[0]
        print(policy_type)
        df_power = pd.read_parquet(file)

        # Ensure the columns exist
        if df_power.shape[1] >= 2:
            df_power.columns = ['time', 'power [kw]']  # Rename first two columns
        else:
            print(f"Warning: {file} has unexpected column format")
            continue

        ax1.plot(df_power['time'], df_power['power [kw]'], color=colours[i % len(colours)], label=policy_type)

    except Exception as e:
        print(f"Error reading {file}: {e}")

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Power [kW]')
ax1.legend(loc='upper left')
plt.savefig("test.png")
plt.show()

