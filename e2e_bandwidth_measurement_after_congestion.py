import subprocess
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import pandas as pd
from datetime import datetime
import time
import os

# import matplotlib.pyplot as plt

# ---- Configuration ----
# bandwidths = [f"{random.choice(range(50, 551, 10))}K" for _ in range(100)]  # 10 bandwidths
# bandwidths = [f"{random.randint(250)}K" for _ in range(50)]
bandwidths = [f"{bw}K" for bw in random.choices([450], k=300)]
ue_ip = "10.0.0.2"
congestion_log = []  # Stores (bandwidth, applied_rate_kbit)


def run_iperf(bw):
    cmd = [
        "docker", "exec", "oai-ext-dn", "iperf",
        "-u", "-c", ue_ip, "-b", bw, "-t", "10", "-i", "1"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

# Load Poisson distribution from CSV
def load_bandwidth_values(filename):
    values = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].isdigit():
                values.append(int(row[0]))
    return values

# Load and initialize index
bandwidth_values = load_bandwidth_values("poisson_distribution_with_drift_value_300(11).csv")
bandwidth_index = 0  # Track which value to use

def apply_congestion(bw):
    global bandwidth_index
    print("BW run iperf:", bw)
    
    if bandwidth_index >= len(bandwidth_values):
        print(" All congestion values from CSV have been used.")
        return

    rate_kbit = bandwidth_values[bandwidth_index]
    burst_kbit = 38  # or int(rate_kbit * 0.3)
    latency_ms = 300

    print(f"   → [Congestion] Applying rate: {rate_kbit}kbit, burst: {burst_kbit}kbit")

    # Store for later use
    congestion_log.append((bw, rate_kbit))

    # First, delete any existing qdisc on eth0
    result_del = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "del", "dev", "eth0", "root"
    ], capture_output=True, text=True)

     
   
    # Then add new qdisc with updated parameters
    
    result = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "add", "dev", "eth0", "root",
        "tbf", "rate", f"{rate_kbit}kbit",
        "burst", f"{burst_kbit}kbit",
        "latency", f"{latency_ms}ms"
    ], capture_output=True, text=True)

    bandwidth_index += 1  # Move to next CSV value

# def apply_congestion(bw):
#     global bandwidth_index
#     print("BW run iperf:", bw)
#     if bandwidth_index >= len(bandwidth_values):
#         print(" All congestion values from CSV have been used.")
#         return

#     rate_kbit = bandwidth_values[bandwidth_index]
#     burst_kbit = 38  #int(rate_kbit * 0.3)
#     latency_ms = 300

#     print(f"   → [Congestion] Applying rate: {rate_kbit}kbit, burst: {burst_kbit}kbit")

#     congestion_log.append((bw, rate_kbit))

#     subprocess.run([
#         "docker", "exec", "oai-upf", "tc", "qdisc", "add", "dev", "eth0", "root",
#         "tbf", "rate", f"{rate_kbit}kbit",
#         "burst", f"{burst_kbit}kbit",
#         "latency", f"{latency_ms}ms"
#     ], stderr=subprocess.DEVNULL)

#     bandwidth_index += 1  # move to the next CSV value


def remove_congestion():
    result = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "del", "dev", "eth0", "root"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "No such file" in result.stderr or result.returncode != 0:
        print("No existing qdisc on UPF eth0, skipping delete.")

def parse_output(output):
    matches = re.findall(r"([\d.]+)\s*(K|M|G)bits/sec", output)
    results = []
    for val, unit in matches:
        val = float(val)
        if unit == "M":
            val *= 1000
        elif unit == "G":
            val *= 1_000_000
        results.append(val)
    return results


def write_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bandwidth", "Measured Throughput [Kbps]", "Status", "Applied Congestion (kbps)"])
        for bw, val, status, cong_rate in data:
            writer.writerow([bw, f"{val:.2f}", status, cong_rate if cong_rate is not None else "N/A"])

def save_congestion_csv(filename="congestion_rates_100.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bandwidth", "Congestion Rate (kbit)"])
        for bw, rate in congestion_log:
            writer.writerow([bw, rate])


def write_conditional_csv(data, filename="single_result_view.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timeslot", "Bandwidth", "Measured Throughput (Kbps)", "Status", "Applied Congestion (if any)"])
        
        for i, row in enumerate(data, start=1):
            writer.writerow([i] + list(row))

with open("applied_congestion_rates.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Bandwidth", "Applied Congestion Rate (kbit)"])
    writer.writerows(congestion_log)

# ======================results part =========================================
results = []

for idx, bw in enumerate(bandwidths):
    bw_str = f"{bw}K" if isinstance(bw, int) else bw
    print(f"[{idx+1}/{len(bandwidths)}] Testing {bw_str} with congestion...")

    # if bandwidth_index >= len(bandwidth_values):
    #     print(" All congestion values from CSV have been used.")
    #     return
    #       # break
        
    rate_kbit = bandwidth_values[bandwidth_index]
    burst_kbit = 38  # or int(rate_kbit * 0.3)
    latency_ms = 300

    print(f"   → [Congestion] Applying rate: {rate_kbit}kbit, burst: {burst_kbit}kbit")

    # Store for later use
    congestion_log.append((bw, rate_kbit))

    # First, delete any existing qdisc on eth0
    result_del = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "del", "dev", "eth0", "root"
    ], capture_output=True, text=True)
     
   
    # Then add new qdisc with updated parameters
    
    result = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "add", "dev", "eth0", "root",
        "tbf", "rate", f"{rate_kbit}kbit",
        "burst", f"{burst_kbit}kbit",
        "latency", f"{latency_ms}ms"
    ], capture_output=True, text=True)

    bandwidth_index += 1  # Move to next CSV value

# apply_congestion(bw_str)
    
    applied_cong_bw = bandwidth_values[bandwidth_index - 1] if bandwidth_index > 0 else None
    # out = run_iperf(bw_str)
    out = run_iperf(bw)
    parsed = parse_output(out)
    val = np.mean(parsed) if parsed else 0.0
    # parsed = parse_bandwidth(out)
    # val = np.mean(parsed) if parsed else 0.0
    print(f"   → With congestion: {val:.2f} Kbps")
    # print("with cong:", val2)
    status = "with congestion"
    results.append((bw_str, val, status, applied_cong_bw))
    # remove_congestion()

# # ---- Save Output Files ----
# write_conditional_csv(results, filename="throughput_single_result.csv")

# results = []

# for idx, bw in enumerate(bandwidths):
#     if idx in congestion_indices:
#         print(f"[{idx+1}/{len(bandwidths)}] Testing {bw} with congestion...")
#         apply_congestion(bw)  # bw is like "300K"
#         # Get the applied congestion value from the previously incremented index
#         applied_cong_bw = bandwidth_values[bandwidth_index - 1]

#         out = run_iperf(bw)
#         parsed = parse_output(out)
#         val = np.mean(parsed) if parsed else 0.0
#         print(f"   → With congestion: {val}")
#         remove_congestion()
#         status = "With Congestion"
#     else:
#         print(f"[{idx+1}/{len(bandwidths)}] Testing {bw} without congestion...")
#         out = run_iperf(bw)
#         parsed = parse_output(out)
#         val = np.mean(parsed) if parsed else 0.0
#         print(f"   → Without congestion: {val}")
#         applied_cong_bw = None
#         status = "Without Congestion"

#     results.append((bw, val, status, applied_cong_bw))
    
# ---- Save Outputs and Plot Graphs ----
write_csv(results, "iperf_random_bandwidth_100.csv")
# plot_results(results, max_to_plot=100)
save_congestion_csv("congestion_rates_100.csv")

# ---- Save Output Files ----
write_conditional_csv(results, filename="throughput_single_result.csv")
# plot_single_type_result(results, filename="throughput_single_result_plot.png")

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("/home/ubuntu/oai-cn5g/throughput_single_result.csv")


#==============================live Plot============================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Data Preprocessing ----------------
df['Bandwidth'] = df['Bandwidth'].astype(str).str.replace('K', '', regex=False)
df['Bandwidth'] = pd.to_numeric(df['Bandwidth'], errors='coerce')
df['Applied Congestion (if any)'] = pd.to_numeric(df['Applied Congestion (if any)'], errors='coerce').fillna(0)
df['Measured Throughput (Kbps)'] = pd.to_numeric(df['Measured Throughput (Kbps)'], errors='coerce')
df = df.dropna(subset=['Bandwidth', 'Measured Throughput (Kbps)', 'Status'])

x = df['Timeslot'].to_numpy()

# ================= Plot 1: Measured Throughput =================
plt.figure(figsize=(15, 5))
y_throughput = df['Measured Throughput (Kbps)'].to_numpy()
colors_throughput = ['red' if s == 'With Congestion' else 'green' for s in df['Status']]

plt.ylim(260, 340)
plt.yticks(np.arange(260, 341, 20))
plt.plot(x, y_throughput, color='black', linestyle='-', linewidth=1)
plt.scatter(x, y_throughput, color=colors_throughput, s=40, marker='o')
plt.xlabel("Time (s)", fontsize=12, fontweight='bold', color='black')
plt.ylabel("Measured Throughput (Kbps)", fontsize=12, fontweight='bold', color='black')
plt.title("Throughput Over Time", fontsize=14, fontweight='bold', color='darkred')
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("throughput_plot.png", dpi=300)
plt.savefig("throughput_plot.eps", format='eps')
plt.show()

# ================= Plot 2: Applied Congestion =================
plt.figure(figsize=(15, 5))
y_congestion = df['Applied Congestion (if any)'].to_numpy()
colors_congestion = ['red' if c > 0 else 'gray' for c in y_congestion]

plt.plot(x, y_congestion, color='black', linestyle='-', linewidth=1)
plt.scatter(x, y_congestion, color=colors_congestion, s=40, marker='o')
plt.xlabel("Time (s)", fontsize=12, fontweight='bold', color='black')
plt.ylabel("Bandwidth Variation (Kbps)", fontsize=12, fontweight='bold', color='black')
plt.title("Bandwidth Variation Over Time", fontsize=14, fontweight='bold', color='darkred')
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("congestion_plot.png", dpi=300)
plt.savefig("congestion_plot.eps", format='eps')
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ---------------- Data Preprocessing ----------------
# df['Bandwidth'] = df['Bandwidth'].astype(str).str.replace('K', '', regex=False)
# df['Bandwidth'] = pd.to_numeric(df['Bandwidth'], errors='coerce')
# df['Applied Congestion (if any)'] = pd.to_numeric(df['Applied Congestion (if any)'], errors='coerce').fillna(0)
# df['Measured Throughput (Kbps)'] = pd.to_numeric(df['Measured Throughput (Kbps)'], errors='coerce')
# df = df.dropna(subset=['Bandwidth', 'Measured Throughput (Kbps)', 'Status'])

# # Enable interactive plotting (optional for dynamic environments)
# # plt.ion()
# x = df['Timeslot'].to_numpy()
# # ---------------- Subplots ----------------
# # fig, axs = plt.subplots(2, 1, figsize=(15, 6))
# fig, axs = plt.subplots(figsize=(15, 5))

# # plt.ylim(260, 340)
# # plt.yticks(np.arange(260, 340, 20))


# # ---------------- Plot 1: Measured Throughput ----------------

# y_throughput = df['Measured Throughput (Kbps)'].to_numpy()
# colors_throughput = ['red' if s == 'With Congestion' else 'green' for s in df['Status']]

# axs.set_ylim(260, 340)
# axs.set_yticks(np.arange(260, 341, 20))
# axs.plot(x, y_throughput, color='black', linestyle='-', linewidth=1)
# axs.scatter(x, y_throughput, color=colors_throughput, s=40, marker='o')
# axs.set_xlabel("Time (s)", fontsize=12, fontweight='bold', color='black')
# axs.set_ylabel("Measured Throughput (Kbps)", fontsize=12, fontweight='bold', color='black')
# axs.set_title("Throughput Over Time", fontsize=14, fontweight='bold', color='darkred')
# #axs[0].set_title("Throughput Over Time\n(Red = Congestion, Green = No Congestion)")
# axs.grid(True, linestyle='--')

# # ---------------- Plot 2: Applied Congestion ----------------
# y_congestion = df['Applied Congestion (if any)'].to_numpy()
# colors_congestion = ['red' if c > 0 else 'gray' for c in y_congestion]

# axs.plot(x, y_congestion, color='black', linestyle='-', linewidth=1)
# axs.scatter(x, y_congestion, color=colors_congestion, s=40, marker='o')
# axs.set_xlabel("Time (s)", fontsize=12, fontweight='bold', color='black')
# axs.set_ylabel("Bandwidth Variation (Kbps)", fontsize=12, fontweight='bold', color='black')
# axs.set_title("Bandwidth Variation Over Time", fontsize=14, fontweight='bold', color='darkred')
# axs.grid(True, linestyle='--')

# # Final Layout
# plt.tight_layout()
# plt.savefig("uniform_throughput_congestion_plot1.png", dpi=300)
# #plt.savefig("uniform_throughput_congestion_plot.eps",format=eps)
# plt.savefig("uniform_throughput_congestion_plot.eps", format='eps')

# plt.show()
# plt.ioff()

# 

# axs[1].grid(True, linestyle='--', alpha=0.5)


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ------------------- Preprocess -------------------
# df['Bandwidth'] = df['Bandwidth'].astype(str).str.replace('K', '', regex=False)
# df['Bandwidth'] = pd.to_numeric(df['Bandwidth'], errors='coerce')
# df['Applied Congestion (if any)'] = pd.to_numeric(df['Applied Congestion (if any)'], errors='coerce').fillna(0)
# df['Measured Throughput (Kbps)'] = pd.to_numeric(df['Measured Throughput (Kbps)'], errors='coerce')
# df = df.dropna(subset=['Bandwidth', 'Measured Throughput (Kbps)', 'Status'])

# # ------------------- Variables -------------------
# timeslot = df['Timeslot'].to_numpy()
# throughput = df['Measured Throughput (Kbps)'].to_numpy()
# congestion = df['Applied Congestion (if any)'].to_numpy()

# # ------------------- Plot Setup -------------------
# fig, axs = plt.subplots(2, 1, figsize=(16, 6))

# # -------- Plot 1: Throughput with congestion pattern --------
# for i in range(1, len(timeslot)):
#     x = timeslot[i-1:i+1]
#     y = throughput[i-1:i+1]

#     if congestion[i] > 0:
#         # Congestion: red with waviness
#         x_dense = np.linspace(x[0], x[1], 10)
#         wave = np.interp(x_dense, x, y) + 10 * np.sin(10 * x_dense)
#         axs[0].plot(x_dense, wave, color='red', linewidth=2)
#     else:
#         # No congestion: smooth green line
#         axs[0].plot(x, y, color='green', linewidth=2)

# axs[0].set_title("Throughput Over Time")
# axs[0].set_xlabel("Time(s)")
# axs[0].set_ylabel("Measured Throughput (Kbps)")
# axs[0].grid(True, linestyle='--', alpha=0.5)

# # -------- Plot 2: Applied Congestion with same pattern --------
# for i in range(1, len(timeslot)):
#     x = timeslot[i-1:i+1]
#     y = congestion[i-1:i+1]

#     if congestion[i] > 0:
#         x_dense = np.linspace(x[0], x[1], 10)
#         wave = np.interp(x_dense, x, y) + 5 * np.sin(10 * x_dense)
#         axs[1].plot(x_dense, wave, color='red', linewidth=2)
#     else:
#         axs[1].plot(x, y, color='gray', linewidth=2)

# axs[1].set_title("Applied Congestion Over Time")
# axs[1].set_xlabel("Time(s)")
# axs[1].set_ylabel("Applied Congestion (kbit)")
# axs[1].grid(True, linestyle='--', alpha=0.5)

# # Finalize
# plt.tight_layout()
# plt.savefig("wavy_congestion_plot.png")
# plt.show()

# # =================output======================================
# print("\n All plots and CSV files have been saved successfully!")
# print("   → throughput_single_result.csv")
# print("   → plot_throughput_vs_bandwidth.png")
# print("   → plot_congestion_vs_bandwidth.png")

# print("\n CSV saved to 'throughput_single_result.csv'")
# # print(" Plot saved to 'throughput_single_result_plot.png'")

# print("\n[INFO] All CSV and plots saved for 100 bandwidths.")

