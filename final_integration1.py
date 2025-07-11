import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import subprocess
import re
import time
import matplotlib.pyplot as plt1
import numpy as np
import csv
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import pandas as pd
import os
import random
from matplotlib.gridspec import GridSpec

# Define IP and other settings
ue_ip = "10.0.0.2"  # UE IP address
cn5g_ip = "192.168.70.135"  # CN5G DN IP address
upf_ip = "192.168.72.134"  # UPF IP address for congestion generation
duration = 10  # Duration of each test in seconds
# bandwidths = ['300K', '200K','50K', '400K'] # '100K','550K', '110K', '75K','250K', '150K', '350K', '150K', '450K', '500K', '100M', '150M', '50M'] 
# bandwidths = [f"{random.randint(50, 550)}K" for _ in range(10)]
# congestion_log = []  # Stores (bandwidth, applied_rate_kbit)
# poisson_lambda = 1  # Lambda for Poisson distribution

# Function to run iPerf test
def run_iperf_test(bandwidth, mode="downlink"):
    if mode == "downlink":
        command = [
            "docker", "exec", "-it", "oai-ext-dn", "iperf",
           "-u",  "-c", ue_ip, "-b", bandwidth, "-t", str(duration), "-i", "1"
        ]
    elif mode == "uplink":
        command = [
            "iperf", "-c", "-u", cn5g_ip, "-b", bandwidth, "-t", str(duration), "-i", "1", "-B", ue_ip
        ]
    else:
        raise ValueError("Invalid mode. Choose 'uplink' or 'downlink'.")

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"Error running iPerf: {e}")
        return ""

# # Load Poisson distribution from CSV
def load_bandwidth_values(filename):
    values = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].isdigit():
                values.append(int(row[0]))
    return values

# # Load and initialize index
# bandwidth_values = load_bandwidth_values("poisson_distribution_with_drift_value_300(8).csv")
# bandwidth_index = 0  # Track which value to use

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
    
def remove_congestion():
    result = subprocess.run([
        "docker", "exec", "oai-upf", "tc", "qdisc", "del", "dev", "eth0", "root"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if "No such file" in result.stderr or result.returncode != 0:
        print("No existing qdisc on UPF eth0, skipping delete.")

# Function to generate congestion traffic at UPF
# def generate_congestion_traffic():
#     command = [
#         "iperf", "-c", "-u", upf_ip, "-b", "900K", "-t", str(duration), "-i", "1"
#     ]
#     try:
#         subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         print("Congestion traffic to UPF generated.")
#     except Exception as e:
#         print(f"Error generating congestion traffic: {e}")

def write_conditional_csv(data, filename="single_result_view.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timeslot", "Bandwidth", "Measured Throughput (Kbps)", "Status", "Applied Congestion"])
        
        for i, row in enumerate(data, start=1):
            writer.writerow([i] + list(row))

# Function to parse bandwidth data from iPerf output
def parse_bandwidth_data(output):
    bandwidth_pattern = r"([\d\.]+)\s*(K|M|G)bits/sec"
    matches = re.findall(bandwidth_pattern, output)
    bandwidth_values = []

    for value, unit in matches:
        if unit == "K":
            bandwidth_values.append(float(value))  # In Kbps
        elif unit == "M":
            bandwidth_values.append(float(value) * 1000)  # Convert Mbps to Kbps
        elif unit == "G":
            bandwidth_values.append(float(value) * 1000000)  # Convert Gbps to Kbps

    return bandwidth_values

# Function to rearrange bandwidths based on Poisson distribution
def rearrange_bandwidths_poisson(bandwidths, poisson_lambda):
    poisson_weights = np.random.poisson(poisson_lambda, len(bandwidths))
    bandwidth_ordered = [x for _, x in sorted(zip(poisson_weights, bandwidths))]
    return bandwidth_ordered


# Function to write combined data to a CSV file
def write_combined_csv(data, filename="combined_data.csv"):
    import pandas as pd
    import os

    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            raise ValueError("Input data must be a Pandas DataFrame or convertible to one.") from e

    # Save to CSV
    file_path = os.path.join("output", filename)
    os.makedirs("output", exist_ok=True)
    data.to_csv(file_path, index=False)
    print(f"Combined data saved to: {file_path}")

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return scaled_data, scaler

# Function to prepare training and testing datasets
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Function to create and train BiLSTM model
def train_bilstm_model(X_train, y_train, X_test, y_test, input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    print("length of training and testing data:", len(X_train),len(X_test))
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    model.save('user_count_predictor_with_lambda_one_testbed')
    return model

# Function to predict future bandwidth using BiLSTM
def predict_future_bandwidth(model, recent_data, future_steps):
    predictions = []
    current_input = recent_data  # Shape: (1, time_steps, features)
   
    print("current_input shape:", current_input.shape)
#    print("prediction_expanded shape:", prediction_expanded.shape)
    for _ in range(future_steps):
        # Predict the next value
        prediction = model.predict(current_input, verbose=0)  # Shape: (1, 1)
        predictions.append(prediction[0, 0])  # Save the predicted scalar value

    return predictions


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate the model and compare predicted vs actual values
def evaluate_model(model, X_test, y_test, scaler):
    # Predict on the test set
    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)  # Inverse scale predictions
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse scale actual values
    print("actual_values:", actual_values)

    # Calculate evaluation metrics
    mse = mean_squared_error(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")

    return predictions, actual_values

import xgboost as xgb

def analyze_residuals_and_train_xgboost(X_test, predicted_counts_scaled, actual_counts_scaled, scaler, sequence_length):
    # Inverse transform the scaled values
    predicted_counts = scaler.inverse_transform(predicted_counts_scaled)
    actual_counts = scaler.inverse_transform(actual_counts_scaled.reshape(-1, 1))

    # Calculate residuals (actual - predicted)
    residuals = actual_counts - predicted_counts

    # Plot residuals vs. predicted values (Residual Plot)
    residuals_flat = residuals.flatten()
    predicted_flat = predicted_counts.flatten()

    # Train an XGBoost model to predict residuals
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    # Reshape X_test and residuals for XGBoost training
    xgb_model.fit(X_test.reshape(-1, sequence_length), residuals_flat)

    # Predict residuals with the trained XGBoost model
    residuals_pred = xgb_model.predict(X_test.reshape(-1, sequence_length)).reshape(-1, 1)

    # Combine predictions from BiLSTM and XGBoost
    final_predictions = predicted_counts + residuals_pred
    print("actual_counts:", actual_counts)
    print("final_predictions", final_predictions)
    # Calculate final MSE
    final_mse = mean_squared_error(actual_counts, final_predictions)
    print(f"Final Mean Squared Error (MSE): {final_mse}")

    # Plot actual vs. final predictions
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 6))
    plt.plot(actual_counts, label='Actual', linewidth=3)
    plt.plot(final_predictions, label='Predicted (BiLSTM + XGBoost)', linewidth=3)
    plt.xlabel('Sample Index', fontweight='bold', fontsize=20)
    plt.ylabel('Bandwidth (Kbps)', fontweight='bold', fontsize=20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig('prediction_performance_by_lambda_one_LSTM_XGBoost.eps', format='eps', dpi=1000)

    return final_predictions, final_mse, actual_counts

from keras.models import load_model

def parse_bandwidth_data_with_prediction(output, model, scaler, sequence_length):
    """
    Parse the iPerf bandwidth data and use the BiLSTM model to predict the bandwidth.

    Parameters:
        output (str): iPerf test output.
        model (keras.Model): Pre-trained BiLSTM model.
        scaler (MinMaxScaler): Scaler used for data preprocessing.
        sequence_length (int): The sequence length used in the BiLSTM model.

    Returns:
        list: A list of predicted bandwidth values.
    """
    # Parse raw bandwidth data from iPerf output
    bandwidth_data = parse_bandwidth_data(output)

    # Scale the parsed bandwidth data
    bandwidth_data = np.array(bandwidth_data).reshape(-1, 1)
    scaled_data = scaler.transform(bandwidth_data)

    # Prepare the input sequences for the BiLSTM model
    X = []
    for i in range(len(scaled_data) - sequence_length + 1):
        X.append(scaled_data[i:i + sequence_length])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add the third dimension

    # Predict bandwidth using the BiLSTM model
    predicted_scaled = model.predict(X, verbose=0)

    # Inverse transform the predictions to get actual bandwidth values
    predicted_bandwidth = scaler.inverse_transform(predicted_scaled)

    return predicted_bandwidth.flatten().tolist()
    
#     # max(q_table[next_state].values())
def extract_achieved_bandwidth(iperf_output):
    bandwidths = re.findall(r"(\d+\.?\d*)\s*Mbits/sec", iperf_output)
    
    # If there are no bandwidth values found, return 0 (default)
    if not bandwidths:
        return 0
    
    # Convert the extracted values to floats and calculate the average
    bandwidths = [float(bandwidth) for bandwidth in bandwidths]
    average_bandwidth = sum(bandwidths) / len(bandwidths)
    
    return average_bandwidth
    # """
    # Extract achieved bandwidth from iPerf output using regex.
    # Returns achieved bandwidth in Kbps.
    # """
    
# Reward calculation function
def calculate_reward(target_bandwidth, achieved_bandwidth):
    return -abs(target_bandwidth - achieved_bandwidth)


def update_q_table(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9):
    """
    Update Q-table using the Q-learning formula.
    """
    max_next_q = max(q_table[next_state].values())
    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max_next_q - q_table[state][action])

# Function to map the values
def map_values(value):
    if  value <= 240:
        return 200
    elif 241 <= value <= 250:
        return 250
    elif 251 <= value <= 350:
        return 300
    elif 351 <= value <= 400:
        return 400
    elif 401 <= value <= 420:
        return 420
    elif 421 <= value <= 450:
        return 450
    elif 451 <= value <= 480:
        return 480
    elif 481 <= value <= 490:
        return 490
    elif value >= 491:
        return 500
    else:
        return value

# Shared data for animation
current_bandwidth = 0
current_throughput = 0
time_data = []  # Timestamps for x-axis
bandwidth_data = []  # Processed bandwidth values
throughput_data = []  # Achieved throughput values
lock = threading.Lock()

# Function to process bandwidth and calculate throughput
def process_bandwidths(throughputs_plot_L):
    global current_bandwidth, current_throughput
    for i, bandwidth_value in enumerate(throughputs_plot_L):
        if isinstance(bandwidth_value, (int, float)):
            try:
                # Simulate iPerf output
                time.sleep(1)  # Simulate delay
                achieved_throughput = random.uniform(bandwidth_value * 0.8, bandwidth_value * 1.2)  # Simulate throughput
                with lock:
                    current_bandwidth = bandwidth_value
                    current_throughput = achieved_throughput
                    bandwidth_data.append(bandwidth_value)
                    throughput_data.append(achieved_throughput)
                    time_data.append(i + 1)  # Simulate time (iterations)
            except Exception as e:
                print(f"Error processing bandwidth {bandwidth_value}: {e}")
    return throughput_data


# Function to update the plot
def update_plot(ax1, ax2, timestamps_L, throughputs_plot_L, timestamps_E1, throughputs_plot_E1, differences, average_difference, episode):
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # Plot predicted and actual throughputs
    ax1.plot(timestamps_L, throughputs_plot_L, label='Predicted Throughput (LSTM)', marker='o')
    ax1.plot(timestamps_E1, throughputs_plot_E1, label='Actual Throughput (E1)', marker='x')
    ax1.set_xlabel('Timestamps')
    ax1.set_ylabel('Throughput')
    ax1.set_title(f'Throughput Comparison at Episode {episode}')
    ax1.legend()
    ax1.grid(True)

    # Plot differences
    ax2.plot(timestamps_L, differences, label='Absolute Differences', marker='s', color='red')
    # ax2.axhline(y=average_differences[-1], color='green', linestyle='--', label=f'Average Difference: {average_differences[-1]:.2f}')
    ax2.set_xlabel('Timestamps')
    ax2.set_ylabel('Difference')
    ax2.set_title('Absolute Differences Between Predicted and Actual Throughput')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and draw
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update



def main():
       
    # Load the pre-trained BiLSTM model
    bilstm_model = load_model('user_count_predictor_with_oai_dataset_with_UPF_congestion_newdataset.h5')

    # Initialize the scaler used during training
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([0, 1000]).reshape(-1, 1))  # Example: Scaling bandwidth in Mbps (adjust as needed)

    # Sequence length used in the BiLSTM model
    sequence_length = 9

    combined_data = []  # To hold all test data for combined CSV
    current_time = 1  # Start time at 1 and increase it continuously

    ## Rearrange bandwidths based on Poisson distribution
    # rearranged_bandwidths = rearrange_bandwidths_poisson(bandwidths, poisson_lambda)
    # print(f"Rearranged Bandwidths (based on Poisson distribution): {rearranged_bandwidths}")

    # for bandwidth in rearranged_bandwidths:
    #     print(f"Running {bandwidth}bps test with congestion...")
    #     congestion_thread = threading.Thread(target=generate_congestion_traffic)
    #     congestion_thread.start()
        # output = run_iperf_test(bandwidth)
        # congestion_thread.join()
    results = []
    # bandwidths = [f"{bw}K" for bw in random.choices([250, 350], k=30)]
    bandwidths = [f"{bw}K" for bw in random.choices([500], k=100)]
    # bandwidths = [f"{random.randint(50, 550)}K" for _ in range(30)]
    congestion_log = []  # Stores (bandwidth, applied_rate_kbit)
    poisson_lambda = 1  # Lambda for Poisson distribution
    values = []
    # Load and initialize index
    bandwidth_values = load_bandwidth_values("poisson_distribution_with_drift_value_1000.csv")
    bandwidth_index = 0  # Track which value to use
    print(bandwidth_values)

    for idx, bw in enumerate(bandwidths):
        # for bw in bandwidths:
        bw_str = f"{bw}K" if isinstance(bw, int) else bw
        print(f"[{idx+1}/{len(bandwidths)}] Testing {bw_str} with congestion...") 
        # apply_congestion(bw_str)
      # apply_congestion()

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

        
        applied_cong_bw = bandwidth_values[bandwidth_index - 1] if bandwidth_index > 0 else None
        out = run_iperf_test(bw_str)
        parsed = parse_bandwidth_data(out)
        val = np.mean(parsed) if parsed else 0.0
        print(f"   → With congestion: {val:.2f} Kbps")
        # print("with cong:", val2)
        status = "with congestion"
        results.append((bw_str, val, status, applied_cong_bw))
        # remove_congestion()

    # # ---- Save Output Files ----
    # write_conditional_csv(results, filename="throughput_single_result.csv")

    # Predict bandwidth using the BiLSTM model
    predicted_bandwidth = parse_bandwidth_data_with_prediction(
        out, bilstm_model, scaler, sequence_length
    )

    # Combine time, predicted bandwidth, and actual bandwidth (from iPerf)
    for predicted_bw in predicted_bandwidth:
        combined_data.append([current_time, predicted_bw, bw_str])
        current_time += 1

    # ---- Save Output Files ----
    write_conditional_csv(results, filename="throughput_single_result_9.csv")
    
    write_combined_csv(combined_data, "combined_bandwidth_results_with_predictions.csv")
    print("Combined data saved to 'combined_bandwidth_results_with_predictions.csv'.")
   
    # # Prepare data for BiLSTM
    # # raw_bandwidth_values = [row[1] for row in combined_data]
    # raw_bandwidth_values = [row[2] for row in results]
    # scaled_data, scaler = scale_data(raw_bandwidth_values)
    # time_steps = 5
    # X, y = prepare_data(scaled_data, time_steps)
    # X = X.reshape(X.shape[0], X.shape[1], 1)


    # Step 1: Read Excel/CSV file with measured throughput
    df = pd.read_csv("throughput_single_result_450rage.csv")  # or use pd.read_excel("filename.xlsx")
    
    # Step 2: Extract the column "Measured Throughput (Kbps)"
    throughput_values = df["Measured Throughput (Kbps)"].astype(float).tolist()
    
    # # Step 3: Prepare input for BiLSTM
    # from sklearn.preprocessing import MinMaxScaler
    
    # def scale_data(data):
    #     scaler = MinMaxScaler()
    #     data = np.array(data).reshape(-1, 1)
    #     scaled = scaler.fit_transform(data)
    #     return scaled, scaler
    
    def prepare_data(data, time_steps=3):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)
    
    scaled_data, scaler = scale_data(throughput_values)
    time_steps = 5
    X, y = prepare_data(scaled_data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)



    # Split into training and testing datasets
    split_index = int(len(X) * 0.5)
#    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    print("length of testing data:",len(X_test))
   
    # Predict on the test set
    predicted_counts_scaled = bilstm_model.predict(X_test, verbose=0)
   
    # Predict and compare
    predictions, actual_values = evaluate_model(bilstm_model, X_test, y_test, scaler)

    # Analyze residuals and train XGBoost
    final_predictions, final_mse, actual_counts = analyze_residuals_and_train_xgboost(
        X_test,
        predicted_counts_scaled,
        y_test,
        scaler,
        sequence_length=time_steps
    )
    final_predictions = final_predictions.flatten()
   
    predicted_bandwidths = final_predictions #[250, 350, 600]  # Example predictions
    print (predicted_bandwidths)
    int_list_rounded = [round(x) for x in predicted_bandwidths]
    print(int_list_rounded)
    # Apply the mapping to the list
    mapped_list = [map_values(x) for x in int_list_rounded]
    predicted_ue_states = mapped_list #[user_count_to_ue_state[count] for count in mapped_list]
    print("predicted states",predicted_ue_states)
    print(len(predicted_ue_states))
    
    actual_counts = actual_counts.flatten()
    # Convert to integer list by rounding the values
    int_list_rounded_actual = [round(x) for x in actual_counts]
    actual_ue_states = int_list_rounded_actual# [user_count_to_ue_state[count] for count in actual_user_counts]
       
    # Apply the mapping to the list
    mapped_list = [map_values(x) for x in int_list_rounded_actual]
    actual_ue_states = mapped_list #[user_count_to_ue_state[count] for count in mapped_list]
    print("actual states",actual_ue_states)
    print(len(actual_ue_states))
    states = actual_ue_states
    states = [str(count) for count in states]
    print("actual_states:",states) 
    
    
    states_L = predicted_ue_states
    states_L = [str(count) for count in states_L]
    actions = [ '$a_1$', '$a_2$', '$a_3$', '$a_4$', '$a_5$', '$a_6$', '$a_7$', '$a_8$']
    print("predicted_states:",states_L)
    
    expected_throughput = {
    '500': 500, '400': 400, '450': 450, '480': 480, '350': 350, 
    '300': 300, '200': 200, '490': 490, '420': 420, '250' :250
    }
    
    # Obtained throughput for each action in each state
    obtained_throughput = {
        '500': [497, 490, 340, 180, 162, 155, 370, 300],
        '400': [145, 148, 412, 359, 227, 345, 250, 200],
        '450': [365, 315, 172, 169, 170, 170, 395, 376],
        '480': [489, 475, 210, 46.4, 272, 131.4, 400, 390],
        '350': [131, 114, 72, 122, 105, 43, 146, 156],
        '300': [200, 188, 300, 165, 170, 164, 202, 170],
        '200': [160, 200, 102, 120, 132, 120, 100, 180],
        '250': [160, 200, 156, 140, 132, 250, 100, 198],
        '490': [498, 43, 193, 71.8, 8.4, 25.6, 360, 215],
        '420': [385, 367, 405, 52, 150, 53, 378, 273],
    }
    # Reward calculation (negative of the absolute difference between expected and obtained throughput)
    rewards = {}
    for ue, expected in expected_throughput.items():
        rewards[ue] = [-(abs(expected - ot)**2) for ot in obtained_throughput[ue]]  # Penalize larger differences more

    # Reward calculation (negative of the absolute difference between expected and obtained throughput)
    rewards_L = {}
    for ue, expected in expected_throughput.items():
    #     rewards_L[ue] = [-abs(expected - ot) for ot in obtained_throughput[ue]]
        rewards[ue] = [-(abs(expected - ot)**2) for ot in obtained_throughput[ue]]  # Penalize larger differences more
        # print(rewards[ue])
    
    # Initialize Q-table with zeros
    q_table = np.zeros((len(states), len(actions)))
    
    # Initialize Q-table with zeros
    q_table_L = np.zeros((len(states_L), len(actions)))
    
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    # epsilon = 0.1  # Exploration-exploitation tradeoff
    epsilon_start = 1.0  # Initial epsilon value (start with high exploration)
    epsilon_min = 0.01  # Minimum epsilon value (limit exploration over time)
    epsilon_decay = 0.995  # Decay rate for epsilon
    
    # Function to choose action using epsilon-greedy policy
    def choose_action(state_index, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, len(actions) - 1)  # Explore: Random action
        else:
            return np.argmax(q_table[state_index])  # Exploit: Choose best action
    episodes= 40
    episode_rewards =[]
    episode_rewards_L =[]
#    differnce_qlr = []
    epsilon = epsilon_start
    differnce_qlr = []

    # Q-learning process
    for episode in range(episodes):  # Number of episodes
                  
        total_reward = 0
        total_reward_L = 0
        # differnce_qlr = []
        best_actions = []
        best_actions_L = []
        
        # Loop through the actual states
        for state_index, state in enumerate(states):
            action_index = choose_action(state_index, epsilon)
            reward = rewards[state][action_index]
            total_reward += reward
    
            # Get the next state (in this case, it's stateless so next state is the same)
            next_state_index = state_index
    
            # Update Q-value using the Bellman equation
            old_value = q_table[state_index, action_index]
            next_max = np.max(q_table[next_state_index])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_index, action_index] = new_value
    #         print('Q-table-actual-states:',q_table )
    
            # Find best action for the current state
            best_action_index = np.argmax(q_table[state_index])
            best_action = actions[best_action_index]
            throughput_value = obtained_throughput[state][best_action_index]
            best_actions.append((state_index, state, best_action, throughput_value))
    
        # Loop through the predicted states
        for state_index, state in enumerate(states_L):
            action_index = choose_action(state_index, epsilon)
            reward = rewards[state][action_index]
            total_reward_L += reward
    
            # Get the next state (in this case, it's a stateless problem so next state is the same)
            next_state_index = state_index
    
            # Update Q-value for predicted throughput
            old_value_L = q_table_L[state_index, action_index]
            next_max_L = np.max(q_table_L[next_state_index])
    
            new_value_L = (1 - alpha) * old_value_L + alpha * (reward + gamma * next_max_L)
            q_table_L[state_index, action_index] = new_value_L
    
            # Get the best action for each state after training
            best_action_index_L = np.argmax(q_table_L[state_index])
            best_action_L = actions[best_action_index_L]
            throughput_value_L = obtained_throughput[state][best_action_index_L]
            best_actions_L.append((state_index, state, best_action_L, throughput_value_L))
        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        # Append total reward for this episode
        episode_rewards.append(total_reward)
        episode_rewards_L.append(total_reward_L)
        
        timestamps_L = list(range(len(states_L)))  # Timestamps for LSTM-predicted UEstates
        throughputs_plot_L = [item[3] for item in best_actions_L]
        best_combinations_plot_L = [f"{item[2]}" for item, state in zip(best_actions_L, states_L)]
         # best_combinations_plot_L = [f"{item[2]}({state})" for item, state in zip(best_actions_L, states_L)]
        print(f"Throughput values_predicted at episode-{episode}:",throughputs_plot_L)
        print(f"Bestaction values_predicted at episode-{episode}:",best_combinations_plot_L)
        
        timestamps_E1 = list(range(len(states)))  # Timestamps for Throughput-E1 states
        throughputs_plot_E1 = [item[3] for item in best_actions]
        best_combinations_plot_E1 = [f"{item[2]}" for item, state in zip(best_actions, states)]
         # best_combinations_plot_E1 = [f"{item[2]}({state})" for item, state in zip(best_actions, states)]
        print(f"Throughput values_actual at episode-{episode}:",throughputs_plot_E1)
        print(f"Bestaction values_actual at episode-{episode}:",best_combinations_plot_E1)
        throughput_predicted = throughputs_plot_L
        throughput_actual = throughputs_plot_E1
        # Calculate the absolute differences
        differences = [abs(p - a) for p, a in zip(throughput_predicted, throughput_actual)]
        # Calculate the average of absolute differences
        average_difference = sum(differences) / len(differences)
            
        # Plotting every 300th episode or final episode
        if episode % 1 == 0 or episode == episodes - 1:
            # differnce_qlr = []
            timestamps_L = list(range(len(states_L)))  # Timestamps for LSTM-predicted UEstates
            throughputs_plot_L = [item[3] for item in best_actions_L]
            best_combinations_plot_L = [f"{item[2]}" for item, state in zip(best_actions_L, states_L)]
             # best_combinations_plot_L = [f"{item[2]}({state})" for item, state in zip(best_actions_L, states_L)]
            print(f"Throughput values_predicted at episode-{episode}:",throughputs_plot_L)
            print(f"Bestaction values_predicted at episode-{episode}:",best_combinations_plot_L)
            
            timestamps_E1 = list(range(len(states)))  # Timestamps for Throughput-E1 states
            throughputs_plot_E1 = [item[3] for item in best_actions]
            best_combinations_plot_E1 = [f"{item[2]}" for item, state in zip(best_actions, states)]
             # best_combinations_plot_E1 = [f"{item[2]}({state})" for item, state in zip(best_actions, states)]
            print(f"Throughput values_actual at episode-{episode}:",throughputs_plot_E1)
            print(f"Bestaction values_actual at episode-{episode}:",best_combinations_plot_E1)
            throughput_predicted = throughputs_plot_L
            throughput_actual = throughputs_plot_E1
            # Calculate the absolute differences
            differences = [abs(p - a) for p, a in zip(throughput_predicted, throughput_actual)]
            # Calculate the average of absolute differences
            average_difference = sum(differences) / len(differences)
            
            plt.ion()  # Turn on interactive mode
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Function to update the plot
            def update_plot(episode):
                # Clear previous plots
                ax1.clear()
                ax2.clear()
            
                # Plot predicted and actual throughputs
                ax1.plot(timestamps_L, throughputs_plot_L, label='Predicted Throughput (LSTM)', marker='o')
                ax1.plot(timestamps_E1, throughputs_plot_E1, label='Actual Throughput (E1)', marker='x')
                ax1.set_xlabel('Timestamps')
                ax1.set_ylabel('Throughput')
                ax1.set_title(f'Throughput Comparison at Episode {episode}')
                ax1.legend()
                ax1.grid(True)
                
                # Plot differences
                ax2.plot(timestamps_L, differences, label='Absolute Differences', marker='s', color='red')
                # ax2.axhline(y=average_difference[-1], color='green', linestyle='--', label=f'Average Difference: {average_difference[-1]:.2f}')
                ax2.set_xlabel('Timestamps')
                ax2.set_ylabel('Difference')
                ax2.set_title('Absolute Differences Between Predicted and Actual Throughput')
                ax2.legend()
                ax2.grid(True)

                
                # Adjust layout and draw
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)  # Pause to allow the plot to update          
            
            print(f"Absolute differences at episode-{episode}:", differences)
            print(f"Average absolute difference at episode-{episode}:", average_difference)
    
            # Specify the filename
            filename_p = f"best_action_throughput_data_predicted_episode_{episode}.csv"
            filename_a = f"best_action_throughput_data_actual_episode_{episode}.csv"
            
            # Prepare data for CSV file
            data_p = []
            data_a = []
            for i in range(len(timestamps_L)):
                data_p.append([timestamps_L[i], best_combinations_plot_L[i], throughputs_plot_L[i], best_combinations_plot_E1[i], throughputs_plot_E1[i]])
            
            # Write to CSV
            with open(filename_p, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(["Time Step (UE State)", "Best Action-LSTM-predicted", "Throughput-LSTM-predicted", "Best Action-E1", "Throughput-E1"])
                # Write data rows
                writer.writerows(data_p)
            
            print(f"Data saved to {filename_p}")
            
        differnce_qlr.append(average_difference)
        print(differnce_qlr) 
        
        # Initialize the plot
            
        update_plot(episode)

    #===============integrate ZTN with IBN=======================
   
    # # Set global font sizes
    # plt.rcParams.update({
    #     'font.size': 13,          # Base font size
    #     'axes.titlesize': 16,     # Title font
    #     'axes.labelsize': 14,     # Axis label font
    #     'xtick.labelsize': 12,    # X tick labels
    #     'ytick.labelsize': 12,    # Y tick labels
    #     'legend.fontsize': 12     # Legend font
    # })
    
    # # ====== Step 1: Define Nile Intents ======
    # nile_intent_in = """
    # define intent buildIntent:
    # for group('professors')
    # from endpoint('gateway')
    # to endpoint('network')
    # start hour('09:00')
    # end hour('12:00')
    # set bandwidth('max', '300', 'kbps')
    # """
    
    # nile_intent_out = """
    # define intent buildIntent:
    # for group('professors')
    # from endpoint('gateway')
    # to endpoint('network')
    # start hour('09:00')
    # end hour('12:00')
    # set bandwidth('max', '450', 'kbps')
    # """
    
    # def extract_bandwidth_from_nile(intent_str):
    #     match = re.search(r"set bandwidth\('max', '(\d+)', 'kbps'\)", intent_str)
    #     return int(match.group(1)) if match else 300
    
    # thresholds = {
    #     "In-Distribution (300 Kbps)": extract_bandwidth_from_nile(nile_intent_in),
    #     "Out-of-Distribution (450 Kbps)": extract_bandwidth_from_nile(nile_intent_out)
    # }
    
    # # ====== Step 2: Function to process one CSV file ======
    # def process_csv(filename, label_suffix):
    #     df = pd.read_csv(filename)
    #     df['Time Step (UE State)'] = pd.to_numeric(df['Time Step (UE State)'], errors='coerce')
    #     df['Throughput-LSTM-predicted'] = pd.to_numeric(df['Throughput-LSTM-predicted'], errors='coerce')
    #     df['Group'] = (df['Time Step (UE State)'] - 1) // 1
    #     grouped = df.groupby('Group')[['Throughput-LSTM-predicted']].mean().reset_index()
    #     grouped.columns = ['Group', f'Avg_Throughput-LSTM_{label_suffix}']
    #     return grouped

    # # # Drop rows with NaNs in critical columns
    # # df = df.dropna(subset=['Time Step (UE State)', 'Throughput-LSTM-predicted'])

    # # # Rename the throughput column with suffix for comparison
    # # df = df[['Time Step (UE State)', 'Throughput-LSTM-predicted']].copy()
    # # df.columns = ['Time Step', f'Throughput_LSTM_{label_suffix}']

    # # return df


    # # ====== Step 3: Load both episodes ======
    # g1 = process_csv('best_action_throughput_data_predicted_episode_15.csv', 'Ep15')
    # g2 = process_csv('best_action_throughput_data_predicted_episode_39.csv', 'Ep39')
    # grouped_avg = pd.merge(g1, g2, on='Group', how='outer').sort_values(by='Group').reset_index(drop=True)
    
    # # ====== Step 4: Setup for match analysis ======
    # match_stats = []
    
    # # ====== Step 5: Plot individual throughput lines ======
    # fig, axes = plt.subplots(figsize=(10, 10))
    # # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # fig.suptitle("Throughput Comparison Across Nile Intent Thresholds", fontsize=14)
    
    # for ax, (label, bw_threshold) in zip(axes, thresholds.items()):
    #     df = grouped_avg.copy()
        
    #     df['Ep15_LSTM_Exceed'] = df['Avg_Throughput-LSTM_Ep15'] >= bw_threshold
    #     df['Ep39_LSTM_Exceed'] = df['Avg_Throughput-LSTM_Ep39'] >= bw_threshold
    
    #     # Count matches
    #     total_groups = len(df)
    #     match15 = df['Ep15_LSTM_Exceed'].sum()
    #     match39 = df['Ep39_LSTM_Exceed'].sum()
    
    #     percent15 = (match15 / total_groups) * 100
    #     percent39 = (match39 / total_groups) * 100
    
    #     print(f"\n=== {label} | Threshold = {bw_threshold} Kbps ===")
    #     print(df.to_string(index=False))
    #     print(f"[Episode 15] Sub-Optimal matched in {match15}/{total_groups} groups = {percent15:.2f}%")
    #     print(f"[Episode 39] Optimal matched in {match39}/{total_groups} groups = {percent39:.2f}%")
    
    #     match_stats.append({
    #         "Intent": label,
    #         "Ep15_Match": percent15,
    #         "Ep15_NoMatch": 100 - percent15,
    #         "Ep39_Match": percent39,
    #         "Ep39_NoMatch": 100 - percent39
    #     })

    #             # Save this threshold evaluation's dataframe if needed
    #     df_filename = f"Throughput_Group_Comparison_{label.replace(' ', '_')}.csv"
    #     df.to_csv(df_filename, index=False)

    #             # Plot individual figure
    #     # fig, ax = plt.subplots(figsize=(10, 5))

    #     # Plot line chart for throughput
    #     group_x = df['Group'].to_numpy()
    #     ax.plot(group_x, df['Avg_Throughput-LSTM_Ep15'].to_numpy(), color='green', marker='o', label='Sub-Optimal')
    #     ax.plot(group_x, df['Avg_Throughput-LSTM_Ep39'].to_numpy(), color='purple', marker='s', label='Optimal')
    #     ax.axhline(y=bw_threshold, color='red', linestyle='--', label=f'Threshold = {bw_threshold} Kbps')

    #     ax.set_title(f"{label} (Threshold = {bw_threshold} Kbps)", fontsize=13)
    #     # ax.set_title(label)
    #     ax.set_xlabel('Group Number (Each = 5 Time Steps)')
    #     ax.set_ylabel('Avg Throughput (Kbps)')
    #     ax.grid(True)
    #     ax.legend()
    
    # # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.tight_layout()
    # fig.savefig(f"{label.replace(' ', '_')}_Comparison.png")
    # fig.savefig(f"{label.replace(' ', '_')}_Comparison.eps", format='eps')

    # #plt.savefig('line_comparison.png')
    # plt.savefig("Bandwidth_Comparison.eps", format='eps')
    # plt.show()
    
    # # Save all match percentage stats
    # match_stats_df = pd.DataFrame(match_stats)
    # match_stats_df.to_csv("match_statistics.csv", index=False)
    # print("\nMatch statistics saved to 'match_statistics.csv'")
    # import re
    # import pandas as pd
    # import matplotlib.pyplot as plt
    
    # === Set global font sizes ===
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # === Step 1: Define Nile Intents ===
    nile_intent_in = """
    define intent buildIntent:
    for group('professors')
    from endpoint('gateway')
    to endpoint('network')
    start hour('09:00')
    end hour('12:00')
    set bandwidth('max', '300', 'kbps')
    """

    nile_intent_out = """
    define intent buildIntent:
    for group('professors')
    from endpoint('gateway')
    to endpoint('network')
    start hour('09:00')
    end hour('12:00')
    set bandwidth('max', '450', 'kbps')
    """
    
    def extract_bandwidth_from_nile(intent_str):
        match = re.search(r"set bandwidth\('max', '(\d+)', 'kbps'\)", intent_str)
        return int(match.group(1)) if match else 300
    
    thresholds = {
        "In-Distribution (300 Kbps)": extract_bandwidth_from_nile(nile_intent_in),
        "Out-of-Distribution (450 Kbps)": extract_bandwidth_from_nile(nile_intent_out)
    }
    
    # === Step 2: Process CSV ===
    def process_csv(filename, label_suffix):
        df = pd.read_csv(filename)
        df['Time Step (UE State)'] = pd.to_numeric(df['Time Step (UE State)'], errors='coerce')
        df['Throughput-LSTM-predicted'] = pd.to_numeric(df['Throughput-LSTM-predicted'], errors='coerce')
        df['Group'] = (df['Time Step (UE State)'] - 1) // 1
        grouped = df.groupby('Group')[['Throughput-LSTM-predicted']].mean().reset_index()
        grouped.columns = ['Group', f'Avg_Throughput-LSTM_{label_suffix}']
        return grouped
    
    # === Step 3: Load both episodes ===
    g1 = process_csv('best_action_throughput_data_predicted_episode_15.csv', 'Ep15')
    g2 = process_csv('best_action_throughput_data_predicted_episode_39.csv', 'Ep39')
    grouped_avg = pd.merge(g1, g2, on='Group', how='outer').sort_values(by='Group').reset_index(drop=True)
    
    # === Step 4: Match Analysis and Plotting ===
    match_stats = []
    
    for label, bw_threshold in thresholds.items():
        df = grouped_avg.copy()
    
        df['Ep15_LSTM_Exceed'] = df['Avg_Throughput-LSTM_Ep15'] >= bw_threshold
        df['Ep39_LSTM_Exceed'] = df['Avg_Throughput-LSTM_Ep39'] >= bw_threshold
    
        total_groups = len(df)
        match15 = df['Ep15_LSTM_Exceed'].sum()
        match39 = df['Ep39_LSTM_Exceed'].sum()
        percent15 = (match15 / total_groups) * 100
        percent39 = (match39 / total_groups) * 100
    
        print(f"\n=== {label} | Threshold = {bw_threshold} Kbps ===")
        print(df.to_string(index=False))
        print(f"[Episode 15] Sub-Optimal matched in {match15}/{total_groups} groups = {percent15:.2f}%")
        print(f"[Episode 39] Optimal matched in {match39}/{total_groups} groups = {percent39:.2f}%")
    
        match_stats.append({
            "Intent": label,
            "Ep15_Match": percent15,
            "Ep15_NoMatch": 100 - percent15,
            "Ep39_Match": percent39,
            "Ep39_NoMatch": 100 - percent39
        })
    
        # === Save group-wise data for this threshold ===
        df_filename = f"Throughput_Group_Comparison_{label.replace(' ', '_')}.csv"
        df.to_csv(df_filename, index=False)
    
        # === Plot and Save Individual Figure ===
        fig, ax = plt.subplots(figsize=(10, 5))
        group_x = df['Group'].to_numpy()
    
        ax.plot(group_x, df['Avg_Throughput-LSTM_Ep15'].to_numpy(), color='green', marker='o', label='Sub-Optimal')
        ax.plot(group_x, df['Avg_Throughput-LSTM_Ep39'].to_numpy(), color='purple', marker='s', label='Optimal')
        ax.axhline(y=bw_threshold, color='red', linestyle='--', label=f'Threshold = {bw_threshold} Kbps')

        
        distribution_type = label.split('(')[0].strip()
        ax.set_title(f"{distribution_type} ({bw_threshold} Kbps)", fontsize=13, fontweight='bold')

        # ax.set_title(f"{label} (Threshold = {bw_threshold} Kbps)")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Throughput (Kbps)')
        ax.grid(True)
        ax.legend()
    
        fig.tight_layout()
        fig.savefig(f"{label.replace(' ', '_')}_Comparison.png")
        fig.savefig(f"{label.replace(' ', '_')}_Comparison.eps", format='eps')
        plt.show()
    
    # === Save Match Summary ===
    match_stats_df = pd.DataFrame(match_stats)
    match_stats_df.to_csv("match_statistics.csv", index=False)
    print("\nMatch statistics saved to 'match_statistics.csv'")

    
    # # ====== Step 6: Plot matching percentages ======
    # labels = []
    # ep15_match = []
    # ep15_nomatch = []
    # ep39_match = []
    # ep39_nomatch = []
    
    # for stat in match_stats:
    #     labels.append(stat['Intent'])
    #     ep15_match.append(stat['Ep15_Match'])
    #     ep15_nomatch.append(stat['Ep15_NoMatch'])
    #     ep39_match.append(stat['Ep39_Match'])
    #     ep39_nomatch.append(stat['Ep39_NoMatch'])
    
    # x = np.arange(len(labels))
    # width = 0.35
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Line plots
    
    # ax.plot(x, ep39_match, marker='s', linestyle='-', color='purple', label='Optimal Best Case (In_Distribution)')
    # ax.plot(x, ep39_nomatch, marker='d', linestyle='--', color='violet', label='Optimal Worst Case (Out_of_Distribution)')
    
    # ax.plot(x, ep15_match, marker='o', linestyle='-', color='green', label='Sub-Optimal Best Case (In_Distribution)')
    # ax.plot(x, ep15_nomatch, marker='x', linestyle='--', color='lightgreen', label='Sub-Optimal Worst Case (Out_of_Distribution)')
    
    
    # # Labeling
    # ax.set_ylabel('Percentage (%)')
    # ax.set_title('Best and Worst Case with Bandwidth Thresholds')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=15)
    # ax.set_ylim(0, 100)
    # ax.grid(True)
    # ax.legend()
    
    # # Annotate points with % values
    # for i in range(len(labels)):
    #     ax.text(x[i], ep15_match[i] + 1, f"{ep15_match[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='green')
    #     ax.text(x[i], ep39_match[i] + 1, f"{ep39_match[i]:.1f}%", ha='center', va='bottom', fontsize=8, color='purple')
    
    # plt.tight_layout()
    # #plt.savefig("match_line_plot.png")
    # plt.savefig("best_worst.eps", format='eps')
    # # plt.savefig("uniform_t.eps", format='eps')
    # plt.show()
 
    # ====== End Of Integration Part ==========
 
    # # Keep the plot window open
    plt.ioff()
    plt.show() 
    # plt.close()
    epi=[]
    for i in range(0, 40, 1):  # range(start, stop, step)
        epi.append(i)
    print(epi)
    l1 = epi[::5]
    print(l1)
    l2 = differnce_qlr[::5]
            # l2 = differnce_qlr
    print(l2)
    # Plot reward vs episodes to check convergence
    plt.figure(figsize=(10, 5))
    # plt.plot(episodes[1:2000], episode_rewards[1:2000], label='Total Reward per Episode')
    plt.bar(l1, l2, color='blue', width=8)
    plt.xlabel('Episodes')
    plt.ylabel('MAE')
    plt.title('MAE vs Episodes')
    plt.xticks(l1)  # Ensure x-axis ticks align with the bar positions
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("qlearning_convergence_MAE.png",dpi=600)
            
# differnce_qlr.append(average_difference)
    print(differnce_qlr)    
# Plot reward vs episodes to check convergence
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), episode_rewards, label='Total Reward per Episode-actual')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Convergence: Reward vs Episodes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=2.0)
    plt.savefig("qlearning_reward_convergence_actual_vary_epsi.png", dpi=300)
    
    print(throughputs_plot_L)
    print(f"Predicted Bandwidth: {actual_ue_states}Kbps, State: {predicted_ue_states}, Action: {throughputs_plot_L}")
            
# Debug and validate throughputs_plot_L
    if throughputs_plot_L is None:
        raise ValueError("throughputs_plot_L is not defined")
    if not isinstance(throughputs_plot_L, list):
        raise ValueError("throughputs_plot_L is not a valid list")
    if not throughputs_plot_L:
        raise ValueError("throughputs_plot_L is empty")
        
# Process each bandwidth in the list
    for bandwidth_value in throughputs_plot_L:
        if isinstance(bandwidth_value, (int, float)) or (isinstance(bandwidth_value, str) and bandwidth_value.isdigit()):
            bandwidth_str = f"{bandwidth_value}K"
            try:
                iperf_output = run_iperf_test(bandwidth_str)
                achieved_bandwidth = extract_achieved_bandwidth(iperf_output)
                print(f"Bandwidth Value: {bandwidth_str} -> Achieved Bandwidth: {achieved_bandwidth}")
            except Exception as e:
                print(f"Error running iPerf for bandwidth {bandwidth_str}: {e}")
        else:
            print(f"Skipping invalid bandwidth value: {bandwidth_value}")

if __name__ == "__main__":
    main()




#In this code, whatever graphs generated in the main function, which is already saved. but i want to see the live plot also for each episode as well as for each action also. Kindly debugg the code generate the same.
