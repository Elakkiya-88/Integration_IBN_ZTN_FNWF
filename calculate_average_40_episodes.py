import pandas as pd
import numpy as np

# Configuration
num_episodes = 40
input_filename_pattern = "best_action_throughput_data_predicted_episode_{}.csv"
throughput_column = "Throughput-LSTM-predicted"  # Change if your column name differs
output_csv = "average_throughput_all_episodes.csv"

# Store per-episode average
episode_averages = []

# Step 1: Compute each episode's average
for i in range(num_episodes):
    try:
        file_name = input_filename_pattern.format(i)
        df = pd.read_csv(file_name)
        
        avg = df[throughput_column].mean()
        episode_averages.append({'episode': i, 'average_throughput': avg})
    
    except Exception as e:
        print(f"Error in episode {i}: {e}")
        episode_averages.append({'episode': i, 'average_throughput': np.nan})  # Preserve index

# Convert to DataFrame
avg_df = pd.DataFrame(episode_averages)

# Step 2: Compute overall average
overall_avg = avg_df['average_throughput'].mean()
avg_df.loc[len(avg_df.index)] = {'episode': 'overall_average', 'average_throughput': overall_avg}

# Step 3: Save results
avg_df.to_csv(output_csv, index=False)
print(f"✅ Saved to {output_csv}")
