
######################ID case###########################################3

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # Constants
I_user = 300
x_t = 250
episodes = 40


# def objective(x, episode):
#     base = -np.abs(I_user - x)
    
#     # Exponential decay (e.g., from ~10 to ~0.5 over 40 episodes)
#     decay = 2 * np.exp(-0.1 * episode)
    
#     ripple = decay * np.sin(0.05 * x)
#     noise = -decay  # deterministic negative "noise" that decays exponentially

#     return base + ripple + noise

# # Evaluate objective over episodes
# episode_indices = np.arange(1, episodes + 1)
# scores = [objective(x_t, ep) for ep in episode_indices]

# # Save to CSV
# df = pd.DataFrame({
#     'Episode': episode_indices,
#     'Bandwidth_x_t': [x_t] * episodes,
#     'Objective_Value_f(x_t)': scores
# })
# df.to_csv('monte_carlo_single_point_250_updated_2.csv', index=False)
# print("✅ CSV file 'monte_carlo_single_point.csv' saved.")

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('/kaggle/input/best-action-file/average_throughput_all_episode_updated_threevalues.csv')  # Replace with your actual file name
df1 = pd.read_csv('/kaggle/input/best-action-file/monte_carlo_single_point-updated_threevalues_350.csv')
# Plot
# plt.figure(figsize=(10, 5))
# plt.plot(df['episode'], df['function'], marker='o', linestyle='-', color='red')
# plt.xlabel('Episode')
# plt.ylabel('function')
# plt.title('Plot of c vs a')
# plt.grid(True)
# plt.show()
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# Update global font and label styles
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df1['Episode'], df1['avg_3_value'], marker='x', linestyle='-', color='blue', label="Monte Carlo Simulation")
plt.plot(df['episode'], df['avg_3_value'], marker='o', linestyle='-', color='red', label="Test-bed Performance")

plt.xlabel('Episode')
plt.ylabel('Objective Value $f(x_t)$')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save as EPS with high resolution
plt.savefig("Monte-carlo-test-bed-compa-ID.eps", format="eps", dpi=300)
plt.show()

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(df1['Episode'], df1['avg_3_value'], marker='x', linestyle='-', color='blue', label="Monte Carlo Simulation")
# plt.plot(df['episode'], df['avg_3_value'], marker='o', linestyle='-', color='red', label = "Test-bed Performance")
# # plt.axhline(y=np.mean(scores), color='green', linestyle='--', label=f'Avg = {np.mean(scores):.2f}')
# # plt.title(f'For Fixed Bandwidth $x_t = {x_t}$ (I_user = {I_user})')
# plt.xlabel('Episode')
# plt.ylabel('Objective Value $f(x_t)$')
# # plt.ylabel(r'$f(x_t) = -\left| \beta_{target}} - B_t \right|$')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# # plt.savefig("Monte-carlo-single-point_comparision_updated.png")
# plt.savefig("Monte-carlo-test-bed-compa-ID.eps", format="eps", dpi=300)
# plt.show()


###########################OOOD####################################
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('/kaggle/input/best-action-file/average_throughput_all_episode_updated_threevalues.csv')  # Replace with your actual file name
df1 = pd.read_csv('/kaggle/input/best-action-file/monte_carlo_single_point-updated_threevalues_350.csv')
# Plot
# plt.figure(figsize=(10, 5))
# plt.plot(df['episode'], df['function'], marker='o', linestyle='-', color='red')
# plt.xlabel('Episode')
# plt.ylabel('function')
# plt.title('Plot of c vs a')
# plt.grid(True)
# plt.show()
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df1['Episode'], df1['Avg_6'], marker='x', linestyle='-', color='blue', label="Monte Carlo Simulation")
plt.plot(df['episode'], df['Avg_6'], marker='o', linestyle='-', color='red', label = "Test-bed Performance")
# plt.axhline(y=np.mean(scores), color='green', linestyle='--', label=f'Avg = {np.mean(scores):.2f}')
# plt.title(f'For Fixed Bandwidth $x_t = {x_t}$ (I_user = {I_user})')
plt.xlabel('Episode')
plt.ylabel('Objective Value $f(B_t)$')
# plt.ylabel(r'$f(x_t) = -\left| \beta_{target}} - B_t \right|$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Monte-carlo-test-bed-compa-OOD.eps", format="eps", dpi=300)
plt.show()
