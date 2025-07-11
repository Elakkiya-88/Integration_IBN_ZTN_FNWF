import numpy as np
import random
import csv
import matplotlib.pyplot as plt

# Parameters
center = 300
lam = 2
min_val = 250
max_val = 350
num_samples = 300

# Initialization
values = []
timeslots = []
timeslot = 0
current_value = center
drift_flags = []
drift_amounts = []

# Probability of initiating a drift phase
drift_chance = 0.05
drift_steps_remaining = 0
drift_step_value = 0

drift_step_values = []  # store individual steps when drift is ongoing

while len(values) < num_samples:
    if drift_step_values:
        # Apply next step in drift
        step = drift_step_values.pop(0)
        current_value += step
        drift_flags.append(True)
        drift_amounts.append(step)
    elif random.random() < drift_chance:
        total_drift = random.randint(-50, -1)  # e.g., -37
        steps = random.randint(2, 10)          # e.g., 4 steps

        # Split total_drift into random parts summing to total_drift
        random_parts = np.random.dirichlet(np.ones(steps))  # [0.3, 0.2, 0.1, 0.4]
        drift_step_values = [int(round(p * total_drift)) for p in random_parts]

        # Adjust to make sure total exactly equals total_drift
        correction = total_drift - sum(drift_step_values)
        drift_step_values[0] += correction

        
        
        
        # # Start a new drift phase
        # total_drift = random.randint(-50, 0)  # e.g., -37
        # steps = random.randint(2, 5)          # e.g., 4 steps

        # # Split total_drift into random parts summing to total_drift
        # random_parts = np.random.dirichlet(np.ones(steps))  # [0.3, 0.2, 0.1, 0.4]
        # drift_step_values = [int(round(p * total_drift)) for p in random_parts]

        # # Adjust to make sure total exactly equals total_drift
        # correction = total_drift - sum(drift_step_values)
        # drift_step_values[0] += correction

        # Apply first step now
        step = drift_step_values.pop(0)
        current_value += step
        drift_flags.append(True)
        drift_amounts.append(step)
    else:
        # Normal Poisson fluctuation
        offset = np.random.poisson(lam)
        direction = random.choice([-10, 10])
        current_value += direction * offset
        drift_flags.append(False)
        drift_amounts.append(0)

    # Clamp value
    current_value = max(min_val, min(max_val, current_value))

    # Save
    values.append(current_value)
    timeslots.append(timeslot)
    timeslot += 1


# while len(values) < num_samples:
#     if drift_steps_remaining > 0:
#         # Apply part of a multi-step drift
#         current_value += drift_step_value
#         drift_steps_remaining -= 1
#         drift_flags.append(True)
#         drift_amounts.append(drift_step_value)
#     elif random.random() < drift_chance:
#         # Start a new drift phase
#         total_drift = random.randint(-50, 0)
#         steps = random.randint(2, 5)
#         drift_step_value = total_drift // steps
#         drift_steps_remaining = steps - 1  # One step applied now

#         current_value += drift_step_value
#         drift_flags.append(True)
#         drift_amounts.append(drift_step_value)
#     else:
#         # Normal Poisson-based variation
#         offset = np.random.poisson(lam)
#         direction = random.choice([-10, 10])
#         current_value += direction * offset
#         drift_flags.append(False)
#         drift_amounts.append(0)

#     # Clamp within range
#     current_value = max(min_val, min(max_val, current_value))

#     # Store
#     values.append(current_value)
#     timeslots.append(timeslot)
#     timeslot += 1
# Save to CSV with Drift and DriftAmount columns
with open("poisson_distribution_with_drift_value_300.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["TimeSlot", "Value", "Drift", "DriftAmount"])
    for t, val, drift, amount in zip(timeslots, values, drift_flags, drift_amounts):
        writer.writerow([t, val, "Yes" if drift else "No", amount if amount is not None else ""])


print("CSV file 'poisson_distribution.csv' generated successfully.")
# Plot the base line
plt.figure(figsize=(12, 6))
plt.plot(timeslots, values, marker='o',label='Value over Time', color='blue')

# Highlight drift points (sudden -50)
for t, val, is_drift in zip(timeslots, values, drift_flags):
    if is_drift:
        plt.scatter(t, val, color='red', s=60, zorder=5, label='Drift Point (-50)' if t == timeslots[drift_flags.index(True)] else "")

# Add reference line
plt.axhline(y=center, color='green', linestyle='--', linewidth=1.5, label='Base Value (300)')

# Final styling
plt.title(f'Poisson-Based Values with Drift Events Highlighted (λ={lam}, Centered at {center})')
plt.xlabel("TimeSlot")
plt.ylabel("BW")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dataset_distribution_with_drifts.png")
plt.show()
# Optional: Save to CSV
with open("drifted_poisson.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timeslot", "Value", "Drift", "Drift_Amount"])
    for i in range(len(values)):
        writer.writerow([timeslots[i], values[i], "Yes" if drift_flags[i] else "No", drift_amounts[i]])

print("CSV generation complete.")

# Plot the distribution over time
plt.figure(figsize=(12, 6))
plt.plot(timeslots, values, marker='o', linestyle='-', color='blue')
plt.title(f'Poisson-Based Evolving Values (λ={lam}, Centered at {center})')
plt.xlabel("TimeSlot")
plt.ylabel("BW")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("dataset_distribution_for_ztn_ibn_integ.png")

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(values, bins=range(min_val, max_val + 1), color='skyblue', edgecolor='black')
plt.title(f'Histogram of Poisson-Distributed Values (λ={lam}, Centered at 300)')
plt.xlabel("BW")
plt.ylabel("Frequency of occurance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dataset_for_ztn_ibn_integ.png")
plt.show()
