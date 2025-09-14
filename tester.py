import numpy as np

# Known data points
known_times = np.array([0, 10, 20, 30])  # e.g., seconds
known_values = np.array([5, 12, 18, 25]) # e.g., temperature

# Desired time points for interpolation
interp_times = np.array([5, 15, 25])

# Perform linear interpolation
interpolated_values = np.interp(interp_times, known_times, known_values)

print(f"Known times: {known_times}")
print(f"Known values: {known_values}")
print(f"Interpolation times: {interp_times}")
print(f"Interpolated values: {interpolated_values}")
