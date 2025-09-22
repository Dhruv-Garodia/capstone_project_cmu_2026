import numpy as np
import matplotlib.pyplot as plt

# --- Target pore size stats ---
mean_nm = 104.0
std_nm = 55.8

# Convert mean/std -> log-normal parameters
sigma = np.sqrt(np.log(1 + (std_nm**2 / mean_nm**2)))
mu = np.log(mean_nm) - 0.5 * sigma**2
print(f"log-normal params: mu={mu:.3f}, sigma={sigma:.3f}")

# --- Continuous log-normal distribution ---
x = np.linspace(20, 500, 500)
pdf = (1/(x*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
pdf /= np.trapz(pdf, x)  # normalize area to 1

# --- Discretized bins from config ---
bins_nm = np.array([35.6, 47.7, 63.9, 85.5, 114.3, 152.9, 204.7, 274.2, 367.2, 491.8, 658.7])
weights = np.array([0.027, 0.061, 0.122, 0.174, 0.212, 0.181, 0.121, 0.065, 0.027, 0.009, 0.001])

weights1 = weights/weights.sum()

# Compute weighted mean and std
mean_discrete = np.sum(bins_nm * weights1)
var_discrete = np.sum(weights1 * (bins_nm - mean_discrete)**2)
std_discrete = np.sqrt(var_discrete)

print("mean_discrete", mean_discrete)
print("std_discrete", std_discrete)

# Plot
plt.figure(figsize=(8,5))
plt.bar(bins_nm, weights, width=20, align="center", alpha=0.6, edgecolor="k", label="Discrete bins")
plt.xlabel("Pore diameter (nm)")
plt.ylabel("Probability")
plt.title("Pore Size Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
