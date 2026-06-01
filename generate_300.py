from synthetic_dataset import SyntheticRVDataset
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Generate 300 synthetic RV systems
# ---------------------------------------------------
ds = SyntheticRVDataset(n_samples=300)
print("Generated dataset with", len(ds), "samples")

rows = []
systems = []

for i in range(len(ds)):
    x, lsp, theta, info = ds[i]
    systems.append((x, info))

    rows.append({
        "system_id": i,

        "P_days": info["P"], # Orbital period
        "K_ms": info["K"], # RV amplitude 
        "eccentricity": info["e"], # Eccentricity
        "omega_deg": info["omega_deg"], # Argument of periapsis

        "n_obs": info["n_obs"], # Num of observations 
        "baseline_days": info["baseline_d"], # Baseline days
        "snr_meas": info["snr_meas"], # Signal noise ratio
        "rv_std_ms": info["rv_std_ms"],# RV standard deviation

        "n_planets": info["n_planets"],
        "n_companions": info["n_companions"],
        "has_ecc": info["has_ecc"],
        "valid": info["valid"]
    })
df = pd.DataFrame(rows)
output_file = "data/synthetic_300.csv"
df.to_csv(output_file, index=False)


# ---------------------------------------------------
# BEST AND WORST CANDIDATES
# ---------------------------------------------------
best5 = df.sort_values(by="snr_meas", ascending=False).head(5) #highest SNR
worst5 = df.sort_values(by="snr_meas", ascending=True).head(5) #lowest SNR

print("Best 5 candidates (highest SNR):")
print(best5[["system_id", "P_days", "K_ms", "snr_meas"]])

print("Worst 5 candidates (lowest SNR):")
print(worst5[["system_id", "P_days", "K_ms", "snr_meas"]])


# ---------------------------------------------------
# PLOT BEST CANDIDATE
# ---------------------------------------------------
best_id = int(best5.iloc[0]["system_id"])

x_best, info_best = systems[best_id]

t_best = x_best[0]
rv_best = x_best[1]
mask_best = x_best[3] > 0

plt.figure(figsize=(8,4))
plt.scatter(t_best[mask_best], rv_best[mask_best])

plt.xlabel("T (normalised)")
plt.ylabel("RV (normalised)")
plt.title(
    f"BEST CANDIDATE | "
    f"SNR={info_best['snr_meas']:.2f}, "
    f"P={info_best['P']:.1f} d"
)
plt.show()

# ---------------------------------------------------
# PLOT WORST CANDIDATE
# ---------------------------------------------------
worst_id = int(worst5.iloc[0]["system_id"])

x_worst, info_worst = systems[worst_id]

t_worst = x_worst[0]
rv_worst = x_worst[1]
mask_worst = x_worst[3] > 0

plt.figure(figsize=(8,4))
plt.scatter(t_worst[mask_worst], rv_worst[mask_worst])

plt.xlabel("T (normalised)")
plt.ylabel("RV (normalised)")
plt.title(
    f"WORST CANDIDATE| "
    f"SNR={info_worst['snr_meas']:.2f}, "
    f"P={info_worst['P']:.1f} d"
)
plt.show()


# ---------------------------------------------------
# HISTOGRAMS
# ---------------------------------------------------

# ---------------------------------------------------
# RV amplitude HISTOGRAM
# ---------------------------------------------------
plt.figure(figsize=(8,4))
plt.hist(df["K_ms"], bins=30)

plt.xlabel("K (m/s)")
plt.ylabel("Count")
plt.title("Synthetic RV Amplitude Histogram")

plt.show()


# ---------------------------------------------------
# Orbital Period HISTOGRAM
# ---------------------------------------------------
plt.figure(figsize=(8,4))
plt.hist(df["P_days"], bins=30)

plt.xlabel("P (days)")
plt.ylabel("Count")
plt.title("Synthetic Orbital Period Histogram")

plt.show()


# ---------------------------------------------------
# Eccentricity HISTOGRAM
# ---------------------------------------------------
plt.figure(figsize=(8,4))
plt.hist(df["eccentricity"], bins=30)

plt.xlabel("Eccentricity")
plt.ylabel("Count")
plt.title("Synthetic Eccentricity Histogram")

plt.show()


# ---------------------------------------------------
# SNR HISTOGRAM
# ---------------------------------------------------
plt.figure(figsize=(8,4))
plt.hist(df["snr_meas"], bins=30)

plt.xlabel("SNR")
plt.ylabel("Count")
plt.title("Synthetic SNR Histogram")

plt.show()