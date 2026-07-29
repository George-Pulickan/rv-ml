import csv
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from pendulum_bayesian import infer_bayesian_parameters

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clean_pendulum_trajectory(T, dt, g, l, theta_0):
    """Integrate the deterministic pendulum dynamics without process noise."""
    if T <= 0 or dt <= 0 or l <= 0:
        raise ValueError("T, dt, and l must be positive.")

    t = np.arange(0.0, T + dt, dt)
    if t[-1] > T:
        t = t[t <= T]
    if t.size == 0:
        raise ValueError("The time grid is empty. Choose a larger T or smaller dt.")

    theta = np.empty_like(t, dtype=float)
    theta_state = float(theta_0)
    omega = 0.0

    for i, _ in enumerate(t):
        if i == 0:
            theta[i] = theta_state
            continue

        acceleration = -(g / l) * np.sin(theta_state)
        theta_next = theta_state + omega * dt + 0.5 * acceleration * dt * dt
        acceleration_next = -(g / l) * np.sin(theta_next)
        omega_next = omega + 0.5 * (acceleration + acceleration_next) * dt

        theta_state = theta_next
        omega = omega_next
        theta[i] = theta_state

    return t, theta


def simulate_pendulum(T, dt, g, l, theta_0, eps=0.01, k=0.1, rho=5.0, seed=None):
    """Simulate a pendulum trajectory with additive measurement and process noise."""
    if seed is not None:
        np.random.seed(seed)

    t, theta_clean = clean_pendulum_trajectory(T, dt, g, l, theta_0)
    theta_noisy = np.empty_like(t, dtype=float)

    theta_state = float(theta_0)
    omega = 0.0
    measurement_noise = np.random.normal(0.0, np.sqrt(eps), size=t.shape)

    centered_time = (t / T) - 0.5
    weights = np.exp(-rho * centered_time ** 2)
    probs = weights / np.sum(weights)

    for i, _ in enumerate(t):
        if i == 0:
            theta_noisy[i] = theta_state + measurement_noise[i]
            continue

        if np.random.rand() < probs[i]:
            theta_state += k * theta_state

        acceleration = -(g / l) * np.sin(theta_state)
        theta_next = theta_state + omega * dt + 0.5 * acceleration * dt * dt
        acceleration_next = -(g / l) * np.sin(theta_next)
        omega_next = omega + 0.5 * (acceleration + acceleration_next) * dt

        theta_state = theta_next
        omega = omega_next
        theta_noisy[i] = theta_state + measurement_noise[i]

    return t, theta_clean, theta_noisy


def plot_trajectories(t, theta_clean, theta_noisy, save_path="pendulum_trajectory.png"):
    """Plot the clean and noisy pendulum trajectories."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, theta_clean, label="Clean trajectory", lw=2, color="tab:blue")
    ax.plot(t, theta_noisy, label="Noisy trajectory", lw=1.5, alpha=0.8, color="tab:orange")
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_title("Pendulum trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig, ax


def generate_trajectory_dataset(n_trajectories=100, T=10.0, dt=0.1, theta_0=0.1,
                                eps=0.03, rho=2.0, seed=42, output_dir="Data",
                                csv_name="realistic_trajectory.csv"):
    """Generate a realistic CSV dataset of pendulum trajectories with varied parameters."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / csv_name

    rng = np.random.default_rng(seed)
    rows = []

    for idx in range(n_trajectories):
        g = 9.81
        l = float(rng.uniform(0.5, 5.0))
        k = float(rng.uniform(0.001, 0.01))

        t, theta_clean, theta_noisy = simulate_pendulum(
            T=T,
            dt=dt,
            g=g,
            l=l,
            theta_0=theta_0,
            eps=eps,
            k=k,
            rho=rho,
            seed=int(seed + idx),
        )

        rows.append({
            "id": idx,
            "T": T,
            "dt": dt,
            "theta_0": theta_0,
            "eps": eps,
            "g": g,
            "l": l,
            "k": k,
            "time_values": ",".join(map(str, t)),
            "theta_values": ",".join(map(str, theta_noisy)),
            "theta_clean_values": ",".join(map(str, theta_clean)),
        })

    fieldnames = ["id", "T", "dt", "theta_0", "eps", "g", "l", "k", "time_values",
                  "theta_values", "theta_clean_values"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved realistic dataset to {csv_path}")
    return csv_path


def load_trajectory_dataset(csv_path):
    """Load the saved CSV dataset and parse the serialized trajectory values."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    trajectories = []
    for row in rows:
        trajectories.append({
            "id": int(row["id"]),
            "T": float(row["T"]),
            "dt": float(row["dt"]),
            "theta_0": float(row["theta_0"]),
            "eps": float(row["eps"]),
            "g": float(row["g"]),
            "l": float(row["l"]),
            "k": float(row["k"]),
            "time": np.fromstring(row["time_values"], sep=","),
            "theta": np.fromstring(row["theta_values"], sep=","),
            "theta_clean": np.fromstring(row["theta_clean_values"], sep=","),
        })

    return trajectories


def plot_dataset_trajectories(csv_path, n_trajectories=5, save_path="Data/first5_trajectories.png"):
    """Reload the CSV dataset and plot the first few trajectories in one figure."""
    trajectories = load_trajectory_dataset(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    for traj in trajectories[:n_trajectories]:
        label = f"traj {traj['id']} (k={traj['k']:.4f}, l={traj['l']:.3f})"
        ax.plot(traj["time"], traj["theta"], lw=1.2, alpha=0.8, label=label)

    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_title(f"First {n_trajectories} trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig, ax


def archive_previous_plots(output_dir="Data", archive_dir="Old Plots"):
    """Move previous plot artifacts into an archive directory before generating new ones."""
    output_path = Path(output_dir)
    archive_path = Path(archive_dir)
    archive_path.mkdir(exist_ok=True)

    for plot_path in output_path.glob("*.png"):
        target_path = archive_path / plot_path.name
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(plot_path), str(target_path))


def build_noise_features(t, y0, g, l):
    """Construct the feature matrix for the KNN noise model."""
    return np.column_stack([t, y0, np.full_like(t, g), np.full_like(t, l)])


def train_synthetic_simulator(trajectories):
    """Fit KNN regressors for the mean and variance of the observation residuals."""
    features = []
    targets_mu = []
    targets_var = []

    for traj in trajectories:
        t = traj["time"]
        y0 = traj["theta_clean"]
        residual = traj["theta"] - y0
        X = build_noise_features(t, y0, traj["g"], traj["l"])
        features.append(X)
        targets_mu.append(residual)
        targets_var.append(residual ** 2)

    X_train = np.vstack(features)
    y_mu = np.concatenate(targets_mu)
    y_var = np.concatenate(targets_var)

    mu_model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    mu_model.fit(X_train, y_mu)

    var_model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    var_model.fit(X_train, y_var)

    return {"mu_model": mu_model, "var_model": var_model}


def simulate_synthetic_trajectory(T, dt, theta_0, g, l, simulator, rng=None):
    """Generate a synthetic trajectory using the learned observation model."""
    if rng is None:
        rng = np.random.default_rng()

    t, y0 = clean_pendulum_trajectory(T, dt, g, l, theta_0)
    X = build_noise_features(t, y0, g, l)

    mu = simulator["mu_model"].predict(X)
    var = np.clip(simulator["var_model"].predict(X), 1e-8, None)
    sigma = np.sqrt(var)
    residual_sample = rng.normal(mu, sigma)
    y_synthetic = y0 + residual_sample
    return t, y0, y_synthetic


def generate_synthetic_training_set(n_trajectories=500, T=10.0, dt=0.1, theta_0=0.2,
                                   simulator=None, seed=123, output_dir="Data",
                                   csv_name="synthetic_trajectories.csv"):
    """Use the learned synthetic simulator to generate a labeled training set for the CNN."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / csv_name

    if simulator is None:
        raise ValueError("A trained simulator object is required.")

    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n_trajectories):
        g = 9.81
        l = float(rng.uniform(0.5, 5.0))
        t, _, y_synthetic = simulate_synthetic_trajectory(
            T=T,
            dt=dt,
            theta_0=theta_0,
            g=g,
            l=l,
            simulator=simulator,
            rng=np.random.default_rng(seed + idx),
        )
        rows.append({
            "id": idx,
            "g": g,
            "l": l,
            "time_values": ",".join(map(str, t)),
            "trajectory_values": ",".join(map(str, y_synthetic)),
        })

    fieldnames = ["id", "g", "l", "time_values", "trajectory_values"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved synthetic training set to {csv_path}")
    return csv_path


def load_synthetic_training_data(csv_path):
    """Load the synthetic CSV dataset and parse the serialized trajectories."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    trajectories = []
    labels = []
    for row in rows:
        trajectories.append(np.fromstring(row["trajectory_values"], sep=","))
        labels.append([float(row["g"]), float(row["l"])])

    return np.asarray(trajectories, dtype=float), np.asarray(labels, dtype=float)


def load_synthetic_trajectory_data(csv_path):
    """Load the synthetic CSV dataset and parse both time and trajectory values."""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    times = []
    trajectories = []
    labels = []
    for row in rows:
        times.append(np.fromstring(row["time_values"], sep=","))
        trajectories.append(np.fromstring(row["trajectory_values"], sep=","))
        labels.append([float(row["g"]), float(row["l"])])

    return np.asarray(times, dtype=float), np.asarray(trajectories, dtype=float), np.asarray(labels, dtype=float)


def extract_fourier_coeff_features(trajectory, n_components=40):
    """Extract a fixed-size feature vector from the real and imaginary parts of Fourier coefficients."""
    trajectory = np.asarray(trajectory, dtype=float)
    coeffs = np.fft.rfft(trajectory)
    if coeffs.size < n_components:
        coeffs = np.pad(coeffs, (0, n_components - coeffs.size))
    coeffs = coeffs[:n_components]
    return np.concatenate([coeffs.real, coeffs.imag])


def extract_time_domain_features(trajectory):
    """Compute a compact set of time-domain summary features for a trajectory."""
    trajectory = np.asarray(trajectory, dtype=float)
    delta = np.diff(trajectory)
    return np.array([
        float(np.mean(trajectory)),
        float(np.var(trajectory)),
        float(np.mean(delta)),
        float(np.var(delta)),
        float(np.max(trajectory)),
        float(np.min(trajectory)),
        float(np.max(delta)),
        float(np.min(delta)),
    ], dtype=float)


def extract_embedding_features(trajectory, n_components=20):
    """Combine Fourier coefficient features with time-domain summary features."""
    fourier_features = extract_fourier_coeff_features(trajectory, n_components=n_components)
    time_features = extract_time_domain_features(trajectory)
    return np.concatenate([fourier_features, time_features])


def save_embedded_synthetic_dataset(csv_path, output_csv_path, n_components=20):
    """Save an embedded version of the synthetic trajectories with Fourier features."""
    trajectories, labels = load_synthetic_training_data(csv_path)
    rows = []
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        feats = extract_embedding_features(traj, n_components=n_components)
        g = float(label[0])
        l = float(label[1])
        row = {"id": idx, "g": g, "l": l, "target_theta": g / l}
        for j, value in enumerate(feats):
            row[f"feature_{j}"] = float(value)
        rows.append(row)

    fieldnames = ["id", "g", "l", "target_theta"] + [f"feature_{j}" for j in range(2 * n_components + 8)]
    with Path(output_csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved embedded synthetic dataset to {output_csv_path}")
    return output_csv_path


def train_parameter_regressor(embedded_csv_path, random_state=42):
    """Train a regression encoder for theta = g/L on embedded trajectory features."""
    with Path(embedded_csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    feature_names = [name for name in rows[0].keys() if name.startswith("feature_")]
    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.array([float(row["target_theta"]) for row in rows], dtype=float)

    model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    model.fit(X, y)
    return model


def predict_with_regressor(model, trajectories):
    """Predict theta = g/L from trajectory sequences using the combined feature extractor."""
    X = np.array([extract_embedding_features(traj) for traj in trajectories], dtype=float)
    return model.predict(X)


def match_clean_trajectory_length(trajectory, T, dt, theta_0, g=9.81, candidate_l_values=None):
    """Find the clean-trajectory length whose reference curve is closest to a given trajectory."""
    if candidate_l_values is None:
        candidate_l_values = np.linspace(0.5, 5.0, 200)

    trajectory = np.asarray(trajectory, dtype=float)
    best_l = None
    best_distance = np.inf
    for l in candidate_l_values:
        _, clean_trajectory = clean_pendulum_trajectory(T, dt, g, l, theta_0)
        distance = np.linalg.norm(trajectory - clean_trajectory)
        if distance < best_distance:
            best_distance = distance
            best_l = float(l)
    return best_l


def match_clean_trajectory_theta(trajectory, T, dt, theta_0, g=9.81, candidate_theta_values=None):
    """Find the clean-trajectory theta = g/L whose reference curve is closest to a trajectory."""
    if candidate_theta_values is None:
        candidate_theta_values = np.linspace(g / 5.0, g / 0.5, 300)

    trajectory = np.asarray(trajectory, dtype=float)
    best_theta = None
    best_distance = np.inf
    for theta in candidate_theta_values:
        _, clean_trajectory = clean_pendulum_trajectory(T, dt, g, g / theta, theta_0)
        distance = np.linalg.norm(trajectory - clean_trajectory)
        if distance < best_distance:
            best_distance = distance
            best_theta = float(theta)
    return best_theta


def train_covariate_shift_discriminator(calibration_trajectories, test_trajectories):
    """Train a logistic-regression discriminator between calibration and test domains."""
    calibration_features = np.array([extract_embedding_features(traj) for traj in calibration_trajectories], dtype=float)
    test_features = np.array([extract_embedding_features(traj) for traj in test_trajectories], dtype=float)

    X = np.vstack([calibration_features, test_features])
    y = np.concatenate([np.zeros(len(calibration_features)), np.ones(len(test_features))])

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X, y)
    return model


def compute_covariate_shift_weights(discriminator, calibration_trajectories, test_trajectories):
    """Compute normalized importance weights for calibration points under covariate shift."""
    calibration_features = np.array([extract_embedding_features(traj) for traj in calibration_trajectories], dtype=float)
    test_features = np.array([extract_embedding_features(traj) for traj in test_trajectories], dtype=float)

    calibration_prob = discriminator.predict_proba(calibration_features)[:, 1]
    test_prob = discriminator.predict_proba(test_features)[:, 1]

    if len(test_prob) == 0:
        return np.ones(len(calibration_prob))

    weights = calibration_prob / (1.0 - calibration_prob + 1e-12)
    weights = weights / np.mean(weights)
    return np.clip(weights, 1e-8, None)


def compute_conformal_quantile(predictions, calibration_trajectories, T, dt, theta_0,
                               g=9.81, confidence=0.05, candidate_theta_values=None,
                               weights=None):
    """Calibrate a weighted conformal radius from synthetic held-out trajectories."""
    scores = []
    for pred, traj in zip(predictions, calibration_trajectories):
        matched_theta = match_clean_trajectory_theta(
            traj,
            T=T,
            dt=dt,
            theta_0=theta_0,
            g=g,
            candidate_theta_values=candidate_theta_values,
        )
        scores.append(abs(float(pred) - matched_theta))

    if not scores:
        return 0.0

    if weights is None:
        weights = np.ones(len(scores), dtype=float)

    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    if total_weight <= 0:
        return 0.0

    threshold_weight = confidence * total_weight
    quantile_index = np.searchsorted(cumulative_weights, threshold_weight, side="left")
    quantile_index = min(quantile_index, len(sorted_scores) - 1)
    q = float(sorted_scores[quantile_index])
    print(q, max(scores), min(scores), np.mean(scores), np.median(scores))
    return q


def compute_conformal_quantile_from_labels(predictions, true_values, confidence=0.68):
    """Calibrate a split-conformal radius from labelled calibration parameters."""
    scores = np.abs(np.asarray(predictions, dtype=float) - np.asarray(true_values, dtype=float))
    if scores.size == 0:
        return 0.0

    rank = int(np.ceil((scores.size + 1) * confidence))
    rank = min(max(rank, 1), scores.size)
    q = float(np.partition(scores, rank - 1)[rank - 1])
    print(q, max(scores), min(scores), np.mean(scores), np.median(scores))
    return q


def build_conformal_prediction_sets(predictions, conformal_radius):
    """Construct symmetric prediction intervals around point predictions."""
    pred_theta = np.asarray(predictions, dtype=float)
    lower = np.clip(pred_theta - conformal_radius, 1e-6, None)
    upper = pred_theta + conformal_radius
    return np.column_stack([lower, upper])


def plot_realistic_prediction_comparison(realistic_trajectories, predictions,
                                         bayesian_results,
                                         prediction_sets=None,
                                         save_path="Data/parameter_prediction_comparison.png"):
    """Compare conformal and Bayesian parameter estimates on realistic trajectories."""
    true_theta = np.array([traj["g"] / traj["l"] for traj in realistic_trajectories], dtype=float)
    cp_theta = np.asarray(predictions, dtype=float)
    bayesian_theta = np.array([result["theta_mean"] for result in bayesian_results], dtype=float)
    bayesian_lower = np.array([result["theta_lower"] for result in bayesian_results], dtype=float)
    bayesian_upper = np.array([result["theta_upper"] for result in bayesian_results], dtype=float)

    mse_cp = np.mean((cp_theta - true_theta) ** 2)
    mse_bayes = np.mean((bayesian_theta - true_theta) ** 2)
    cp_contains_bayes = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_theta, cp_theta, alpha=0.8, color="tab:blue", label="CP")
    ax.scatter(true_theta, bayesian_theta, alpha=0.8, color="tab:green", marker="x", s=80, label="Bayesian")
    axis_min = min(true_theta.min(), cp_theta.min(), bayesian_theta.min())
    axis_max = max(true_theta.max(), cp_theta.max(), bayesian_theta.max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="tab:red")

    if prediction_sets is not None:
        cp_lower_theta = prediction_sets[:, 0]
        cp_upper_theta = prediction_sets[:, 1]
        cp_yerr = np.column_stack([cp_theta - cp_lower_theta, cp_upper_theta - cp_theta])
        cp_contains_bayes = np.mean((bayesian_theta >= cp_lower_theta) & (bayesian_theta <= cp_upper_theta))
        ax.errorbar(true_theta, cp_theta, yerr=cp_yerr.T, fmt="none", ecolor="tab:gray",
                    elinewidth=1.2, capsize=3, alpha=0.7)

    bayes_yerr = np.column_stack([bayesian_theta - bayesian_lower, bayesian_upper - bayesian_theta])
    ax.errorbar(true_theta, bayesian_theta, yerr=bayes_yerr.T, fmt="none", ecolor="tab:green",
                elinewidth=1.2, capsize=3, alpha=0.7)

    ax.set_xlabel(r"True $\theta = g/L$")
    ax.set_ylabel(r"Predicted $\theta = g/L$")
    ax.set_title("CP vs. Bayesian parameter identification")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    stats = f"CP target confidence = 0.68\nCP MSE = {mse_cp:.3f}\nBayesian MSE = {mse_bayes:.3f}"
    if np.isfinite(cp_contains_bayes):
        stats += f"\nBayes in CP = {cp_contains_bayes:.2f}"
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9})

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved parameter comparison plot to {save_path}")
    plt.show()
    return fig, ax


def plot_residuals(calibration_true_values, calibration_predictions, test_true_values, test_predictions,
                   save_path="Data/residuals_comparison.png"):
    """Plot residuals on the calibration and test data for the regressor."""
    calibration_true = np.asarray(calibration_true_values, dtype=float)
    calibration_pred = np.asarray(calibration_predictions, dtype=float)
    test_true = np.asarray(test_true_values, dtype=float)
    test_pred = np.asarray(test_predictions, dtype=float)

    calibration_residuals = calibration_pred - calibration_true
    test_residuals = test_pred - test_true

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="tab:red", linestyle="--", lw=1)
    ax.scatter(np.arange(len(calibration_residuals)), calibration_residuals, color="tab:blue", alpha=0.7, label="Calibration")
    ax.scatter(np.arange(len(test_residuals)) + 0.2, test_residuals, color="tab:orange", alpha=0.7, label="Test")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual (predicted - true)")
    ax.set_title("Residuals on calibration and test data")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved residual plot to {save_path}")
    plt.show()
    return fig, ax


def plot_unseen_trajectories(heldout_trajectories, predictions, prediction_sets, bayesian_results, n_examples=4,
                             save_path="Data/bayesian_trajectory_comparison.png"):
    """Plot held-out trajectories with CP/Bayesian fit curves and uncertainty shadows."""
    n_examples = min(n_examples, len(heldout_trajectories))
    n_rows = int(np.ceil(n_examples / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    rng = np.random.default_rng(9)
    for ax, idx in zip(axes, range(n_examples)):
        traj = heldout_trajectories[idx]
        result = bayesian_results[idx]
        t = traj["time"]
        y_obs = traj["theta"]
        y0 = traj["theta_clean"]
        cp_theta = float(predictions[idx])
        cp_lo_theta, cp_hi_theta = map(float, prediction_sets[idx])

        theta_mean = result["theta_mean"]
        theta_std = result["theta_std"]
        cp_l = traj["g"] / cp_theta
        l_mean = traj["g"] / theta_mean
        _, y_cp = clean_pendulum_trajectory(
            T=traj["T"],
            dt=traj["dt"],
            g=traj["g"],
            l=cp_l,
            theta_0=traj["theta_0"],
        )
        _, y_bayes = clean_pendulum_trajectory(
            T=traj["T"],
            dt=traj["dt"],
            g=traj["g"],
            l=l_mean,
            theta_0=traj["theta_0"],
        )

        theta_samples = np.clip(rng.normal(theta_mean, theta_std, size=30), 1e-6, None)
        for theta_sample in theta_samples:
            l_sample = traj["g"] / theta_sample
            _, y_shadow = clean_pendulum_trajectory(
                T=traj["T"],
                dt=traj["dt"],
                g=traj["g"],
                l=l_sample,
                theta_0=traj["theta_0"],
            )
            ax.plot(t, y_shadow, lw=0.7, color="tab:green", alpha=0.03)

        if cp_hi_theta > cp_lo_theta > 0:
            for theta_sample in rng.uniform(cp_lo_theta, cp_hi_theta, size=30):
                l_sample = traj["g"] / theta_sample
                _, y_shadow = clean_pendulum_trajectory(
                    T=traj["T"],
                    dt=traj["dt"],
                    g=traj["g"],
                    l=l_sample,
                    theta_0=traj["theta_0"],
                )
                ax.plot(t, y_shadow, lw=0.7, color="tab:purple", alpha=0.03)

        ax.plot(t, y_obs, label="Observed", lw=1.2, color="tab:orange")
        ax.plot(t, y0, label="Clean", lw=1.8, color="tab:blue")
        ax.plot(t, y_cp, label="CP fit", lw=1.4, linestyle="--", color="tab:purple")
        ax.plot(t, y_bayes, label="Bayesian fit", lw=1.4, linestyle="--", color="tab:green")
        ax.set_title(
            f"Trajectory {traj['id']}: true theta={traj['g'] / traj['l']:.3f}, "
            f"CP={cp_theta:.3f}, Bayes={theta_mean:.3f}"
        )
        ax.grid(True, alpha=0.3)

    for ax in axes[n_examples:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("Held-out trajectories with CP and Bayesian uncertainty", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Bayesian trajectory comparison plot to {save_path}")

    plt.show()
    return fig, axes


if __name__ == "__main__":
    archive_previous_plots(output_dir="Data", archive_dir="Old Plots")

    realistic_csv_path = generate_trajectory_dataset(
        n_trajectories=100,
        T=10.0,
        dt=0.1,
        theta_0=0.3,
        eps=0.003,
        rho=1.5,
        seed=42,
    )
    realistic_trajectories = load_trajectory_dataset(realistic_csv_path)

    rng = np.random.default_rng(7)
    indices = rng.permutation(len(realistic_trajectories))
    n_train = int(0.5 * len(realistic_trajectories))
    train_indices = indices[:n_train]
    heldout_indices = indices[n_train:]

    train_trajectories = [realistic_trajectories[i] for i in train_indices]
    heldout_trajectories = [realistic_trajectories[i] for i in heldout_indices]

    simulator = train_synthetic_simulator(train_trajectories)
    synthetic_csv_path = generate_synthetic_training_set(
        n_trajectories=1000,
        T=10.0,
        dt=0.1,
        theta_0=0.3,
        simulator=simulator,
        seed=123,
        output_dir="Data",
        csv_name="synthetic_trajectories.csv",
    )
    synthetic_calibration_csv_path = generate_synthetic_training_set(
        n_trajectories=300,
        T=10.0,
        dt=0.1,
        theta_0=0.3,
        simulator=simulator,
        seed=456,
        output_dir="Data",
        csv_name="synthetic_trajectories_calibration.csv",
    )

    embedded_csv_path = save_embedded_synthetic_dataset(
        synthetic_csv_path,
        "Data/synthetic_trajectories_embedded.csv",
        n_components=20,
    )
    parameter_model = train_parameter_regressor(embedded_csv_path, random_state=42)
    realistic_sequences = [traj["theta"] for traj in heldout_trajectories]
    predictions = predict_with_regressor(parameter_model, realistic_sequences)

    _, calibration_trajectories, calibration_labels = load_synthetic_trajectory_data(synthetic_calibration_csv_path)
    calibration_predictions = predict_with_regressor(parameter_model, calibration_trajectories)
    calibration_true_values = np.array([float(label[0]) / float(label[1]) for label in calibration_labels], dtype=float)
    conformal_radius = compute_conformal_quantile_from_labels(
        calibration_predictions,
        calibration_true_values,
        confidence=0.68,
    )
    prediction_sets = build_conformal_prediction_sets(predictions, conformal_radius)

    bayesian_results = infer_bayesian_parameters(heldout_trajectories, theta_min=1.0, theta_max=20.0, N=500)
    plot_realistic_prediction_comparison(
        heldout_trajectories,
        predictions,
        bayesian_results,
        prediction_sets=prediction_sets,
        save_path="Data/parameter_prediction_comparison.png",
    )
    plot_unseen_trajectories(
        heldout_trajectories,
        predictions,
        prediction_sets,
        bayesian_results,
        n_examples=4,
        save_path="Data/bayesian_trajectory_comparison.png",
    )
