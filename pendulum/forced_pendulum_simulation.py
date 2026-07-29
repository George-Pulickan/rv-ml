import csv
import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from forced_pendulum_bayesian import infer_forced_bayesian_parameters
from pendulum_simulation import extract_embedding_features

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMETER_NAMES = [r"$\theta = g/L$", r"$\lambda$"]


def forced_acceleration(time, angle, theta, forcing):
    """Acceleration for x'' = -theta sin(x) + forcing cos(1.7t) x(1 - x^2)."""
    return -theta * np.sin(angle) + forcing * np.cos(1.7 * time) * angle * (1.0 - angle * angle)


def clean_forced_pendulum_trajectory(T, dt, theta, forcing, theta_0):
    if T <= 0 or dt <= 0 or theta <= 0:
        raise ValueError("T, dt, and theta must be positive.")

    t = np.arange(0.0, T + dt, dt)
    if t[-1] > T:
        t = t[t <= T]

    y = np.empty_like(t, dtype=float)
    angle = float(theta_0)
    velocity = 0.0
    for i, _ in enumerate(t):
        if i == 0:
            y[i] = angle
            continue

        acceleration = forced_acceleration(t[i - 1], angle, theta, forcing)
        angle_next = angle + velocity * dt + 0.5 * acceleration * dt * dt
        acceleration_next = forced_acceleration(t[i], angle_next, theta, forcing)
        velocity_next = velocity + 0.5 * (acceleration + acceleration_next) * dt

        angle = angle_next
        velocity = velocity_next
        y[i] = angle

    return t, y


def simulate_realistic_forced_pendulum(
    T,
    dt,
    theta,
    forcing,
    theta_0,
    eps=0.006,
    process_scale=0.02,
    rho=1.5,
    seed=None,
):
    """Generate a noisy trajectory with unmodelled process perturbations."""
    if seed is not None:
        np.random.seed(seed)

    t, clean = clean_forced_pendulum_trajectory(T, dt, theta, forcing, theta_0)
    y = np.empty_like(t, dtype=float)
    measurement_noise = np.random.normal(0.0, np.sqrt(eps), size=t.shape)

    angle = float(theta_0)
    velocity = 0.0
    centered_time = (t / T) - 0.5
    weights = np.exp(-rho * centered_time ** 2)
    probs = weights / np.sum(weights)

    for i, _ in enumerate(t):
        if i == 0:
            y[i] = angle + measurement_noise[i]
            continue

        if np.random.rand() < probs[i]:
            angle += process_scale * angle * (1.0 - angle * angle)

        acceleration = forced_acceleration(t[i - 1], angle, theta, forcing)
        angle_next = angle + velocity * dt + 0.5 * acceleration * dt * dt
        acceleration_next = forced_acceleration(t[i], angle_next, theta, forcing)
        velocity_next = velocity + 0.5 * (acceleration + acceleration_next) * dt

        angle = angle_next
        velocity = velocity_next
        y[i] = angle + measurement_noise[i]

    return t, clean, y


def generate_realistic_dataset(
    n_trajectories=90,
    T=12.0,
    dt=0.1,
    theta_0=0.8,
    eps=0.006,
    seed=42,
    output_dir="Data",
    csv_name="forced_realistic_trajectory.csv",
):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / csv_name
    rng = np.random.default_rng(seed)
    rows = []

    for idx in range(n_trajectories):
        theta = float(rng.uniform(2.0, 16.0))
        forcing = float(rng.uniform(-2.0, 2.0))
        process_scale = float(rng.uniform(0.005, 0.035))
        t, clean, noisy = simulate_realistic_forced_pendulum(
            T=T,
            dt=dt,
            theta=theta,
            forcing=forcing,
            theta_0=theta_0,
            eps=eps,
            process_scale=process_scale,
            seed=int(seed + idx),
        )
        rows.append({
            "id": idx,
            "T": T,
            "dt": dt,
            "theta_0": theta_0,
            "eps": eps,
            "theta_param": theta,
            "forcing": forcing,
            "process_scale": process_scale,
            "time_values": ",".join(map(str, t)),
            "theta_values": ",".join(map(str, noisy)),
            "theta_clean_values": ",".join(map(str, clean)),
        })

    fieldnames = [
        "id", "T", "dt", "theta_0", "eps", "theta_param", "forcing",
        "process_scale", "time_values", "theta_values", "theta_clean_values",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved forced realistic dataset to {csv_path}")
    return csv_path


def load_realistic_dataset(csv_path):
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
            "theta_param": float(row["theta_param"]),
            "forcing": float(row["forcing"]),
            "process_scale": float(row["process_scale"]),
            "time": np.fromstring(row["time_values"], sep=","),
            "theta": np.fromstring(row["theta_values"], sep=","),
            "theta_clean": np.fromstring(row["theta_clean_values"], sep=","),
        })
    return trajectories


def generate_synthetic_dataset(
    n_trajectories=1200,
    T=12.0,
    dt=0.1,
    theta_0=0.8,
    eps=0.006,
    seed=123,
    output_dir="Data",
    csv_name="forced_synthetic_trajectories.csv",
):
    """Generate labelled synthetic trajectories from the ideal forced simulator."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / csv_name
    rng = np.random.default_rng(seed)
    rows = []

    for idx in range(n_trajectories):
        theta = float(rng.uniform(2.0, 16.0))
        forcing = float(rng.uniform(-2.0, 2.0))
        t, clean = clean_forced_pendulum_trajectory(T, dt, theta, forcing, theta_0)
        noisy = clean + rng.normal(0.0, np.sqrt(eps), size=clean.shape)
        rows.append({
            "id": idx,
            "theta_param": theta,
            "forcing": forcing,
            "time_values": ",".join(map(str, t)),
            "trajectory_values": ",".join(map(str, noisy)),
        })

    fieldnames = ["id", "theta_param", "forcing", "time_values", "trajectory_values"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved forced synthetic dataset to {csv_path}")
    return csv_path


def load_synthetic_dataset(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    trajectories = []
    labels = []
    for row in rows:
        trajectories.append(np.fromstring(row["trajectory_values"], sep=","))
        labels.append([float(row["theta_param"]), float(row["forcing"])])
    return np.asarray(trajectories, dtype=float), np.asarray(labels, dtype=float)


def save_embedded_dataset(csv_path, output_csv_path, n_components=20):
    trajectories, labels = load_synthetic_dataset(csv_path)
    rows = []
    for idx, (trajectory, label) in enumerate(zip(trajectories, labels)):
        features = extract_embedding_features(trajectory, n_components=n_components)
        row = {
            "id": idx,
            "target_theta": float(label[0]),
            "target_forcing": float(label[1]),
        }
        for j, value in enumerate(features):
            row[f"feature_{j}"] = float(value)
        rows.append(row)

    fieldnames = ["id", "target_theta", "target_forcing"] + [f"feature_{j}" for j in range(2 * n_components + 8)]
    with Path(output_csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved forced embedded dataset to {output_csv_path}")
    return output_csv_path


def train_parameter_regressor(embedded_csv_path):
    with Path(embedded_csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feature_names = [name for name in rows[0] if name.startswith("feature_")]
    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.array([[float(row["target_theta"]), float(row["target_forcing"])] for row in rows], dtype=float)
    model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    model.fit(X, y)
    return model


def predict_with_regressor(model, trajectories):
    X = np.array([extract_embedding_features(traj) for traj in trajectories], dtype=float)
    return model.predict(X)


def conformal_radii(predictions, true_values, confidence=0.68):
    """Return coordinate-wise split-conformal radii at the supplied coordinate confidence."""
    scores = np.abs(np.asarray(predictions, dtype=float) - np.asarray(true_values, dtype=float))
    radii = []
    for dim in range(scores.shape[1]):
        dim_scores = scores[:, dim]
        rank = int(np.ceil((dim_scores.size + 1) * confidence))
        rank = min(max(rank, 1), dim_scores.size)
        radii.append(float(np.partition(dim_scores, rank - 1)[rank - 1]))
    return np.asarray(radii, dtype=float)


def build_prediction_boxes(predictions, radii):
    predictions = np.asarray(predictions, dtype=float)
    lower = predictions - radii
    upper = predictions + radii
    lower[:, 0] = np.clip(lower[:, 0], 1e-6, None)
    return np.stack([lower, upper], axis=1)


def summarize_results(heldout_trajectories, predictions, prediction_boxes, bayesian_results):
    true_values = np.array([[traj["theta_param"], traj["forcing"]] for traj in heldout_trajectories], dtype=float)
    bayes_means = np.array([result["mean"] for result in bayesian_results], dtype=float)
    bayes_lower = np.array([result["lower"] for result in bayesian_results], dtype=float)
    bayes_upper = np.array([result["upper"] for result in bayesian_results], dtype=float)

    true_in_cp = (true_values >= prediction_boxes[:, 0, :]) & (true_values <= prediction_boxes[:, 1, :])
    bayes_in_cp = (bayes_means >= prediction_boxes[:, 0, :]) & (bayes_means <= prediction_boxes[:, 1, :])
    true_in_bayes = (true_values >= bayes_lower) & (true_values <= bayes_upper)

    return {
        "cp_mse": np.mean((predictions - true_values) ** 2, axis=0),
        "bayes_mse": np.mean((bayes_means - true_values) ** 2, axis=0),
        "true_in_cp_marginal": np.mean(true_in_cp, axis=0),
        "true_in_cp_joint": float(np.mean(np.all(true_in_cp, axis=1))),
        "bayes_in_cp_marginal": np.mean(bayes_in_cp, axis=0),
        "bayes_in_cp_joint": float(np.mean(np.all(bayes_in_cp, axis=1))),
        "true_in_bayes_marginal": np.mean(true_in_bayes, axis=0),
        "true_in_bayes_joint": float(np.mean(np.all(true_in_bayes, axis=1))),
    }


def plot_parameter_comparison(heldout_trajectories, predictions, prediction_boxes, bayesian_results,
                              joint_confidence=0.68,
                              save_path="Data/forced_parameter_prediction_comparison.png"):
    true_values = np.array([[traj["theta_param"], traj["forcing"]] for traj in heldout_trajectories], dtype=float)
    bayes_means = np.array([result["mean"] for result in bayesian_results], dtype=float)
    bayes_lower = np.array([result["lower"] for result in bayesian_results], dtype=float)
    bayes_upper = np.array([result["upper"] for result in bayesian_results], dtype=float)

    summary = summarize_results(heldout_trajectories, predictions, prediction_boxes, bayesian_results)
    cp_mse = summary["cp_mse"]
    bayes_mse = summary["bayes_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85))
    for dim, ax in enumerate(axes):
        ax.scatter(true_values[:, dim], predictions[:, dim], color="tab:blue", alpha=0.82, s=8, label="CP")
        ax.scatter(true_values[:, dim], bayes_means[:, dim], color="tab:green", marker="x", s=16, alpha=0.82, label="Bayesian")

        cp_yerr = np.column_stack([
            predictions[:, dim] - prediction_boxes[:, 0, dim],
            prediction_boxes[:, 1, dim] - predictions[:, dim],
        ])
        ax.errorbar(true_values[:, dim], predictions[:, dim], yerr=cp_yerr.T, fmt="none",
                    ecolor="tab:gray", elinewidth=0.55, capsize=1.0, alpha=0.6)

        bayes_yerr = np.column_stack([
            bayes_means[:, dim] - bayes_lower[:, dim],
            bayes_upper[:, dim] - bayes_means[:, dim],
        ])
        ax.errorbar(true_values[:, dim], bayes_means[:, dim], yerr=bayes_yerr.T, fmt="none",
                    ecolor="tab:green", elinewidth=0.55, capsize=1.0, alpha=0.6)

        axis_min = min(true_values[:, dim].min(), predictions[:, dim].min(), bayes_means[:, dim].min())
        axis_max = max(true_values[:, dim].max(), predictions[:, dim].max(), bayes_means[:, dim].max())
        margin = 0.05 * (axis_max - axis_min + 1e-9)
        ax.plot([axis_min - margin, axis_max + margin], [axis_min - margin, axis_max + margin],
                linestyle="--", color="tab:red")
        ax.set_xlabel(f"True {PARAMETER_NAMES[dim]}", fontsize=7)
        ax.set_ylabel(f"Pred. {PARAMETER_NAMES[dim]}", fontsize=7)
        ax.grid(True, alpha=0.28)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        ax.set_title(f"CP={cp_mse[dim]:.3f}, Bayes={bayes_mse[dim]:.3f}", fontsize=7)

    axes[0].legend(loc="upper left", fontsize=5.8, frameon=True, borderpad=0.2, handletextpad=0.25)
    fig.tight_layout(pad=0.2, w_pad=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved forced parameter comparison plot to {save_path}")
    plt.close(fig)
    return fig, axes


def plot_forced_trajectories(heldout_trajectories, predictions, prediction_boxes, bayesian_results, n_examples=4,
                             save_path="Data/forced_trajectory_comparison.png"):
    n_examples = min(n_examples, len(heldout_trajectories))
    n_rows = int(np.ceil(n_examples / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.6 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    rng = np.random.default_rng(11)

    for ax, idx in zip(axes, range(n_examples)):
        traj = heldout_trajectories[idx]
        bayes = bayesian_results[idx]
        t = traj["time"]
        cp_theta, cp_forcing = predictions[idx]
        bayes_theta, bayes_forcing = bayes["mean"]

        _, cp_curve = clean_forced_pendulum_trajectory(traj["T"], traj["dt"], cp_theta, cp_forcing, traj["theta_0"])
        _, bayes_curve = clean_forced_pendulum_trajectory(traj["T"], traj["dt"], bayes_theta, bayes_forcing, traj["theta_0"])

        lower, upper = prediction_boxes[idx]
        for sample in rng.uniform(lower, upper, size=(25, 2)):
            _, shadow = clean_forced_pendulum_trajectory(traj["T"], traj["dt"], sample[0], sample[1], traj["theta_0"])
            ax.plot(t, shadow, color="tab:purple", lw=0.7, alpha=0.03)

        bayes_lower = bayes["lower"].copy()
        bayes_upper = bayes["upper"].copy()
        bayes_lower[0] = max(bayes_lower[0], 1e-6)
        for sample in rng.uniform(bayes_lower, bayes_upper, size=(25, 2)):
            _, shadow = clean_forced_pendulum_trajectory(traj["T"], traj["dt"], sample[0], sample[1], traj["theta_0"])
            ax.plot(t, shadow, color="tab:green", lw=0.7, alpha=0.03)

        ax.plot(t, traj["theta"], label="Observed", lw=1.1, color="tab:orange")
        ax.plot(t, traj["theta_clean"], label="Clean", lw=1.8, color="tab:blue")
        ax.plot(t, cp_curve, label="CP fit", lw=1.4, linestyle="--", color="tab:purple")
        ax.plot(t, bayes_curve, label="Bayesian fit", lw=1.4, linestyle="--", color="tab:green")
        ax.set_title(
            f"Trajectory {traj['id']}: true "
            f"$\\theta$={traj['theta_param']:.2f}, $\\lambda$={traj['forcing']:.2f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    for ax in axes[n_examples:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("Forced pendulum held-out trajectories", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved forced trajectory plot to {save_path}")
    plt.close(fig)
    return fig, axes


if __name__ == "__main__":
    joint_confidence = 0.68
    coordinate_confidence = 1.0 - (1.0 - joint_confidence) / 2.0

    realistic_csv = generate_realistic_dataset()
    realistic = load_realistic_dataset(realistic_csv)

    rng = np.random.default_rng(7)
    indices = rng.permutation(len(realistic))
    n_train = int(0.5 * len(realistic))
    heldout = [realistic[i] for i in indices[n_train:]]

    synthetic_csv = generate_synthetic_dataset(csv_name="forced_synthetic_trajectories.csv")
    calibration_csv = generate_synthetic_dataset(
        n_trajectories=400,
        seed=456,
        csv_name="forced_synthetic_trajectories_calibration.csv",
    )
    embedded_csv = save_embedded_dataset(synthetic_csv, "Data/forced_synthetic_trajectories_embedded.csv")
    model = train_parameter_regressor(embedded_csv)

    predictions = predict_with_regressor(model, [traj["theta"] for traj in heldout])
    calibration_trajectories, calibration_labels = load_synthetic_dataset(calibration_csv)
    calibration_predictions = predict_with_regressor(model, calibration_trajectories)
    radii = conformal_radii(calibration_predictions, calibration_labels, confidence=coordinate_confidence)
    prediction_boxes = build_prediction_boxes(predictions, radii)
    print(
        f"Joint confidence={joint_confidence:.2f}, coordinate confidence={coordinate_confidence:.2f}, "
        f"radii theta={radii[0]:.4f}, forcing={radii[1]:.4f}"
    )

    bayesian_results = infer_forced_bayesian_parameters(
        heldout,
        theta_min=1.5,
        theta_max=18.0,
        forcing_min=-2.5,
        forcing_max=2.5,
        n_theta=70,
        n_forcing=61,
    )

    summary = summarize_results(heldout, predictions, prediction_boxes, bayesian_results)
    print(
        "Summary: "
        f"CP MSE={summary['cp_mse']}, Bayes MSE={summary['bayes_mse']}, "
        f"true in CP marginal={summary['true_in_cp_marginal']}, true in CP joint={summary['true_in_cp_joint']:.3f}, "
        f"Bayes in CP marginal={summary['bayes_in_cp_marginal']}, Bayes in CP joint={summary['bayes_in_cp_joint']:.3f}, "
        f"true in Bayes marginal={summary['true_in_bayes_marginal']}, true in Bayes joint={summary['true_in_bayes_joint']:.3f}"
    )

    plot_parameter_comparison(heldout, predictions, prediction_boxes, bayesian_results, joint_confidence=joint_confidence)
    plot_forced_trajectories(heldout, predictions, prediction_boxes, bayesian_results)
