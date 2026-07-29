import numpy as np
from scipy.stats import norm


def forced_acceleration(time, angle, theta, forcing):
    """Acceleration for x'' = -theta sin(x) + forcing cos(1.7t) x(1 - x^2)."""
    return -theta * np.sin(angle) + forcing * np.cos(1.7 * time) * angle * (1.0 - angle * angle)


def simulate(theta, forcing, t, y0):
    """Integrate the forced pendulum with a velocity-Verlet scheme."""
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t, dtype=float)
    angle = float(y0[0])
    velocity = float(y0[1]) if len(y0) > 1 else 0.0

    for i, _ in enumerate(t):
        if i == 0:
            y[i] = angle
            continue

        dt = float(t[i] - t[i - 1])
        acceleration = forced_acceleration(t[i - 1], angle, theta, forcing)
        angle_next = angle + velocity * dt + 0.5 * acceleration * dt * dt
        acceleration_next = forced_acceleration(t[i], angle_next, theta, forcing)
        velocity_next = velocity + 0.5 * (acceleration + acceleration_next) * dt

        angle = angle_next
        velocity = velocity_next
        y[i] = angle

    return y


def log_likelihood(theta, forcing, t, observations, y0, sigma):
    prediction = simulate(theta, forcing, t, y0)
    residual = observations - prediction
    return np.sum(norm.logpdf(residual, loc=0.0, scale=sigma))


def posterior_grid(
    t,
    observations,
    y0,
    sigma,
    theta_min=1.5,
    theta_max=18.0,
    forcing_min=-2.5,
    forcing_max=2.5,
    n_theta=70,
    n_forcing=61,
):
    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    forcing_grid = np.linspace(forcing_min, forcing_max, n_forcing)
    log_post = np.empty((n_theta, n_forcing), dtype=float)

    for i, theta in enumerate(theta_grid):
        for j, forcing in enumerate(forcing_grid):
            log_post[i, j] = log_likelihood(theta, forcing, t, observations, y0, sigma)

    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    normalizer = posterior.sum()
    if normalizer <= 0 or not np.isfinite(normalizer):
        posterior = np.ones_like(posterior, dtype=float)
        normalizer = posterior.sum()
    posterior /= normalizer

    return theta_grid, forcing_grid, posterior


def posterior_summary(theta_grid, forcing_grid, posterior):
    theta_mesh, forcing_mesh = np.meshgrid(theta_grid, forcing_grid, indexing="ij")
    theta_mean = float(np.sum(theta_mesh * posterior))
    forcing_mean = float(np.sum(forcing_mesh * posterior))

    theta_centered = theta_mesh - theta_mean
    forcing_centered = forcing_mesh - forcing_mean
    cov = np.array([
        [np.sum(theta_centered * theta_centered * posterior), np.sum(theta_centered * forcing_centered * posterior)],
        [np.sum(theta_centered * forcing_centered * posterior), np.sum(forcing_centered * forcing_centered * posterior)],
    ], dtype=float)
    cov += 1e-10 * np.eye(2)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    map_index = np.unravel_index(np.argmax(posterior), posterior.shape)

    return {
        "theta_grid": theta_grid,
        "forcing_grid": forcing_grid,
        "posterior": posterior,
        "mean": np.array([theta_mean, forcing_mean], dtype=float),
        "std": std,
        "cov": cov,
        "lower": np.array([theta_mean - std[0], forcing_mean - std[1]], dtype=float),
        "upper": np.array([theta_mean + std[0], forcing_mean + std[1]], dtype=float),
        "map": np.array([theta_grid[map_index[0]], forcing_grid[map_index[1]]], dtype=float),
    }


def infer_forced_bayesian_parameter(
    traj,
    theta_min=1.5,
    theta_max=18.0,
    forcing_min=-2.5,
    forcing_max=2.5,
    n_theta=70,
    n_forcing=61,
):
    t = np.asarray(traj["time"], dtype=float)
    observations = np.asarray(traj["theta"], dtype=float)
    y0 = [float(traj["theta_0"]), 0.0]
    sigma = np.sqrt(max(float(traj.get("eps", 0.01)), 1e-6))

    theta_grid, forcing_grid, posterior = posterior_grid(
        t,
        observations,
        y0,
        sigma,
        theta_min=theta_min,
        theta_max=theta_max,
        forcing_min=forcing_min,
        forcing_max=forcing_max,
        n_theta=n_theta,
        n_forcing=n_forcing,
    )
    return posterior_summary(theta_grid, forcing_grid, posterior)


def infer_forced_bayesian_parameters(trajectories, **kwargs):
    return [infer_forced_bayesian_parameter(traj, **kwargs) for traj in trajectories]
