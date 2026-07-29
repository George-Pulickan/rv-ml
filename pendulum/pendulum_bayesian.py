import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

##############################################################
# Pendulum simulator
##############################################################

def pendulum_rhs(t, x, theta):
    """
    x = [angle, angular velocity]
    theta = g/L
    """
    return [
        x[1],
        -theta * np.sin(x[0])
    ]


def simulate(theta, t, y0):
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t, dtype=float)
    angle = float(y0[0])
    velocity = float(y0[1]) if len(y0) > 1 else 0.0
    for i, _ in enumerate(t):
        if i == 0:
            y[i] = angle
            continue
        dt = float(t[i] - t[i - 1])
        acceleration = -theta * np.sin(angle)
        angle_next = angle + velocity * dt + 0.5 * acceleration * dt * dt
        acceleration_next = -theta * np.sin(angle_next)
        velocity_next = velocity + 0.5 * (acceleration + acceleration_next) * dt
        angle = angle_next
        velocity = velocity_next
        y[i] = angle
    return y


##############################################################
# Bayesian inference
##############################################################

def log_likelihood(theta, t, observations, y0, sigma):
    """
    Gaussian observation noise.
    """

    prediction = simulate(theta, t, y0)

    residual = observations - prediction

    return np.sum(norm.logpdf(residual, loc=0.0, scale=sigma))


def posterior_grid(
    t,
    observations,
    y0,
    sigma,
    theta_min=1.0,
    theta_max=20.0,
    N=400
):
    theta_grid = np.linspace(theta_min, theta_max, N)

    # Uniform prior
    log_prior = np.zeros(N)

    log_like = np.array([
        log_likelihood(th, t, observations, y0, sigma)
        for th in theta_grid
    ])

    log_post = log_prior + log_like

    # Numerical stabilization
    log_post -= np.max(log_post)

    posterior = np.exp(log_post)

    normalizer = np.trapezoid(posterior, theta_grid)
    if normalizer <= 0 or not np.isfinite(normalizer):
        posterior = np.ones_like(theta_grid, dtype=float)
        normalizer = np.trapezoid(posterior, theta_grid)
    posterior /= normalizer

    return theta_grid, posterior


##############################################################
# Bayesian inference helpers
##############################################################

def infer_bayesian_parameter(traj, theta_min=1.0, theta_max=20.0, N=500):
    """Infer the pendulum parameter theta = g/L for a single trajectory."""
    t = np.asarray(traj["time"], dtype=float)
    observations = np.asarray(traj["theta"], dtype=float)
    y0 = [float(traj["theta_0"]), 0.0]
    sigma = np.sqrt(max(float(traj.get("eps", 0.01)), 1e-6))

    theta_grid, posterior = posterior_grid(
        t,
        observations,
        y0,
        sigma,
        theta_min=theta_min,
        theta_max=theta_max,
        N=N,
    )

    theta_map = theta_grid[np.argmax(posterior)]
    theta_mean = np.trapezoid(theta_grid * posterior, theta_grid)
    theta_var = np.trapezoid((theta_grid - theta_mean) ** 2 * posterior, theta_grid)
    theta_std = np.sqrt(max(theta_var, 1e-12))

    return {
        "theta_grid": theta_grid,
        "posterior": posterior,
        "theta_map": theta_map,
        "theta_mean": float(theta_mean),
        "theta_std": float(theta_std),
        "theta_lower": float(theta_mean - theta_std),
        "theta_upper": float(theta_mean + theta_std),
    }


def infer_bayesian_parameters(trajectories, theta_min=1.0, theta_max=20.0, N=500):
    """Infer Bayesian posterior summaries for a list of trajectories."""
    return [
        infer_bayesian_parameter(traj, theta_min=theta_min, theta_max=theta_max, N=N)
        for traj in trajectories
    ]


if __name__ == "__main__":
    np.random.seed(0)

    theta_true = 9.81 / 1.25
    print("True theta =", theta_true)

    t = np.linspace(0, 8, 200)
    y0 = [0.8, 0.0]
    sigma = 0.05

    training_data = []
    for _ in range(20):
        ic = [
            np.random.uniform(0.2, 1.2),
            np.random.uniform(-0.5, 0.5),
        ]
        clean = simulate(theta_true, t, ic)
        noisy = clean + sigma * np.random.randn(len(t))
        training_data.append((ic, noisy))

    heldout_ic = [0.6, 0.1]
    clean = simulate(theta_true, t, heldout_ic)
    heldout = clean + sigma * np.random.randn(len(t))

    theta_grid, posterior = posterior_grid(
        t,
        heldout,
        heldout_ic,
        sigma,
        theta_min=4,
        theta_max=12,
        N=500,
    )

    theta_map = theta_grid[np.argmax(posterior)]
    print("MAP estimate =", theta_map)

    plt.figure(figsize=(8, 4))
    plt.plot(theta_grid, posterior, lw=3)
    plt.axvline(theta_true, color="red", linestyle="--", label="True")
    plt.axvline(theta_map, color="green", linestyle=":", label="MAP")
    plt.xlabel(r"$\theta = g/L$")
    plt.ylabel("Posterior density")
    plt.title("Posterior over pendulum parameter")
    plt.legend()
    plt.tight_layout()
    plt.show()
