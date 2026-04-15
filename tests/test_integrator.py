"""RK4 sanity: energy of a 1D harmonic oscillator drifts by < 1e-6 over 10 periods."""

import numpy as np

from chaos_playground.shared.integrator import integrate


def test_harmonic_oscillator_energy_conservation():
    omega = 2.0 * np.pi

    def f(_t, y):
        x, v = y
        return np.array([v, -omega * omega * x])

    y0 = np.array([1.0, 0.0])
    dt = 1e-3
    n_steps = int(10.0 / dt)  # 10 seconds == 10 periods
    _ts, ys = integrate(f, y0, dt, n_steps)

    e0 = 0.5 * ys[0, 1] ** 2 + 0.5 * omega ** 2 * ys[0, 0] ** 2
    e = 0.5 * ys[:, 1] ** 2 + 0.5 * omega ** 2 * ys[:, 0] ** 2
    rel_drift = np.max(np.abs(e - e0) / e0)
    assert rel_drift < 1e-6, f"RK4 energy drift too large: {rel_drift:.2e}"
