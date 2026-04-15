"""Classical 4th-order Runge-Kutta integrator.

Reused by the double pendulum and Lorenz renderers. Kept deliberately simple:
a pure function taking a callable `f(t, y) -> dy/dt`, an initial state, a
timestep, and a step count. Returns the full trajectory as an (N+1, D) array.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

OdeFn = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: OdeFn, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    f: OdeFn,
    y0: np.ndarray,
    dt: float,
    n_steps: int,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate dy/dt = f(t, y) from t0 for n_steps of size dt.

    Returns (ts, ys) with shapes (n_steps+1,) and (n_steps+1, D).
    """
    y0 = np.asarray(y0, dtype=float)
    ys = np.empty((n_steps + 1, y0.size), dtype=float)
    ts = np.empty(n_steps + 1, dtype=float)
    ys[0] = y0
    ts[0] = t0
    y = y0
    t = t0
    for i in range(n_steps):
        y = rk4_step(f, t, y, dt)
        t += dt
        ys[i + 1] = y
        ts[i + 1] = t
    return ts, ys
