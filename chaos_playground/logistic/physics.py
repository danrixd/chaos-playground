"""Logistic map: x_{n+1} = r * x_n * (1 - x_n).

Scanning r produces the classic bifurcation fractal — period-doubling
cascade into chaos with windows of stability.
"""

from __future__ import annotations

import numpy as np


def bifurcation(
    r_min: float,
    r_max: float,
    n_r: int = 1600,
    n_warmup: int = 800,
    n_record: int = 400,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rs, xs) arrays sampling the attractor of the logistic map.

    For each r in `n_r` values spanning [r_min, r_max], iterate for
    `n_warmup` steps to settle onto the attractor, then record the next
    `n_record` values. Returns flattened (r, x) pairs suitable for scatter.
    """
    rs = np.linspace(r_min, r_max, n_r)
    x = np.full_like(rs, x0)
    for _ in range(n_warmup):
        x = rs * x * (1.0 - x)

    out_r = np.empty(n_r * n_record, dtype=float)
    out_x = np.empty(n_r * n_record, dtype=float)
    for k in range(n_record):
        x = rs * x * (1.0 - x)
        out_r[k * n_r:(k + 1) * n_r] = rs
        out_x[k * n_r:(k + 1) * n_r] = x
    return out_r, out_x
