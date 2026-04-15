"""Lorenz 1963 system.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

Canonical butterfly parameters: sigma=10, rho=28, beta=8/3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def derivatives(_t: float, y: np.ndarray, p: LorenzParams) -> np.ndarray:
    x, yy, z = y
    return np.array([
        p.sigma * (yy - x),
        x * (p.rho - z) - yy,
        x * yy - p.beta * z,
    ])
