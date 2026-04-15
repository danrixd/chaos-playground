"""Double pendulum equations of motion.

State: y = [theta1, omega1, theta2, omega2]
Derivation from the Lagrangian L = T - V with generalized coordinates
(theta1, theta2) — see e.g. Goldstein, Classical Mechanics, Ch. 1.
The closed-form derivatives below are the standard result; any introductory
classical-mechanics text arrives at the same expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoublePendulumParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


def derivatives(_t: float, y: np.ndarray, p: DoublePendulumParams) -> np.ndarray:
    theta1, omega1, theta2, omega2 = y
    m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g

    delta = theta2 - theta1
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    denom1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d
    denom2 = (l2 / l1) * denom1

    a1 = (
        m2 * l1 * omega1 * omega1 * sin_d * cos_d
        + m2 * g * np.sin(theta2) * cos_d
        + m2 * l2 * omega2 * omega2 * sin_d
        - (m1 + m2) * g * np.sin(theta1)
    ) / denom1

    a2 = (
        -m2 * l2 * omega2 * omega2 * sin_d * cos_d
        + (m1 + m2) * g * np.sin(theta1) * cos_d
        - (m1 + m2) * l1 * omega1 * omega1 * sin_d
        - (m1 + m2) * g * np.sin(theta2)
    ) / denom2

    return np.array([omega1, a1, omega2, a2])


def cartesian(y: np.ndarray, p: DoublePendulumParams) -> np.ndarray:
    """Map state trajectory (..., 4) to bob positions (..., 4): x1, y1, x2, y2."""
    theta1 = y[..., 0]
    theta2 = y[..., 2]
    x1 = p.l1 * np.sin(theta1)
    y1 = -p.l1 * np.cos(theta1)
    x2 = x1 + p.l2 * np.sin(theta2)
    y2 = y1 - p.l2 * np.cos(theta2)
    return np.stack([x1, y1, x2, y2], axis=-1)
