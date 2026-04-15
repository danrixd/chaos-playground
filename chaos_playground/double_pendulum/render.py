"""Render two near-identical double pendulums diverging into chaos.

Two initial conditions differ only by 1e-3 rad in theta1. Over ~10 s of
simulated time the trajectories pull apart visibly — the canonical
"sensitivity to initial conditions" image. Output is an autoplaying GIF
sized to stay under GitHub's README threshold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from chaos_playground.shared.integrator import integrate

from .physics import DoublePendulumParams, cartesian, derivatives


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "animations" / "double_pendulum.gif"


def simulate(params: DoublePendulumParams, y0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    def f(t, y):
        return derivatives(t, y, params)

    _ts, ys = integrate(f, y0, dt, n_steps)
    return ys


def render(output_path: Path = OUTPUT) -> Path:
    params = DoublePendulumParams()
    dt_sim = 1.0 / 600.0
    t_total = 12.0
    n_sim = int(t_total / dt_sim)

    y0_a = np.array([2.3, 0.0, 2.4, 0.0])
    y0_b = y0_a + np.array([1e-3, 0.0, 0.0, 0.0])

    ys_a = simulate(params, y0_a, dt_sim, n_sim)
    ys_b = simulate(params, y0_b, dt_sim, n_sim)
    xy_a = cartesian(ys_a, params)
    xy_b = cartesian(ys_b, params)

    fps = 30
    stride = max(1, int(round(1.0 / (fps * dt_sim))))
    xy_a = xy_a[::stride]
    xy_b = xy_b[::stride]
    n_frames = xy_a.shape[0]
    trail_len = 60

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=90)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")
    r = params.l1 + params.l2 + 0.2
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02, 0.98, r"$\Delta\theta_0 = 10^{-3}$ rad",
        transform=ax.transAxes, ha="left", va="top",
        color="#cbd5e1", fontsize=11,
    )

    color_a = "#38bdf8"
    color_b = "#f472b6"

    trail_a, = ax.plot([], [], color=color_a, lw=1.2, alpha=0.55)
    trail_b, = ax.plot([], [], color=color_b, lw=1.2, alpha=0.55)
    rod_a, = ax.plot([], [], color=color_a, lw=2.0)
    rod_b, = ax.plot([], [], color=color_b, lw=2.0)
    bob_a, = ax.plot([], [], "o", color=color_a, markersize=7)
    bob_b, = ax.plot([], [], "o", color=color_b, markersize=7)

    def init():
        for line in (trail_a, trail_b, rod_a, rod_b, bob_a, bob_b):
            line.set_data([], [])
        return trail_a, trail_b, rod_a, rod_b, bob_a, bob_b

    def update(i):
        lo = max(0, i - trail_len)
        trail_a.set_data(xy_a[lo:i + 1, 2], xy_a[lo:i + 1, 3])
        trail_b.set_data(xy_b[lo:i + 1, 2], xy_b[lo:i + 1, 3])
        rod_a.set_data([0, xy_a[i, 0], xy_a[i, 2]], [0, xy_a[i, 1], xy_a[i, 3]])
        rod_b.set_data([0, xy_b[i, 0], xy_b[i, 2]], [0, xy_b[i, 1], xy_b[i, 3]])
        bob_a.set_data([xy_a[i, 2]], [xy_a[i, 3]])
        bob_b.set_data([xy_b[i, 2]], [xy_b[i, 3]])
        return trail_a, trail_b, rod_a, rod_b, bob_a, bob_b

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000 / fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB)")
