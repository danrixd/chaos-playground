"""Side-by-side regular-vs-chaotic regimes of the double pendulum.

At small initial angles the double pendulum is nearly integrable: the
motion is a quasi-periodic beating of the two normal modes and a single
bob traces a neat Lissajous-like figure. Push the initial angles past
roughly 2 rad and the same system becomes chaotic — the trajectory never
repeats and fills an irregular region of configuration space. Both
panels use identical physics parameters and identical integration; the
only difference is the magnitude of the initial angles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from chaos_playground.shared.integrator import integrate

from .physics import DoublePendulumParams, cartesian, derivatives


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "animations" / "double_pendulum_regimes.gif"


def _simulate(params: DoublePendulumParams, y0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    def f(t, y):
        return derivatives(t, y, params)

    _ts, ys = integrate(f, y0, dt, n_steps)
    return ys


def _total_energy(ys: np.ndarray, p: DoublePendulumParams) -> float:
    theta1, omega1, theta2, omega2 = ys[0]
    m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g
    t = 0.5 * (m1 + m2) * l1 ** 2 * omega1 ** 2 + 0.5 * m2 * l2 ** 2 * omega2 ** 2
    t += m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    v = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    return t + v


def render(output_path: Path = OUTPUT) -> Path:
    params = DoublePendulumParams()
    dt_sim = 1.0 / 600.0
    t_total = 14.0
    n_sim = int(t_total / dt_sim)

    # Regular regime: small initial angles, zero initial velocity.
    y0_regular = np.array([0.35, 0.0, 0.30, 0.0])
    # Chaotic regime: large initial angles, zero initial velocity.
    y0_chaotic = np.array([2.6, 0.0, 2.7, 0.0])

    ys_r = _simulate(params, y0_regular, dt_sim, n_sim)
    ys_c = _simulate(params, y0_chaotic, dt_sim, n_sim)
    xy_r = cartesian(ys_r, params)
    xy_c = cartesian(ys_c, params)

    e_r = _total_energy(ys_r, params)
    e_c = _total_energy(ys_c, params)

    fps = 30
    stride = max(1, int(round(1.0 / (fps * dt_sim))))
    xy_r = xy_r[::stride]
    xy_c = xy_c[::stride]
    n_frames = xy_r.shape[0]
    trail_len = 140

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5), dpi=90)
    fig.patch.set_facecolor("#0b0f14")

    r = params.l1 + params.l2 + 0.2
    panels = [
        (axes[0], xy_r, "#34d399", "regular", f"E = {e_r:.2f} J", 0.9),
        (axes[1], xy_c, "#f472b6", "chaotic", f"E = {e_c:.2f} J", 1.0),
    ]

    artists = []
    for ax, _xy, color, label, energy_label, trail_alpha in panels:
        ax.set_facecolor("#0b0f14")
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.03, 0.97, label,
            transform=ax.transAxes, ha="left", va="top",
            color="#e2e8f0", fontsize=12, fontweight="bold",
        )
        ax.text(
            0.03, 0.90, energy_label,
            transform=ax.transAxes, ha="left", va="top",
            color="#94a3b8", fontsize=9,
        )
        trail, = ax.plot([], [], color=color, lw=1.1, alpha=trail_alpha * 0.55)
        rod, = ax.plot([], [], color=color, lw=2.0)
        bob, = ax.plot([], [], "o", color=color, markersize=7)
        artists.append((trail, rod, bob))

    def init():
        out = []
        for trail, rod, bob in artists:
            trail.set_data([], [])
            rod.set_data([], [])
            bob.set_data([], [])
            out.extend([trail, rod, bob])
        return out

    def update(i):
        out = []
        for (trail, rod, bob), xy in zip(artists, (xy_r, xy_c)):
            lo = max(0, i - trail_len)
            trail.set_data(xy[lo:i + 1, 2], xy[lo:i + 1, 3])
            rod.set_data([0, xy[i, 0], xy[i, 2]], [0, xy[i, 1], xy[i, 3]])
            bob.set_data([xy[i, 2]], [xy[i, 3]])
            out.extend([trail, rod, bob])
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000 / fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB)")
