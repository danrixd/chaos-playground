"""Render the Lorenz attractor as a rotating 3D trajectory.

A single trajectory traces out the butterfly. The camera azimuth rotates
slowly so the 3D shape reads clearly in a 2D GIF. A leading head marker
shows where "now" is along the curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from chaos_playground.shared.integrator import integrate

from .physics import LorenzParams, derivatives


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "animations" / "lorenz.gif"


def render(output_path: Path = OUTPUT) -> Path:
    params = LorenzParams()

    def f(t, y):
        return derivatives(t, y, params)

    y0 = np.array([1.0, 1.0, 1.0])
    dt = 5e-3
    t_total = 40.0
    n_steps = int(t_total / dt)
    _ts, ys = integrate(f, y0, dt, n_steps)

    fps = 30
    n_frames = 300
    stride = max(1, ys.shape[0] // n_frames)
    ys = ys[::stride]
    n_frames = ys.shape[0]

    fig = plt.figure(figsize=(5.4, 5.0), dpi=90)
    fig.patch.set_facecolor("#0b0f14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0f14")
    ax.set_axis_off()

    xs, ys_, zs = ys[:, 0], ys[:, 1], ys[:, 2]
    ax.set_xlim(xs.min() - 2, xs.max() + 2)
    ax.set_ylim(ys_.min() - 2, ys_.max() + 2)
    ax.set_zlim(zs.min() - 2, zs.max() + 2)

    line, = ax.plot([], [], [], color="#a78bfa", lw=0.9, alpha=0.85)
    head, = ax.plot([], [], [], "o", color="#fde68a", markersize=5)

    title = ax.text2D(
        0.02, 0.96, r"Lorenz  $\sigma=10,\ \rho=28,\ \beta=8/3$",
        transform=ax.transAxes, color="#cbd5e1", fontsize=10,
    )
    _ = title

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        head.set_data([], [])
        head.set_3d_properties([])
        return line, head

    def update(i):
        line.set_data(xs[: i + 1], ys_[: i + 1])
        line.set_3d_properties(zs[: i + 1])
        head.set_data([xs[i]], [ys_[i]])
        head.set_3d_properties([zs[i]])
        ax.view_init(elev=22, azim=(i / n_frames) * 360.0)
        return line, head

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=1000 / fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB)")
