"""Render the logistic-map bifurcation fractal as a progressive zoom GIF.

Each frame recomputes the bifurcation diagram on a narrower r-window
centered on a period-doubling region near the edge of chaos. The
self-similar structure becomes visible as the zoom progresses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .physics import bifurcation


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "animations" / "logistic.gif"


def _frame_windows(n_frames: int) -> list[tuple[float, float, float, float]]:
    """Linearly interpolate (log-like) a zoom into r ≈ 3.5699 (onset of chaos)."""
    r_center_start, r_halfwidth_start = 3.3, 0.7
    r_center_end, r_halfwidth_end = 3.5699456, 0.004
    x_center_start, x_halfwidth_start = 0.5, 0.5
    x_center_end, x_halfwidth_end = 0.5, 0.18

    windows = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        # ease the zoom so most frames show the approach (exponential zoom)
        s = 1.0 - (1.0 - t) ** 2
        hw_r = r_halfwidth_start * (r_halfwidth_end / r_halfwidth_start) ** s
        c_r = r_center_start + (r_center_end - r_center_start) * s
        hw_x = x_halfwidth_start * (x_halfwidth_end / x_halfwidth_start) ** s
        c_x = x_center_start + (x_center_end - x_center_start) * s
        windows.append((c_r - hw_r, c_r + hw_r, c_x - hw_x, c_x + hw_x))
    return windows


def render(output_path: Path = OUTPUT) -> Path:
    fps = 20
    n_frames = 120
    windows = _frame_windows(n_frames)

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=90)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("r", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("x", color="#cbd5e1", fontsize=10)

    scatter = ax.scatter([], [], s=0.3, c="#34d399", alpha=0.35, linewidths=0)
    label = ax.text(
        0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top",
        color="#cbd5e1", fontsize=9,
    )

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        label.set_text("")
        return scatter, label

    def update(i):
        r_lo, r_hi, x_lo, x_hi = windows[i]
        rs, xs = bifurcation(r_lo, r_hi, n_r=1400, n_warmup=500, n_record=250)
        mask = (xs >= x_lo) & (xs <= x_hi)
        pts = np.column_stack([rs[mask], xs[mask]])
        scatter.set_offsets(pts)
        ax.set_xlim(r_lo, r_hi)
        ax.set_ylim(x_lo, x_hi)
        label.set_text(f"logistic map    r ∈ [{r_lo:.5f}, {r_hi:.5f}]")
        return scatter, label

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=1000 / fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB)")
