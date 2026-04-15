"""Render Kirkwood-gap formation as an emerging histogram over simulation time.

~1500 massless test particles seeded uniformly in semi-major axis across
the main asteroid belt, integrated under Sun + Jupiter gravity (CR3BP).
As the integration runs, resonance-pumped eccentricity destabilizes the
particles near the p:q mean-motion resonances with Jupiter, and gaps
carve themselves out of the histogram at the Kirkwood locations
(3:1 ≈ 2.50 AU, 5:2 ≈ 2.82 AU, 7:3 ≈ 2.96 AU, 2:1 ≈ 3.28 AU).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .physics import (
    A_JUP,
    JUP_MASS_RATIO,
    MU_SUN,
    N_JUP,
    KirkwoodParams,
    initial_conditions,
    resonance_locations,
    rk4_step,
    semi_major_axis,
)


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "animations" / "kirkwood.gif"


def simulate(
    n_particles: int = 1200,
    a_min: float = 2.0,
    a_max: float = 3.5,
    e0: float = 0.12,
    dt: float = 0.08,
    t_total: float = 3000.0,
    n_snapshots: int = 90,
    seed: int = 1,
    mu_jup_boost: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the CR3BP integration and return (snapshot_times, a_history).

    `mu_jup_boost` multiplies Jupiter's mass relative to the real value, to
    accelerate resonance-driven chaos onto a short integration window (the
    canonical Kirkwood gaps emerge over ~10^4–10^6 yr at real mass; we use
    a boosted mass so the effect is visible in a few thousand simulated
    years — a standard didactic trick, not a claim about the real solar
    system).

    a_history has shape (n_snapshots, n_particles); entries are NaN for
    particles that have been ejected (a < 0 in the vis-viva sense) at that
    snapshot.
    """
    p = KirkwoodParams(
        mu_sun=MU_SUN,
        mu_jup=MU_SUN * JUP_MASS_RATIO * mu_jup_boost,
        a_jup=A_JUP,
        n_jup=N_JUP,
    )
    a0, x, y, vx, vy = initial_conditions(
        n_particles=n_particles, a_min=a_min, a_max=a_max, e=e0, seed=seed, p=p
    )

    n_steps = int(round(t_total / dt))
    snap_every = max(1, n_steps // n_snapshots)
    n_snaps = n_steps // snap_every + 1

    a_hist = np.empty((n_snaps, n_particles), dtype=float)
    t_hist = np.empty(n_snaps, dtype=float)

    a_hist[0] = a0
    t_hist[0] = 0.0
    snap_idx = 1
    t = 0.0
    for step in range(1, n_steps + 1):
        x, y, vx, vy = rk4_step(t, x, y, vx, vy, dt, p)
        t += dt
        if step % snap_every == 0 and snap_idx < n_snaps:
            a_hist[snap_idx] = semi_major_axis(x, y, vx, vy, p)
            t_hist[snap_idx] = t
            snap_idx += 1

    return t_hist[:snap_idx], a_hist[:snap_idx]


def render(output_path: Path = OUTPUT) -> Path:
    t_hist, a_hist = simulate()
    # Resonance locations depend only on Jupiter's a_J, not its mass —
    # so the canonical values are independent of the mass boost.
    resonances = resonance_locations()

    a_lo, a_hi = 2.0, 3.5
    n_bins = 90
    bin_edges = np.linspace(a_lo, a_hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=90)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("semi-major axis  a  (AU)", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("number of particles", color="#cbd5e1", fontsize=10)
    ax.set_xlim(a_lo, a_hi)

    init_counts, _ = np.histogram(a_hist[0], bins=bin_edges)
    y_max = init_counts.max() * 1.25
    ax.set_ylim(0, y_max)

    bars = ax.bar(
        bin_centers, init_counts, width=bin_width,
        color="#fbbf24", edgecolor="none", alpha=0.85,
    )

    for label, a_res in resonances.items():
        if a_lo <= a_res <= a_hi:
            ax.axvline(a_res, color="#f87171", lw=1.0, alpha=0.55, linestyle="--")
            ax.text(
                a_res, y_max * 0.08, label,
                color="#fca5a5", fontsize=8,
                ha="center", va="bottom",
            )

    ax.text(
        0.02, 0.96, "Kirkwood gaps  (CR3BP, Sun + Jupiter)",
        transform=ax.transAxes, ha="left", va="top",
        color="#cbd5e1", fontsize=10,
    )
    time_label = ax.text(
        0.98, 0.88, "", transform=ax.transAxes, ha="right", va="top",
        color="#cbd5e1", fontsize=10,
    )

    def init():
        for bar, h in zip(bars, init_counts):
            bar.set_height(h)
        time_label.set_text(f"t = 0 yr")
        return (*bars, time_label)

    def update(i):
        counts, _ = np.histogram(a_hist[i], bins=bin_edges)
        for bar, h in zip(bars, counts):
            bar.set_height(h)
        time_label.set_text(f"t = {t_hist[i]:,.0f} yr")
        return (*bars, time_label)

    fps = 15
    anim = FuncAnimation(
        fig, update, frames=len(t_hist), init_func=init,
        blit=False, interval=1000 / fps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = render()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB)")
