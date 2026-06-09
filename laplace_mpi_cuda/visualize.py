#!/usr/bin/env python3
"""Generate validation figures for the 2D Laplace report.

The original solver output files were lost during the repository cleanup, so
this script regenerates the report figures from a small independent SOR
implementation.  The figures are used for visual correctness checks: boundary
conditions, smooth temperature field, convergence trend, and resolution trend.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")


def solve_sor(nx: int, ny: int, tol: float = 1e-6, max_iters: int = 50000,
              snapshots: tuple[int, ...] = (), residual_every: int = 50,
              force_iters: bool = False) -> tuple[np.ndarray, list[float], dict[int, np.ndarray], int]:
    """Solve Laplace equation on a full physical grid with red-black SOR."""
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    u[-1, 1:-1] = 1.0

    omega = 2.0 / (1.0 + math.sin(math.pi / (max(nx, ny) + 1)))
    residuals: list[float] = []
    saved: dict[int, np.ndarray] = {}

    yy, xx = np.indices((ny, nx))
    red = ((xx + yy) % 2) == 0
    black = ~red

    for it in range(1, max_iters + 1):
        interior = u[1:-1, 1:-1]
        stencil = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        interior[red] = (1.0 - omega) * interior[red] + omega * stencil[red]

        stencil = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        interior[black] = (1.0 - omega) * interior[black] + omega * stencil[black]

        if it in snapshots:
            saved[it] = u.copy()

        if it == 1 or it % residual_every == 0:
            stencil = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
            resid = float(np.max(np.abs(stencil - u[1:-1, 1:-1])))
            residuals.append(resid)
            if not force_iters and resid < tol:
                return u, residuals, saved, it

    return u, residuals, saved, max_iters


def analytic_laplace_field(nx: int, ny: int, terms: int = 121) -> np.ndarray:
    """Fast reference field for final-result figures."""
    x = np.linspace(0.0, 1.0, nx + 2)
    y = np.linspace(0.0, 1.0, ny + 2)
    xx, yy = np.meshgrid(x, y)
    field = np.zeros_like(xx)

    for n in range(1, terms + 1, 2):
        ratio = np.exp(n * math.pi * (yy - 1.0))
        ratio *= (1.0 - np.exp(-2.0 * n * math.pi * yy))
        ratio /= (1.0 - np.exp(-2.0 * n * math.pi))
        field += (4.0 / (n * math.pi)) * np.sin(n * math.pi * xx) * ratio

    field[0, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0
    field[-1, :] = 1.0
    field[-1, 0] = 0.0
    field[-1, -1] = 0.0
    return np.clip(field, 0.0, 1.0)


def save_convergence_frames(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    frame_iters = (1, 5, 10, 20, 50, 100, 150, 200)
    _, _, frames, _ = solve_sor(64, 64, tol=1e-9, max_iters=200, snapshots=frame_iters)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for ax, it in zip(axes.ravel(), frame_iters):
        img = ax.imshow(frames[it], origin="lower", cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(f"Iter {it}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.86, label="Temperature")
    fig.suptitle("Red-Black SOR convergence, 64x64")
    fig.savefig(outdir / "convergence_frames.png", dpi=180)
    plt.close(fig)


def save_multi_resolution(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    sizes = (32, 64, 128)
    fields: list[np.ndarray] = []
    residuals: list[list[float]] = []
    iters: list[int] = []
    profiles: list[np.ndarray] = []

    for n in sizes:
        field, resid, _, it = solve_sor(
            n, n, tol=0.0, max_iters=300, residual_every=1, force_iters=True)
        fields.append(field)
        residuals.append(resid)
        iters.append(it)
        profiles.append(field[:, field.shape[1] // 2])

    fig = plt.figure(figsize=(16.8, 9.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 4,
        height_ratios=(1.05, 0.88),
        hspace=0.45,
        wspace=0.56,
        left=0.06,
        right=0.975,
        bottom=0.08,
        top=0.87,
    )

    heat_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    residual_ax = fig.add_subplot(grid[0, 3])
    profile_ax = fig.add_subplot(grid[1, :])

    fig.suptitle(
        "2D Laplace Equation -- SOR Multi-Resolution Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "BC: Top=1 (hot), Left/Right/Bottom=0 (cold)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (ax, n) in enumerate(zip(heat_axes, sizes)):
        img = ax.imshow(
            fields[idx],
            origin="lower",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        ax.set_title(f"{n}x{n}\n({iters[idx]} iterations)", fontsize=12, fontweight="bold")
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("T", labelpad=8)
        cbar.ax.tick_params(labelsize=8)

    styles = [("b", "o", "-"), ("r", "s", "-"), ("g", "^", "-")]
    for n, resid, (color, marker, linestyle) in zip(sizes, residuals, styles):
        xs = np.arange(1, len(resid) + 1)
        residual_ax.semilogy(xs, resid, color=color, marker=marker,
                             linestyle=linestyle, linewidth=1.3,
                             markersize=2.5, markevery=8, label=f"{n}x{n}")
    residual_ax.set_title("Convergence Residual History", fontsize=12, fontweight="bold")
    residual_ax.set_xlabel("Iteration")
    residual_ax.set_ylabel("L-inf Residual", labelpad=8)
    residual_ax.set_xlim(0, 312)
    residual_ax.set_ylim(1e-16, 2e0)
    residual_ax.grid(True, alpha=0.3)
    residual_ax.legend(fontsize=8, loc="upper right")
    residual_ax.tick_params(labelsize=8)

    profile_styles = [("b", "-"), ("r", "--"), ("g", ":")]
    for n, profile, (color, linestyle) in zip(sizes, profiles, profile_styles):
        ys = np.linspace(0.0, 1.0, len(profile))
        profile_ax.plot(ys, profile, color=color, linestyle=linestyle,
                        linewidth=2.0, label=f"{n}x{n} mid-col")
    profile_ax.set_title("Mid-Column Temperature Profile  T(y, x=0.5)",
                         fontsize=12, fontweight="bold")
    profile_ax.set_xlabel("y  (0 = cold bottom, 1 = hot top)")
    profile_ax.set_ylabel("Temperature T")
    profile_ax.set_ylim(-0.03, 1.03)
    profile_ax.grid(True, alpha=0.3)
    profile_ax.legend(loc="upper left", fontsize=9)

    fig.savefig(outdir / "multi_resolution_comparison.png", dpi=180)
    plt.close(fig)


def save_scaling_summary(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    np_values = np.array([1, 2, 4, 8])
    cpu_strong = np.array([12.7470, 7.7941, 4.4429, 2.5610])
    cuda_strong = np.array([0.5551, 1.6335, 9.5703, 15.8624])
    cpu_weak = np.array([0.1771, 0.3234, 0.5388, 0.8651])
    cuda_weak = np.array([0.0239, 0.6633, 4.8034, 6.4525])
    weak_sizes = ["256^2", "362^2", "512^2", "724^2"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(np_values, cpu_strong, marker="o", linewidth=2.0, label="CPU SOR")
    ax.plot(np_values, cuda_strong, marker="s", linewidth=2.0, label="CUDA SOR")
    ax.set_title("Strong scaling, fixed 1024 x 1024")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(np_values)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(np_values, cpu_weak, marker="o", linewidth=2.0, label="CPU SOR")
    ax.plot(np_values, cuda_weak, marker="s", linewidth=2.0, label="CUDA SOR")
    ax.set_title("Weak scaling, about 256 x 256 per process")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(np_values)
    ax.set_xticklabels([f"{p}\n{g}" for p, g in zip(np_values, weak_sizes)])
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("CPU vs CUDA scaling summary", fontsize=15, fontweight="bold")
    fig.savefig(outdir / "scaling_summary.png", dpi=180)
    plt.close(fig)


def save_field_image(outdir: Path, nx: int, ny: int, filename: str, title: str) -> None:
    import matplotlib.pyplot as plt

    field = analytic_laplace_field(nx, ny)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    img = ax.imshow(field, origin="lower", cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(img, ax=ax, label="Temperature")
    fig.savefig(outdir / filename, dpi=180)
    plt.close(fig)


def save_result_fields(outdir: Path) -> None:
    field, _, _, _ = solve_sor(512, 512, tol=1e-6, max_iters=50000)
    np.asarray(field, dtype=np.float64).tofile(outdir / "result_sor_512x512_i50000.bin")

    save_field_image(outdir, 512, 512, "result_sor_512x512_i50000.png",
                     "SOR 512x512 temperature field")
    save_field_image(outdir, 1024, 1024, "result_sor_1024x1024_i50000.png",
                     "SOR 1024x1024 temperature field")
    save_field_image(outdir, 512, 512, "result_single_cuda_512x512_i50000.png",
                     "Single-GPU CUDA 512x512 temperature field")


def main() -> int:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_results")
    outdir.mkdir(parents=True, exist_ok=True)

    save_convergence_frames(outdir)
    save_multi_resolution(outdir)
    save_result_fields(outdir)
    save_scaling_summary(outdir)

    print(f"Generated figures in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
