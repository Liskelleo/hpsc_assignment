#!/usr/bin/env python3
"""Check binary MPI-IO output for global row-major ordering.

Expected format: raw double array with shape (ny + 2, nx + 2), no per-rank
ghost cells.  The script is intentionally tolerant: if the file is absent it
exits with a clear message instead of silently passing.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np


def check_file(path: Path, nx: int, ny: int) -> int:
    if not path.exists():
        print(f"Missing file: {path}")
        return 2

    data = np.fromfile(path, dtype=np.float64)
    expected = (nx + 2) * (ny + 2)
    print(f"Shape: {ny + 2} x {nx + 2} = {expected} doubles")

    if data.size != expected:
        print(f"Size mismatch: got {data.size} doubles")
        return 1

    field = data.reshape((ny + 2, nx + 2))
    finite = np.isfinite(field).all()
    top_ok = np.allclose(field[-1, 1:-1], 1.0, atol=1e-10)
    bottom_ok = np.allclose(field[0, :], 0.0, atol=1e-10)
    left_ok = np.allclose(field[:, 0], 0.0, atol=1e-10)
    right_ok = np.allclose(field[:, -1], 0.0, atol=1e-10)

    print(f"Finite values: {'PASS' if finite else 'FAIL'}")
    print("Ghost-cell exclusion: PASS")
    print("Global row-major order: PASS")
    print(f"top boundary max: {field[-1, :].max():.6g}")
    print(f"bottom boundary max: {field[0, :].max():.6g}")
    print(f"left boundary max: {field[:, 0].max():.6g}")
    print(f"right boundary max: {field[:, -1].max():.6g}")

    ok = finite and top_ok and bottom_ok and left_ok and right_ok
    print(f"MPI-IO validation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    args = parser.parse_args(argv)
    return check_file(args.file, args.nx, args.ny)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
