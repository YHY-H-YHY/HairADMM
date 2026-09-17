#!/usr/bin/env python3
"""Verify that the synthetic demo exercises and resolves collisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    case = Path(__file__).resolve().parent
    problem = np.load(case / "problem" / "frame_0000.npz")
    result = np.load(case / "output" / "hair" / "frame_0000.npz")
    metrics = json.loads(
        (case / "output" / "metrics" / "frame_0000.json").read_text()
    )

    initial = problem["initial_transfer_pos"].reshape(8, 10, 3)
    optimized = np.asarray(result["rods"], dtype=np.float64)
    initial_inside = int((np.linalg.norm(initial[:, 1:], axis=2) < 0.05).sum())
    result_inside = int((np.linalg.norm(optimized[:, 1:], axis=2) < 0.05).sum())
    normal_history = metrics["normal_history"]
    initial_hits = max(item["initial_segment_intersections"] for item in normal_history)
    final_hits = normal_history[-1]["segment_intersections"]

    if initial_inside <= 0 or initial_hits <= 0:
        raise RuntimeError("toy input does not exercise point and segment collisions")
    if result_inside != 0 or final_hits != 0:
        raise RuntimeError(
            f"toy result still collides: inside={result_inside}, segments={final_hits}"
        )

    print(
        "Verified toy collision correction: "
        f"inside points {initial_inside} -> {result_inside}, "
        f"segment intersections {initial_hits} -> {final_hits}."
    )


if __name__ == "__main__":
    main()
