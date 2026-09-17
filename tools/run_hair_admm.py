#!/usr/bin/env python3
"""Public one-step entry point for HairADMM.

This wrapper intentionally exposes only the final release configuration:
2 mm uniform segment sampling plus Embree feedback, without Edge energy,
temporal gating, or a second solve pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final one-step HairADMM post-processor."
    )
    parser.add_argument(
        "--objective_tensor_dir",
        type=Path,
        required=True,
        help="directory containing canonical frame_XXXX.npz problem bundles",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument(
        "--output_format", choices=("npz", "obj", "both"), default="both"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    objective_dir = args.objective_tensor_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not objective_dir.is_dir():
        parser.error(f"objective tensor directory does not exist: {objective_dir}")

    # Strict tensor mode embeds every objective input and the body mesh in each
    # frame bundle. case_dir is retained only for the internal solver contract.
    case_dir = objective_dir.parent
    argv = [
        "solve_admm_qp_aligned.py",
        "--case_dir", str(case_dir),
        "--output_dir", str(output_dir),
        "--qp_objective_tensor_dir", str(objective_dir),
        "--qp_tensor_weight_mode", "corrected",
        "--start_frame", str(args.start_frame),
        "--end_frame", str(args.end_frame),
        "--output_format", args.output_format,
        "--collision_standard", "sdf_projection",
        "--sdf_margin_mm", "1",
        "--sdf_uniform_segment_samples",
        "--sdf_max_spacing_mm", "2",
        "--sdf_embree_feedback",
        "--sdf_embree_refine_iters", "2",
        "--sdf_active_set",
        "--sdf_active_band_mm", "1",
        "--sdf_projection_sign", "pseudonormal",
        "--outer_iters", "4",
        "--shape_outer_iters", "4",
        "--inner_iters", "10",
        "--min_inner_iters", "10",
        "--rho", "100000",
        "--w_edge_length", "0",
        "--w_bend_continuity", "0",
        "--w_temporal_correction", "0",
    ]
    if args.force:
        argv.append("--force")

    sys.argv = argv
    from solve_admm_qp_aligned import main as solver_main

    solver_main()


if __name__ == "__main__":
    main()
