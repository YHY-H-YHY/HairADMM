#!/usr/bin/env python3
"""Generate and solve the public one-frame HairADMM demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(*parts: object) -> None:
    subprocess.run([str(part) for part in parts], check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    case = Path(__file__).resolve().parent
    run(sys.executable, case / "create_case.py")
    run(
        sys.executable,
        root / "tools" / "run_hair_admm.py",
        "--objective_tensor_dir", case / "problem",
        "--output_dir", case / "output",
        "--output_format", "both",
        "--force",
    )
    run(sys.executable, case / "verify_result.py")
    run(
        sys.executable,
        root / "tools" / "preprocess_strands_bin.py",
        "--hair_dir", case / "output" / "hair",
        "--body_dir", case / "body",
        "--out", root / "tools" / "web" / "cache" / "toy_case",
        "--manifest", "frames_toy_case.json",
    )
    print("\nDemo complete.")
    print("Start the viewer with:")
    print(f"  cd {root / 'tools' / 'web'}")
    print(f"  {sys.executable} -m http.server 8888")
    print("Then open:")
    print("  http://localhost:8888/viewer.html?manifest=frames_toy_case.json")


if __name__ == "__main__":
    main()
