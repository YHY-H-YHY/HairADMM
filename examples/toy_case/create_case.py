#!/usr/bin/env python3
"""Generate a tiny, synthetic, redistributable HairADMM problem.

The example contains eight short strands around a triangulated sphere. Several
initial strand samples are deliberately placed inside the sphere so the public
demo exercises both SDF projection and segment collision handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hairs_adaption.qp_objective_bundle import save_qp_objective_bundle


def uv_sphere(radius: float = 0.05, rings: int = 12, sectors: int = 24):
    vertices = [[0.0, 0.0, radius]]
    for ring in range(1, rings):
        theta = np.pi * ring / rings
        for sector in range(sectors):
            phi = 2.0 * np.pi * sector / sectors
            vertices.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    vertices.append([0.0, 0.0, -radius])
    top = 0
    bottom = len(vertices) - 1
    faces = []
    for sector in range(sectors):
        current = 1 + sector
        nxt = 1 + (sector + 1) % sectors
        faces.append([top, current, nxt])
    for ring in range(rings - 2):
        row = 1 + ring * sectors
        next_row = row + sectors
        for sector in range(sectors):
            a = row + sector
            b = row + (sector + 1) % sectors
            c = next_row + sector
            d = next_row + (sector + 1) % sectors
            faces.append([a, c, b])
            faces.append([b, c, d])
    last_row = 1 + (rings - 2) * sectors
    for sector in range(sectors):
        current = last_row + sector
        nxt = last_row + (sector + 1) % sectors
        faces.append([current, bottom, nxt])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for x, y, z in vertices * 100.0:
            output.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces + 1:
            output.write(f"f {a} {b} {c}\n")


def weighted_laplacian(points: np.ndarray, knn: np.ndarray, weights: np.ndarray):
    safe = np.where(knn >= 0, knn, 0)
    neighbor = points[safe]
    valid = knn >= 0
    return points - np.sum(weights[:, :, None] * neighbor * valid[:, :, None], axis=1)


def main() -> None:
    output_root = Path(__file__).resolve().parent
    tensor_dir = output_root / "problem"
    body_dir = output_root / "body"
    tensor_path = tensor_dir / "frame_0000.npz"
    body_path = body_dir / "body_0.obj"

    strand_count = 8
    strand_length = 10
    starts = np.arange(strand_count, dtype=np.int64) * strand_length
    lengths = np.full(strand_count, strand_length, dtype=np.int64)
    body_vertices, body_faces = uv_sphere()

    source = np.empty((strand_count, strand_length, 3), dtype=np.float64)
    initial = np.empty_like(source)
    for strand in range(strand_count):
        phi = 2.0 * np.pi * strand / strand_count
        radial = np.asarray([np.cos(phi), np.sin(phi), 0.0])
        tangent = np.asarray([-np.sin(phi), np.cos(phi), 0.0])
        root = np.asarray([0.018 * np.cos(phi), 0.018 * np.sin(phi), 0.04665])
        for point in range(strand_length):
            t = point / (strand_length - 1)
            source[strand, point] = (
                root
                + radial * (0.032 * t)
                + tangent * (0.003 * np.sin(np.pi * t))
                + np.asarray([0.0, 0.0, 0.050 * t])
            )
            initial[strand, point] = source[strand, point]

        # Alternate strands contain an inward sag after the fixed root. This
        # creates both point penetrations and segment intersections.
        if strand % 2 == 1:
            initial[strand, 2:6, 2] -= np.asarray([0.018, 0.028, 0.024, 0.012])
            initial[strand, 2:6] -= radial[None, :] * 0.006

    source_flat = source.reshape(-1, 3)
    initial_flat = initial.reshape(-1, 3)
    point_count = len(source_flat)
    guide_strands = np.asarray([0, 2, 4, 6], dtype=np.int64)
    guide_ids = np.concatenate([
        np.arange(starts[strand], starts[strand] + strand_length)
        for strand in guide_strands
    ])

    guide_knn = np.empty((len(guide_ids), 2), dtype=np.int64)
    for row, global_id in enumerate(guide_ids):
        strand = global_id // strand_length
        local = global_id % strand_length
        position = int(np.where(guide_strands == strand)[0][0])
        left = guide_strands[(position - 1) % len(guide_strands)]
        right = guide_strands[(position + 1) % len(guide_strands)]
        guide_knn[row] = [left * strand_length + local, right * strand_length + local]
    guide_weights = np.full(guide_knn.shape, 0.5, dtype=np.float64)
    guide_lap = weighted_laplacian(source_flat[guide_ids], np.searchsorted(guide_ids, guide_knn), guide_weights)

    normal_knn = np.empty((point_count, 2), dtype=np.int64)
    for global_id in range(point_count):
        strand = global_id // strand_length
        local = global_id % strand_length
        distances = np.minimum(
            (guide_strands - strand) % strand_count,
            (strand - guide_strands) % strand_count,
        )
        nearest = guide_strands[np.argsort(distances)[:2]]
        normal_knn[global_id] = nearest * strand_length + local
    normal_weights = np.full(normal_knn.shape, 0.5, dtype=np.float64)
    normal_lap = weighted_laplacian(source_flat, normal_knn, normal_weights)

    local_index = np.tile(np.arange(strand_length), strand_count)
    local_weights = np.clip(local_index / (strand_length - 1), 0.05, 1.0)
    roots = source[:, 0].copy()
    save_qp_objective_bundle(
        tensor_path,
        local_weight_mode=np.asarray("corrected"),
        initial_transfer_pos=initial_flat,
        source_aligned=source_flat,
        hair_starts=starts,
        hair_lengths=lengths,
        guide_strand_indices=guide_strands,
        guide_knn_indices=guide_knn,
        guide_knn_weights=guide_weights,
        guide_ori_laplacian=guide_lap,
        normal_knn_indices=normal_knn,
        normal_knn_weights=normal_weights,
        normal_ori_laplacian=normal_lap,
        fit_local_weights=local_weights,
        lap_local_weights=local_weights,
        source_mesh_roots=roots,
        target_mesh_roots=roots,
        target_body_vertices=body_vertices,
        target_body_faces=body_faces,
        w_fid=np.asarray(1000.0),
        w_lap=np.asarray(3000.0),
        w_lap_normal=np.asarray(30000.0),
        use_shape=np.asarray(True),
        use_fit=np.asarray(True),
        use_lap=np.asarray(True),
    )
    write_obj(body_path, body_vertices, body_faces)
    print(f"Wrote {tensor_path.relative_to(ROOT)}")
    print(f"Wrote {body_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
