#!/usr/bin/env python3
"""Internal two-stage engine for the public one-step HairADMM solver."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from hairs_adaption.qp_objective_bundle import (
    file_sha256,
    load_qp_objective_bundle,
    to_padded_objective,
)

from admm_qp_common import (
    bending_matrix,
    collision_matrix,
    collision_planes,
    difference_matrix,
    solve_spd,
    topology,
    L_MAX,
    _compute_invdist_weights_and_lap,
    _compute_per_point_weights,
    _fast_read_obj,
    _get_mesh_roots,
    _load_guide_data,
    _parse_hair_obj,
    _prepare_body_qpdetect,
    build_padded_rods,
    load_knn,
    write_hair_obj,
)


@dataclass(frozen=True)
class AdaptiveCollisionSample:
    edge: int
    fraction: float
    plane_point: np.ndarray
    plane_normal: np.ndarray


def coherent_segment_constraints(
    x: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    edge_strand: np.ndarray,
    edge_local: np.ndarray,
    intersector,
    face_normals: np.ndarray,
    margin: float,
    cluster_gap: int = 5,
) -> tuple[int, list[AdaptiveCollisionSample], int]:
    """Build deterministic neighborhood samples for intersecting segments.

    The formal SDF path consumes only ``edge`` and ``fraction`` from each
    sample.  Plane metadata remains only for the non-formal legacy branch and
    is never added to the formal ADMM constraint operator.
    """
    if not len(edge_start):
        return 0, [], 0
    edge = x[edge_end] - x[edge_start]
    length = np.linalg.norm(edge, axis=1)
    valid = length > 1e-10
    if not valid.any():
        return 0, [], 0
    valid_edges = np.where(valid)[0]
    direction = edge[valid] / length[valid, None]
    locations, ray_ids, triangle_ids = intersector.intersects_location(
        x[edge_start[valid]], direction, multiple_hits=True
    )
    if not len(locations):
        return 0, [], 0
    actual_edges = valid_edges[ray_ids]
    distance = np.einsum(
        "ij,ij->i",
        locations - x[edge_start[actual_edges]],
        direction[ray_ids],
    )
    inside = (distance > 1e-7) & (
        distance < length[actual_edges] - 1e-7
    )
    if not inside.any():
        return 0, [], 0
    actual_edges = actual_edges[inside]
    locations = np.asarray(locations[inside], dtype=np.float64)
    triangle_ids = np.asarray(triangle_ids[inside], dtype=np.int64)
    fractions = distance[inside] / length[actual_edges]
    hit_count = int(len(np.unique(actual_edges)))

    records = sorted(
        zip(actual_edges, fractions, locations, triangle_ids),
        key=lambda item: (
            int(edge_strand[item[0]]), int(edge_local[item[0]])
        ),
    )
    clusters = []
    current = []
    previous_strand = None
    previous_local = None
    for record in records:
        edge_id = int(record[0])
        strand = int(edge_strand[edge_id])
        local = int(edge_local[edge_id])
        if (
            current
            and (
                strand != previous_strand
                or local - previous_local > cluster_gap
            )
        ):
            clusters.append(current)
            current = []
        current.append(record)
        previous_strand = strand
        previous_local = local
    if current:
        clusters.append(current)

    samples = []
    conflict_clusters = 0
    for cluster in clusters:
        normals = np.asarray(
            [face_normals[int(record[3])] for record in cluster],
            dtype=np.float64,
        )
        if len(normals) > 1 and float(np.min(normals @ normals.T)) < -0.5:
            conflict_clusters += 1
        affected_edges = np.unique([int(record[0]) for record in cluster])
        affected_points = np.unique(np.r_[
            edge_start[affected_edges], edge_end[affected_edges]
        ])
        points = x[affected_points]
        costs = []
        for record, normal in zip(cluster, normals):
            gap = (points - record[2]) @ normal
            correction = np.maximum(margin - gap, 0.0)
            costs.append(float(np.dot(correction, correction)))
        selected = int(np.argmin(costs))
        plane_point = np.asarray(cluster[selected][2], dtype=np.float64)
        plane_normal = normals[selected]

        for edge_id in affected_edges:
            candidates = [
                (record, normal)
                for record, normal in zip(cluster, normals)
                if int(record[0]) == int(edge_id)
            ]
            alignment = [float(normal @ plane_normal) for _, normal in candidates]
            record = candidates[int(np.argmax(alignment))][0]
            center = float(record[1])
            for offset in (-0.10, 0.0, 0.10):
                samples.append(AdaptiveCollisionSample(
                    edge=int(edge_id),
                    fraction=float(np.clip(center + offset, 0.02, 0.98)),
                    plane_point=plane_point,
                    plane_normal=plane_normal,
                ))
    unique = {}
    for sample in samples:
        key = (sample.edge, int(round(sample.fraction * 10000)))
        unique.setdefault(key, sample)
    return hit_count, list(unique.values()), conflict_clusters


def suppress_opposing_collision_rows(
    plane_normals: np.ndarray,
    active: np.ndarray,
    sample_kind: np.ndarray,
    sample_strand: np.ndarray,
    sample_local: np.ndarray,
    cluster_gap: float = 5.0,
) -> tuple[int, int]:
    """Prefer an Embree escape side over nearby opposing local planes."""
    adaptive_rows = np.flatnonzero((sample_kind == "adaptive") & active)
    if not len(adaptive_rows):
        return 0, 0
    order = adaptive_rows[np.lexsort((
        sample_local[adaptive_rows], sample_strand[adaptive_rows]
    ))]
    clusters: list[np.ndarray] = []
    current = []
    previous_strand = None
    previous_local = None
    for row in order:
        strand = int(sample_strand[row])
        local = float(sample_local[row])
        if (
            current
            and (
                strand != previous_strand
                or local - previous_local > cluster_gap
            )
        ):
            clusters.append(np.asarray(current, dtype=np.int64))
            current = []
        current.append(int(row))
        previous_strand = strand
        previous_local = local
    if current:
        clusters.append(np.asarray(current, dtype=np.int64))

    conflict_clusters = 0
    suppressed_rows = 0
    for rows in clusters:
        escape_normal = plane_normals[rows[0]]
        strand = int(sample_strand[rows[0]])
        local_min = float(np.min(sample_local[rows])) - 1.0
        local_max = float(np.max(sample_local[rows])) + 1.0
        nearby = np.flatnonzero(
            active
            & (sample_kind != "adaptive")
            & (sample_strand == strand)
            & (sample_local >= local_min)
            & (sample_local <= local_max)
        )
        if not len(nearby):
            continue
        opposing = nearby[
            (plane_normals[nearby] @ escape_normal) < -0.5
        ]
        if len(opposing):
            conflict_clusters += 1
            active[opposing] = False
            suppressed_rows += len(opposing)
    return conflict_clusters, suppressed_rows


def frame_id(path: Path) -> int:
    return int(re.search(r"\d+", path.stem).group())


def sparse_laplacian(knn: np.ndarray, weights: np.ndarray) -> sp.csr_matrix:
    point_count, neighbor_count = knn.shape
    rows, cols, data = [], [], []
    for neighbor in range(neighbor_count):
        valid = (
            (knn[:, neighbor] >= 0)
            & (knn[:, neighbor] < point_count)
            & (weights[:, neighbor] > 1e-7)
        )
        source = np.where(valid)[0]
        rows.append(source)
        cols.append(knn[source, neighbor])
        data.append(-weights[source, neighbor])
    diagonal = np.arange(point_count)
    rows.append(diagonal)
    cols.append(diagonal)
    data.append(np.sum(weights, axis=1))
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(point_count, point_count),
        dtype=np.float64,
    )


def cross_strand_knn(
    points: np.ndarray, starts: np.ndarray, lengths: np.ndarray, k: int
) -> np.ndarray:
    point_count = len(points)
    strand_id = np.empty(point_count, dtype=np.int32)
    for strand, (start, length) in enumerate(zip(starts, lengths)):
        strand_id[int(start): int(start) + int(length)] = strand
    tree = cKDTree(points)
    max_length = int(np.max(lengths))
    query_k = min(max(k * (max_length + 1), 64), point_count)
    _, candidates = tree.query(points, k=query_k)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    result = np.full((point_count, k), -1, dtype=np.int64)
    for point in range(point_count):
        other = candidates[point][strand_id[candidates[point]] != strand_id[point]]
        if len(other) < k:
            raise RuntimeError(f"point {point} has fewer than {k} cross-strand neighbors")
        result[point] = other[:k]
    return result


def compact_subset(
    strand_ids: np.ndarray,
    padded: np.ndarray,
    source_padded: np.ndarray,
    lengths: np.ndarray,
    weight_padded: np.ndarray,
    target_padded: np.ndarray | None = None,
):
    subset_lengths = lengths[strand_ids].astype(np.int32)
    starts = np.zeros(len(strand_ids), dtype=np.int64)
    if len(starts) > 1:
        starts[1:] = np.cumsum(subset_lengths[:-1])
    global_ids = []
    for strand, length in zip(strand_ids, subset_lengths):
        base = int(strand) * L_MAX
        global_ids.extend(range(base, base + int(length)))
    global_ids = np.asarray(global_ids, dtype=np.int64)
    result = {
        "strand_ids": strand_ids,
        "starts": starts,
        "lengths": subset_lengths,
        "global_ids": global_ids,
        "init": padded.reshape(-1, 3)[global_ids],
        "source": source_padded.reshape(-1, 3)[global_ids],
        "weights": weight_padded[global_ids],
    }
    if target_padded is not None:
        result["lap_target"] = target_padded.reshape(-1, 3)[global_ids]
    return result


def shape_system(
    x: np.ndarray,
    source: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    edge = x[edge_end] - x[edge_start]
    length = np.maximum(np.linalg.norm(edge, axis=1), 1e-5)
    source_edge = source[edge_end] - source[edge_start]
    source_length = np.linalg.norm(source_edge, axis=1)
    direction = np.zeros_like(source_edge)
    valid_source = source_length >= 1e-8
    direction[valid_source] = (
        source_edge[valid_source] / source_length[valid_source, None]
    )

    short_weight = np.ones(len(length), dtype=np.float64)
    short = length < 3e-4
    short_weight[short] = length[short] ** 2 / 9e-8
    raw_length = length / short_weight
    coefficient = 1.0 / (length * raw_length)

    rows = np.concatenate([edge_start, edge_end, edge_start, edge_end])
    cols = np.concatenate([edge_start, edge_end, edge_end, edge_start])
    values = np.concatenate(
        [coefficient, coefficient, -coefficient, -coefficient]
    )
    hessian = sp.csr_matrix(
        (values, (rows, cols)), shape=(len(x), len(x)), dtype=np.float64
    )
    rhs = np.zeros_like(x)
    linear = direction / raw_length[:, None]
    np.subtract.at(rhs, edge_start, linear)
    np.add.at(rhs, edge_end, linear)
    return hessian, rhs, direction


def regularity_system(
    x_init: np.ndarray,
    source: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    bend_prev: np.ndarray,
    bend_mid: np.ndarray,
    bend_next: np.ndarray,
    w_edge: float,
    w_bend: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    point_count = len(x_init)
    hessian = sp.csr_matrix((point_count, point_count), dtype=np.float64)
    rhs = np.zeros_like(x_init)
    if w_edge > 0 and len(edge_start):
        edge_matrix = difference_matrix(point_count, edge_start, edge_end)
        init_edge = x_init[edge_end] - x_init[edge_start]
        init_length = np.linalg.norm(init_edge, axis=1)
        source_edge = source[edge_end] - source[edge_start]
        source_direction = source_edge / np.maximum(
            np.linalg.norm(source_edge, axis=1)[:, None], 1e-12
        )
        edge_target = init_length[:, None] * source_direction
        hessian = hessian + w_edge * (edge_matrix.T @ edge_matrix)
        rhs += w_edge * (edge_matrix.T @ edge_target)
    if w_bend > 0 and len(bend_prev):
        bend = bending_matrix(
            point_count, bend_prev, bend_mid, bend_next
        )
        init_prev = x_init[bend_mid] - x_init[bend_prev]
        init_next = x_init[bend_next] - x_init[bend_mid]
        source_prev = source[bend_mid] - source[bend_prev]
        source_next = source[bend_next] - source[bend_mid]
        target_prev = (
            np.linalg.norm(init_prev, axis=1)[:, None]
            * source_prev
            / np.maximum(np.linalg.norm(source_prev, axis=1)[:, None], 1e-12)
        )
        target_next = (
            np.linalg.norm(init_next, axis=1)[:, None]
            * source_next
            / np.maximum(np.linalg.norm(source_next, axis=1)[:, None], 1e-12)
        )
        bend_target = target_next - target_prev
        hessian = hessian + w_bend * (bend.T @ bend)
        rhs += w_bend * (bend.T @ bend_target)
    return hessian.tocsr(), rhs


def qp_energy(
    x: np.ndarray,
    x_init: np.ndarray,
    fit_weights: np.ndarray,
    w_fid: float,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    source_direction: np.ndarray,
    bend_prev: np.ndarray,
    bend_mid: np.ndarray,
    bend_next: np.ndarray,
    lap_kind: str,
    w_lap: float,
    w_edge: float = 0.0,
    w_bend: float = 0.0,
    lap_matrix: sp.csr_matrix | None = None,
    ori_lap: np.ndarray | None = None,
    lap_target: np.ndarray | None = None,
    lap_weights: np.ndarray | None = None,
) -> dict[str, float]:
    fit = 0.5 * w_fid * float(
        np.sum(fit_weights[:, None] * (x - x_init) ** 2)
    )
    if lap_kind == "matrix":
        residual = lap_matrix @ x - ori_lap
    else:
        residual = x - lap_target
    if lap_weights is None:
        lap_weights = fit_weights
    lap = 0.5 * w_lap * float(
        np.sum(lap_weights[:, None] * residual ** 2)
    )
    edge = x[edge_end] - x[edge_start]
    length = np.maximum(np.linalg.norm(edge, axis=1), 1e-12)
    shape = 0.5 * float(
        np.sum((edge / length[:, None] - source_direction) ** 2)
    )
    init_edge = x_init[edge_end] - x_init[edge_start]
    edge_target = np.linalg.norm(init_edge, axis=1)[:, None] * source_direction
    edge_energy = 0.5 * w_edge * float(np.sum((edge - edge_target) ** 2))
    bend_energy = 0.0
    if len(bend_prev):
        bend_value = x[bend_prev] - 2.0 * x[bend_mid] + x[bend_next]
        init_prev = x_init[bend_mid] - x_init[bend_prev]
        init_next = x_init[bend_next] - x_init[bend_mid]
        source_prev = source_direction[:-1]
        source_next = source_direction[1:]
        contiguous = edge_end[:-1] == edge_start[1:]
        bend_target = (
            np.linalg.norm(init_next, axis=1)[:, None] * source_next[contiguous]
            - np.linalg.norm(init_prev, axis=1)[:, None] * source_prev[contiguous]
        )
        bend_energy = 0.5 * w_bend * float(
            np.sum((bend_value - bend_target) ** 2)
        )
    total = fit + lap + shape + edge_energy + bend_energy
    return {
        "fit": fit,
        "lap": lap,
        "shape": shape,
        "edge": edge_energy,
        "bend": bend_energy,
        "total": total,
    }


def fixed_sdf_collision_matrix(
    point_count: int,
    roots: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    reference_x: np.ndarray,
    max_spacing: float,
    adaptive_samples: list[tuple[int, float]] | None = None,
    include_uniform_segments: bool = True,
) -> tuple[sp.csr_matrix, int, np.ndarray]:
    """Build a fixed point/segment sampling operator for SDF projection."""
    root_mask = np.zeros(point_count, dtype=bool)
    root_mask[roots] = True
    nonroots = np.flatnonzero(~root_mask)

    row_parts = [np.arange(len(nonroots), dtype=np.int64)]
    col_parts = [nonroots.astype(np.int64, copy=False)]
    data_parts = [np.ones(len(nonroots), dtype=np.float64)]
    kinds = [np.full(len(nonroots), "point", dtype="<U5")]
    row = len(nonroots)

    if include_uniform_segments and len(segment_start):
        length = np.linalg.norm(
            reference_x[segment_end] - reference_x[segment_start], axis=1
        )
        intervals = np.maximum(
            np.ceil(length / max(max_spacing, 1e-12)).astype(np.int32), 1
        )
        edge_ids = np.repeat(
            np.arange(len(segment_start), dtype=np.int64),
            np.maximum(intervals - 1, 0),
        )
        if len(edge_ids):
            fractions = np.concatenate([
                np.arange(1, int(count), dtype=np.float64) / float(count)
                for count in intervals if count > 1
            ])
            line_rows = np.arange(row, row + len(edge_ids), dtype=np.int64)
            starts = segment_start[edge_ids]
            ends = segment_end[edge_ids]
            row_parts.append(np.repeat(line_rows, 2))
            col_parts.append(np.column_stack([starts, ends]).ravel())
            data_parts.append(
                np.column_stack([1.0 - fractions, fractions]).ravel()
            )
            kinds.append(np.full(len(edge_ids), "line", dtype="<U5"))
            row += len(edge_ids)

    if adaptive_samples:
        adaptive_edges = np.asarray(
            [sample[0] for sample in adaptive_samples], dtype=np.int64
        )
        adaptive_fractions = np.asarray(
            [sample[1] for sample in adaptive_samples], dtype=np.float64
        )
        adaptive_rows = np.arange(
            row, row + len(adaptive_edges), dtype=np.int64
        )
        starts = segment_start[adaptive_edges]
        ends = segment_end[adaptive_edges]
        row_parts.append(np.repeat(adaptive_rows, 2))
        col_parts.append(np.column_stack([starts, ends]).ravel())
        data_parts.append(np.column_stack([
            1.0 - adaptive_fractions, adaptive_fractions
        ]).ravel())
        kinds.append(np.full(len(adaptive_edges), "adaptive", dtype="<U8"))
        row += len(adaptive_edges)

    matrix = sp.csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(row, point_count),
    )
    return matrix, len(nonroots), np.concatenate(kinds)


def sdf_type_from_name(name: str):
    if name == "winding_number":
        return igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
    if name == "fast_winding_number":
        return igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    return igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL


def project_exterior_sdf(
    query: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    face_normals: np.ndarray,
    margin: float,
    sdf_type,
) -> tuple[np.ndarray, np.ndarray]:
    """Project query points onto the margin-offset exterior SDF set."""
    signed, face_ids, closest = igl.signed_distance(
        query, body_vertices, body_faces, sdf_type
    )
    signed = np.asarray(signed, dtype=np.float64).reshape(-1)
    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
    projected = np.asarray(query, dtype=np.float64).copy()
    active = signed < margin
    if np.any(active):
        projected[active] = (
            closest[active] + margin * face_normals[face_ids[active]]
        )
    return projected, signed


class ConservativeSDFProjector:
    """Lazy exact SDF queries using a conservative distance lower bound.

    A previously exterior sample with signed distance ``d`` cannot reach the
    surface after moving by less than ``d``.  Rows whose cached distance minus
    accumulated displacement stays above ``active_band`` therefore have an
    identity exterior projection and do not need another closest-triangle
    query.  Near-surface and interior rows are always refreshed exactly.
    """

    def __init__(
        self,
        body_vertices: np.ndarray,
        body_faces: np.ndarray,
        face_normals: np.ndarray,
        margin: float,
        active_band: float,
        sdf_type: int,
        enabled: bool,
        reusable_pseudonormal=None,
    ):
        self.body_vertices = body_vertices
        self.body_faces = body_faces
        self.face_normals = face_normals
        self.margin = margin
        self.active_band = max(active_band, margin)
        self.sdf_type = sdf_type
        self.enabled = enabled
        self.reference = None
        self.cached_signed = None
        self.winding_rows = np.zeros(0, dtype=bool)
        self.reusable_pseudonormal = None
        if sdf_type == igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL:
            self.reusable_pseudonormal = reusable_pseudonormal
            if self.reusable_pseudonormal is None:
                self.reusable_pseudonormal = ReusablePseudonormalSDF(
                    body_vertices, body_faces, face_normals
                )

    def add_winding_rows(self, rows: np.ndarray, row_count: int) -> None:
        if len(self.winding_rows) < row_count:
            self.winding_rows = np.pad(
                self.winding_rows, (0, row_count - len(self.winding_rows))
            )
        self.winding_rows[np.asarray(rows, dtype=np.int64)] = True

    def exact_project(
        self, query: np.ndarray, row_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.reusable_pseudonormal is None:
            return project_exterior_sdf(
                query, self.body_vertices, self.body_faces,
                self.face_normals, self.margin, self.sdf_type,
            )
        signed, face_ids, closest = (
            self.reusable_pseudonormal.signed_distance(query)
        )
        projected = np.asarray(query, dtype=np.float64).copy()
        active = signed < self.margin
        if np.any(active):
            projected[active] = (
                closest[active]
                + self.margin * self.face_normals[face_ids[active]]
            )
        if row_ids is None:
            row_ids = np.arange(len(query), dtype=np.int64)
        if len(self.winding_rows):
            use_winding = self.winding_rows[np.asarray(row_ids, dtype=np.int64)]
            if np.any(use_winding):
                winding_projected, winding_signed = project_exterior_sdf(
                    query[use_winding], self.body_vertices, self.body_faces,
                    self.face_normals, self.margin,
                    igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
                )
                projected[use_winding] = winding_projected
                signed[use_winding] = winding_signed
        return projected, signed

    def project(
        self, query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        query = np.asarray(query, dtype=np.float64)
        if (
            not self.enabled
            or self.reference is None
            or self.reference.shape != query.shape
        ):
            projected, signed = self.exact_project(
                query, np.arange(len(query), dtype=np.int64)
            )
            self.reference = query.copy()
            self.cached_signed = signed.copy()
            return projected, signed, len(query)

        displacement = np.linalg.norm(query - self.reference, axis=1)
        signed_lower_bound = self.cached_signed - displacement
        candidates = signed_lower_bound < self.active_band
        candidate_rows = np.flatnonzero(candidates)
        projected = query.copy()
        signed = signed_lower_bound
        if len(candidate_rows):
            candidate_projected, candidate_signed = self.exact_project(
                query[candidate_rows], candidate_rows
            )
            projected[candidate_rows] = candidate_projected
            signed = signed.copy()
            signed[candidate_rows] = candidate_signed
            self.reference[candidate_rows] = query[candidate_rows]
            self.cached_signed[candidate_rows] = candidate_signed
        return projected, signed, len(candidate_rows)


class ReusablePseudonormalSDF:
    """Pseudonormal signed distance with one reusable libigl AABB tree."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_normals: np.ndarray,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.face_normals = np.asarray(face_normals, dtype=np.float64)
        self.vertex_normals = igl.per_vertex_normals(
            self.vertices, self.faces,
            igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,
        )
        _, face_edges, edge_faces = igl.edge_topology(
            self.vertices, self.faces
        )
        self.face_edges = np.asarray(face_edges, dtype=np.int64)
        edge_faces = np.asarray(edge_faces, dtype=np.int64)
        self.edge_normals = self.face_normals[
            np.maximum(edge_faces[:, 0], 0)
        ].copy()
        has_second_face = edge_faces[:, 1] >= 0
        self.edge_normals[has_second_face] += self.face_normals[
            edge_faces[has_second_face, 1]
        ]
        edge_norm = np.linalg.norm(self.edge_normals, axis=1, keepdims=True)
        self.edge_normals /= np.maximum(edge_norm, 1e-30)
        self.tree = igl.AABB_f64_3()
        self.tree.init(self.vertices, self.faces)

    def signed_distance(
        self, query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        squared, face_ids, closest = self.tree.squared_distance(
            self.vertices, self.faces, query,
            return_index=True, return_closest_point=True,
        )
        # libigl squeezes the batch dimension when exactly one point is
        # queried.  Normalize every return value back to batched form.
        squared = np.asarray(squared, dtype=np.float64).reshape(-1)
        face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
        triangles = self.vertices[self.faces[face_ids]]
        edge_0 = triangles[:, 1] - triangles[:, 0]
        edge_1 = triangles[:, 2] - triangles[:, 0]
        relative = closest - triangles[:, 0]
        dot_00 = np.einsum("ij,ij->i", edge_0, edge_0)
        dot_01 = np.einsum("ij,ij->i", edge_0, edge_1)
        dot_11 = np.einsum("ij,ij->i", edge_1, edge_1)
        dot_20 = np.einsum("ij,ij->i", relative, edge_0)
        dot_21 = np.einsum("ij,ij->i", relative, edge_1)
        denominator = dot_00 * dot_11 - dot_01 * dot_01
        denominator = np.where(
            np.abs(denominator) < 1e-30, 1.0, denominator
        )
        bary_1 = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
        bary_2 = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
        barycentric = np.column_stack([
            1.0 - bary_1 - bary_2, bary_1, bary_2
        ])
        on_boundary = barycentric <= 1e-7
        zero_count = np.count_nonzero(on_boundary, axis=1)
        pseudonormals = self.face_normals[face_ids].copy()

        at_vertex = zero_count >= 2
        if np.any(at_vertex):
            local_vertex = np.argmax(barycentric[at_vertex], axis=1)
            vertex_ids = self.faces[face_ids[at_vertex], local_vertex]
            pseudonormals[at_vertex] = self.vertex_normals[vertex_ids]

        at_edge = zero_count == 1
        if np.any(at_edge):
            zero_barycentric = np.argmax(on_boundary[at_edge], axis=1)
            local_edge = (zero_barycentric + 1) % 3
            edge_ids = self.face_edges[face_ids[at_edge], local_edge]
            pseudonormals[at_edge] = self.edge_normals[edge_ids]

        orientation = np.einsum(
            "ij,ij->i", query - closest, pseudonormals
        )
        signed = np.sqrt(np.maximum(squared, 0.0))
        signed[orientation < 0] *= -1.0
        return signed, face_ids, closest


def solve_stage_sdf_projection(
    stage_name: str,
    data: dict,
    static_hessian: sp.csr_matrix,
    static_rhs: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    face_normals: np.ndarray,
    intersector,
    args,
    initial_x: np.ndarray | None = None,
    reusable_pseudonormal=None,
):
    """Unified fixed-sampling ADMM with an exterior-SDF proximal step."""
    restart_initial_x = (
        None if initial_x is None
        else np.asarray(initial_x, dtype=np.float64).copy()
    )
    if initial_x is not None and initial_x.shape == data["init"].shape:
        x = np.asarray(initial_x, dtype=np.float64).copy()
    else:
        x = data["init"].copy()
    (
        roots,
        edge_start,
        edge_end,
        collision_edge_start,
        collision_edge_end,
        bend_prev,
        bend_mid,
        bend_next,
    ) = topology(data["starts"], data["lengths"])
    root_mask = np.zeros(len(x), dtype=bool)
    root_mask[roots] = True
    free = np.flatnonzero(~root_mask)
    fixed = roots
    x[fixed] = data["init"][fixed]

    regularity_hessian, regularity_rhs = regularity_system(
        data["init"], data["source"], edge_start, edge_end,
        bend_prev, bend_mid, bend_next,
        args.w_edge_length, args.w_bend_continuity,
    )
    candidate_band = args.sdf_candidate_band_mm * 0.001
    sdf_type = sdf_type_from_name(args.sdf_projection_sign)
    margin = args.sdf_margin_mm * 0.001
    edge_operator = difference_matrix(len(x), edge_start, edge_end)
    edge_free = edge_operator[:, free].tocsr()
    edge_fixed_value = np.asarray(edge_operator[:, fixed] @ x[fixed])
    initial_edge = np.asarray(edge_operator @ data["init"])
    initial_edge_length = np.linalg.norm(initial_edge, axis=1)
    edge_radius = args.edge_constraint_ratio * np.maximum(
        initial_edge_length, args.edge_constraint_reference_mm * 0.001
    )
    temporal_gate_enabled = (
        args.temporal_gate_reference_dir is not None
        and "temporal_gate_prev" in data
        and "temporal_gate_next" in data
    )
    temporal_gate_active: set[int] = set()
    if temporal_gate_enabled:
        temporal_gate_offset = (
            data["temporal_gate_prev"] + data["temporal_gate_next"]
        )
        temporal_gate_init_acceleration = (
            data["temporal_gate_init_prev"]
            - 2.0 * data["init"]
            + data["temporal_gate_init_next"]
        )
        temporal_gate_radius = (
            args.temporal_gate_init_factor
            * np.linalg.norm(temporal_gate_init_acceleration, axis=1)
        )

    def update_temporal_gate_active() -> tuple[float, float]:
        if not temporal_gate_enabled:
            return 0.0, 0.0
        acceleration = temporal_gate_offset - 2.0 * x
        magnitude = np.linalg.norm(acceleration, axis=1)
        violating = free[magnitude[free] > temporal_gate_radius[free]]
        temporal_gate_active.update(map(int, violating))
        ratio = np.divide(
            magnitude[free], temporal_gate_radius[free],
            out=np.zeros(len(free), dtype=np.float64),
            where=temporal_gate_radius[free] > 1e-30,
        )
        ratio[
            (temporal_gate_radius[free] <= 1e-30)
            & (magnitude[free] > 1e-30)
        ] = np.inf
        return (
            1000.0 * float(np.max(magnitude[free], initial=0.0)),
            float(np.max(ratio, initial=0.0)),
        )

    def build_temporal_gate():
        active = np.asarray(sorted(temporal_gate_active), dtype=np.int64)
        if not len(active):
            return (
                sp.csr_matrix((0, len(free))), active,
                np.empty((0, 3)), np.empty(0),
            )
        free_lookup = np.full(len(x), -1, dtype=np.int64)
        free_lookup[free] = np.arange(len(free), dtype=np.int64)
        rows = np.arange(len(active), dtype=np.int64)
        operator = sp.csr_matrix(
            (-2.0 * np.ones(len(active)), (rows, free_lookup[active])),
            shape=(len(active), len(free)),
        )
        return (
            operator, active, temporal_gate_offset[active],
            temporal_gate_radius[active],
        )

    def project_temporal_gate(
        value: np.ndarray, radius: np.ndarray,
    ) -> np.ndarray:
        if not len(value):
            return value.copy()
        magnitude = np.linalg.norm(value, axis=1)
        scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-30))
        return value * scale[:, None]

    def project_edge_lengths(edge_value: np.ndarray) -> np.ndarray:
        if not args.edge_length_constraint:
            return edge_value.copy()
        length = np.linalg.norm(edge_value, axis=1)
        scale = np.minimum(
            1.0, edge_radius / np.maximum(length, 1e-30)
        )
        return edge_value * scale[:, None]

    point_strand = np.empty(len(x), dtype=np.int32)
    point_local = np.empty(len(x), dtype=np.int32)
    for strand, (start, length) in enumerate(
        zip(data["starts"], data["lengths"])
    ):
        start = int(start)
        length = int(length)
        point_strand[start: start + length] = strand
        point_local[start: start + length] = np.arange(length)
    collision_edge_strand = point_strand[collision_edge_start]
    collision_edge_local = point_local[collision_edge_start]

    adaptive_samples: dict[tuple[int, int], tuple[int, float]] = {}
    refined_edges: set[int] = set()

    def add_feedback_constraints(
        feedback: list[AdaptiveCollisionSample],
    ) -> tuple[int, int]:
        """Add exact Embree exchange rows using the same SDF proximal."""
        added = 0
        hit_edges = sorted({int(sample.edge) for sample in feedback})

        def add_sample(edge: int, fraction: float):
            nonlocal added
            key = (edge, int(round(fraction * 10000)))
            if key not in adaptive_samples:
                adaptive_samples[key] = (edge, fraction)
                added += 1

        for sample in feedback:
            add_sample(int(sample.edge), float(sample.fraction))
        refined_edges.update(hit_edges)
        return added, len(hit_edges)

    initial_segment_hits = 0
    if args.sdf_embree_feedback:
        initial_segment_hits, initial_feedback, _ = (
            coherent_segment_constraints(
                x, collision_edge_start, collision_edge_end,
                collision_edge_strand, collision_edge_local,
                intersector, face_normals, margin,
            )
        )
        add_feedback_constraints(initial_feedback)

    persistent_candidate_rows = np.zeros(0, dtype=bool)
    persistent_winding_rows = np.zeros(0, dtype=bool)
    current_full_rows = np.zeros(0, dtype=np.int64)

    def build_collision_operator():
        nonlocal persistent_candidate_rows, persistent_winding_rows
        nonlocal current_full_rows
        collision_full, point_count, sample_kind_full = (
            fixed_sdf_collision_matrix(
                len(x), roots, collision_edge_start, collision_edge_end,
                data["init"], args.sdf_max_spacing_mm * 0.001,
                list(adaptive_samples.values()),
                include_uniform_segments=args.sdf_uniform_segment_samples,
            )
        )
        reference_samples = np.asarray(collision_full @ x)
        if candidate_band >= 0:
            if reusable_pseudonormal is not None:
                reference_signed, _, _ = (
                    reusable_pseudonormal.signed_distance(reference_samples)
                )
            else:
                _, reference_signed = project_exterior_sdf(
                    reference_samples, body_vertices, body_faces,
                    face_normals, 0.0,
                    igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
                )
            if len(persistent_candidate_rows) < len(sample_kind_full):
                persistent_candidate_rows = np.pad(
                    persistent_candidate_rows,
                    (0, len(sample_kind_full) - len(persistent_candidate_rows)),
                )
            persistent_candidate_rows |= (
                (sample_kind_full == "adaptive")
                | (reference_signed < candidate_band)
            )
            keep = persistent_candidate_rows
            current_full_rows = np.flatnonzero(keep)
            collision = collision_full[current_full_rows].tocsr()
            sample_kind = sample_kind_full[keep]
        else:
            collision = collision_full
            sample_kind = sample_kind_full
            current_full_rows = np.arange(len(sample_kind_full), dtype=np.int64)
        if len(persistent_winding_rows) < len(sample_kind_full):
            persistent_winding_rows = np.pad(
                persistent_winding_rows,
                (0, len(sample_kind_full) - len(persistent_winding_rows)),
            )
        point_count = int(np.count_nonzero(sample_kind == "point"))
        collision_free = collision[:, free].tocsr()
        projector = ConservativeSDFProjector(
            body_vertices, body_faces, face_normals, margin,
            args.sdf_active_band_mm * 0.001, sdf_type,
            args.sdf_active_set, reusable_pseudonormal,
        )
        winding_local = np.flatnonzero(
            persistent_winding_rows[current_full_rows]
        )
        projector.add_winding_rows(winding_local, len(current_full_rows))
        return (
            collision_full, collision, collision_free,
            point_count, sample_kind, projector,
        )

    (
        collision_full, collision, collision_free,
        point_count, sample_kind, projector,
    ) = build_collision_operator()
    history = []
    outer_budget = args.outer_iters
    refinement_outers = 0
    outer = 0
    shape_hessian = None
    shape_rhs = None
    source_direction = None

    while outer < outer_budget:
        outer_start = time.perf_counter()
        gate_input_max_mm, gate_input_max_ratio = (
            update_temporal_gate_active()
        )
        gate_free, gate_points, gate_offset, gate_radius = (
            build_temporal_gate()
        )
        shape_linearization_updated = (
            shape_hessian is None
            or outer < args.shape_outer_iters
            or outer >= args.outer_iters
        )
        if shape_linearization_updated:
            shape_hessian, shape_rhs, source_direction = shape_system(
                x, data["source"], edge_start, edge_end
            )
        hessian = (
            static_hessian + regularity_hessian + shape_hessian
        ).tocsr()
        rhs = static_rhs + regularity_rhs + shape_rhs
        hessian_free = hessian[free][:, free].tocsr()
        rhs_free = rhs[free] - hessian[free][:, fixed] @ x[fixed]
        system = (
            hessian_free + args.rho * (collision_free.T @ collision_free)
        ).tocsr()
        if args.edge_length_constraint:
            system = (
                system
                + args.edge_constraint_rho * (edge_free.T @ edge_free)
            ).tocsr()
        if gate_free.shape[0]:
            system = (
                system
                + args.temporal_gate_rho * (gate_free.T @ gate_free)
            ).tocsr()
        diagonal = system.diagonal()
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        preconditioner = sp.diags(1.0 / diagonal, format="csr")
        direct_solver = None
        if stage_name == "normal" and args.normal_direct:
            direct_solver = spla.factorized(system.tocsc())

        projection_seconds = 0.0
        projection_query_count = 0
        projection_call_count = 0
        cx = np.asarray(collision @ x)
        projection_start = time.perf_counter()
        z, signed, queried = projector.project(cx)
        projection_seconds += time.perf_counter() - projection_start
        projection_query_count += queried
        projection_call_count += 1
        u = np.zeros_like(cx)
        edge_value = np.asarray(edge_operator @ x)
        edge_z = project_edge_lengths(edge_value)
        edge_u = np.zeros_like(edge_value)
        gate_value = np.asarray(gate_free @ x[free]) + gate_offset
        gate_z = project_temporal_gate(gate_value, gate_radius)
        gate_u = np.zeros_like(gate_value)
        primal_rms = dual_rms = dual_state_rms = None
        dual_relative = dual_threshold = None
        residual_trace = []
        cg_retry_count = 0
        inner_start = time.perf_counter()

        for inner in range(args.inner_iters):
            iteration_count = inner + 1
            check_residual = (
                iteration_count >= args.min_inner_iters
                and iteration_count % args.residual_check_every == 0
            )
            previous_z = z.copy() if check_residual else None
            admm_rhs = rhs_free + args.rho * (
                collision_free.T @ (z - u)
            )
            if args.edge_length_constraint:
                admm_rhs += args.edge_constraint_rho * (
                    edge_free.T @ (edge_z - edge_u - edge_fixed_value)
                )
            if gate_free.shape[0]:
                admm_rhs += args.temporal_gate_rho * (
                    gate_free.T @ (gate_z - gate_u - gate_offset)
                )
            if direct_solver is None:
                x_free, infos = solve_spd(
                    system, np.asarray(admm_rhs), x[free],
                    args.cg_iters, args.cg_tol, preconditioner,
                )
                retry_iters = args.cg_iters
                while (
                    any(info != 0 for info in infos)
                    and retry_iters < args.cg_retry_iters
                ):
                    retry_iters = min(2 * retry_iters, args.cg_retry_iters)
                    x_free, infos = solve_spd(
                        system, np.asarray(admm_rhs), x_free,
                        retry_iters, args.cg_tol, preconditioner,
                    )
                    cg_retry_count += 1
            else:
                x_free = np.column_stack([
                    direct_solver(np.asarray(admm_rhs)[:, coord])
                    for coord in range(3)
                ])
                infos = [0, 0, 0]
            if any(info != 0 for info in infos):
                raise RuntimeError(
                    f"{stage_name}: SDF projection CG did not converge: {infos}"
                )
            x[free] = x_free
            x[fixed] = data["init"][fixed]
            cx = np.asarray(collision @ x)
            projection_start = time.perf_counter()
            z, signed, queried = projector.project(cx + u)
            projection_seconds += time.perf_counter() - projection_start
            projection_query_count += queried
            projection_call_count += 1
            u += cx - z
            if args.edge_length_constraint:
                edge_value = np.asarray(edge_operator @ x)
                edge_z = project_edge_lengths(edge_value + edge_u)
                edge_u += edge_value - edge_z
            if gate_free.shape[0]:
                gate_value = np.asarray(gate_free @ x[free]) + gate_offset
                gate_z = project_temporal_gate(
                    gate_value + gate_u, gate_radius
                )
                gate_u += gate_value - gate_z

            if check_residual:
                active_count = max(int(np.count_nonzero(signed < margin)), 1)
                primal = cx - z
                dual = args.rho * (
                    collision_free.T @ (z - previous_z)
                )
                primal_rms = float(
                    np.linalg.norm(primal) / np.sqrt(3 * active_count)
                )
                dual_rms = float(
                    np.linalg.norm(dual)
                    / np.sqrt(3 * max(len(free), 1))
                )
                dual_state = args.rho * (collision_free.T @ u)
                dual_state_rms = float(
                    np.linalg.norm(dual_state)
                    / np.sqrt(3 * max(len(free), 1))
                )
                dual_relative = dual_rms / max(dual_state_rms, 1e-12)
                dual_threshold = args.dual_tol
                residual_trace.append({
                    "iteration": iteration_count,
                    "primal_rms_mm": 1000.0 * primal_rms,
                    "dual_rms": dual_rms,
                    "dual_state_rms": dual_state_rms,
                    "dual_relative": dual_relative,
                    "dual_threshold": dual_threshold,
                })
                if (
                    primal_rms <= args.primal_tol_mm * 0.001
                    and dual_rms <= dual_threshold
                ):
                    break

        inner_seconds = time.perf_counter() - inner_start
        final_samples = np.asarray(collision @ x)
        projection_start = time.perf_counter()
        _, final_signed, queried = projector.project(final_samples)
        projection_seconds += time.perf_counter() - projection_start
        projection_query_count += queried
        projection_call_count += 1

        winding_point_count = None
        winding_margin_count = None
        if args.sdf_winding_feedback:
            _, winding_signed = project_exterior_sdf(
                final_samples[:point_count], body_vertices, body_faces,
                face_normals, margin,
                igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
            )
            winding_point_count = int(np.count_nonzero(winding_signed < 0.0))
            winding_margin_rows = np.flatnonzero(winding_signed < margin)
            winding_margin_count = int(len(winding_margin_rows))
            if len(winding_margin_rows):
                persistent_winding_rows[
                    current_full_rows[winding_margin_rows]
                ] = True
                projector.add_winding_rows(
                    winding_margin_rows, len(current_full_rows)
                )

        segment_start = time.perf_counter()
        segment_hits, feedback, conflict_clusters = (
            coherent_segment_constraints(
                x, collision_edge_start, collision_edge_end,
                collision_edge_strand, collision_edge_local,
                intersector, face_normals, margin,
            )
        )
        segment_seconds = time.perf_counter() - segment_start
        added_samples = 0
        refined_hit_edges = 0
        can_run_feedback = (
            outer + 1 < outer_budget
            or refinement_outers < args.sdf_embree_refine_iters
        )
        if args.sdf_embree_feedback and segment_hits and can_run_feedback:
            added_samples, refined_hit_edges = add_feedback_constraints(
                feedback
            )
        history.append({
            "outer": outer,
            "refinement_outer": outer >= args.outer_iters,
            "shape_linearization_updated": shape_linearization_updated,
            "sample_count": int(collision.shape[0]),
            "full_sample_count": int(collision_full.shape[0]),
            "point_samples": int(np.count_nonzero(sample_kind == "point")),
            "line_samples": int(np.count_nonzero(sample_kind == "line")),
            "active_samples": int(np.count_nonzero(final_signed < margin)),
            "sdf_exact_queries": int(projection_query_count),
            "sdf_projection_calls": int(projection_call_count),
            "sdf_mean_queries_per_call": float(
                projection_query_count / max(projection_call_count, 1)
            ),
            "admm_constraint_rows": int(collision.shape[0]),
            "inner_iterations": iteration_count,
            "cg_retry_count": cg_retry_count,
            "edge_constraint_max_ratio": float(np.max(
                np.linalg.norm(edge_value, axis=1)
                / np.maximum(edge_radius, 1e-30)
            )),
            "temporal_gate_active_points": len(temporal_gate_active),
            "temporal_gate_input_acceleration_max_mm": gate_input_max_mm,
            "temporal_gate_input_max_ratio_to_limit": gate_input_max_ratio,
            "temporal_gate_primal_max_mm": 1000.0 * float(np.max(
                np.linalg.norm(gate_value - gate_z, axis=1), initial=0.0
            )),
            "primal_rms_mm": (
                None if primal_rms is None else 1000.0 * primal_rms
            ),
            "dual_rms": dual_rms,
            "dual_state_rms": dual_state_rms,
            "dual_relative": dual_relative,
            "dual_threshold": dual_threshold,
            "residual_trace": residual_trace,
            "local_point_penetrations": int(
                np.count_nonzero(final_signed[:point_count] < 0)
            ),
            "winding_point_penetrations": winding_point_count,
            "winding_margin_violations": winding_margin_count,
            "segment_intersections": None,
            "initial_segment_intersections": initial_segment_hits,
            "adaptive_constraint_samples": int(np.count_nonzero(
                sample_kind == "adaptive"
            )),
            "adaptive_samples_added": added_samples,
            "adaptive_refined_edges": len(refined_edges),
            "adaptive_hit_edges_refined": refined_hit_edges,
            "adaptive_conflict_clusters": conflict_clusters,
            "suppressed_conflict_clusters": 0,
            "suppressed_collision_rows": 0,
            "timing_seconds": {
                "collision_detection": projection_seconds,
                "admm_inner": inner_seconds,
                "segment_intersections": segment_seconds,
                "outer_total": time.perf_counter() - outer_start,
            },
        })
        print(
            f"    {stage_name} outer={outer + 1} "
            f"inner={iteration_count} "
            f"sdf_active={history[-1]['active_samples']} "
            f"line={segment_hits} added={added_samples}",
            flush=True,
        )

        history[-1]["segment_intersections"] = segment_hits
        if (
            segment_hits > args.sdf_stage_segment_budget
            and outer + 1 >= outer_budget
            and refinement_outers < args.sdf_embree_refine_iters
        ):
            outer_budget += 1
            refinement_outers += 1
        if (
            winding_point_count
            and outer + 1 >= outer_budget
            and refinement_outers < args.sdf_winding_feedback_iters
        ):
            outer_budget += 1
            refinement_outers += 1
        if added_samples and outer + 1 < outer_budget:
            (
                collision_full, collision, collision_free,
                point_count, sample_kind, projector,
            ) = build_collision_operator()
        outer += 1

    final_segment_hits = int(history[-1]["segment_intersections"] or 0)
    if (
        final_segment_hits > args.sdf_stage_segment_budget
        and args.sdf_multiresolution_fallback
        and args.sdf_max_spacing_mm > args.sdf_fine_spacing_mm
    ):
        fine_args = copy.copy(args)
        fine_args.sdf_max_spacing_mm = args.sdf_fine_spacing_mm
        fine_args.sdf_embree_refine_iters = args.sdf_fine_refine_iters
        fine_args.sdf_multiresolution_fallback = False
        print(
            f"    {stage_name} exact line={final_segment_hits} "
            f"> budget={args.sdf_stage_segment_budget}; "
            f"restart at {fine_args.sdf_max_spacing_mm:g}mm resolution",
            flush=True,
        )
        result = solve_stage_sdf_projection(
            stage_name, data, static_hessian, static_rhs,
            body_vertices, body_faces, face_normals, intersector,
            fine_args, restart_initial_x, reusable_pseudonormal,
        )
        fine_x, fine_history, fine_topology, _ = result
        if fine_history:
            fine_history[0]["multiresolution_fallback"] = {
                "coarse_spacing_mm": args.sdf_max_spacing_mm,
                "coarse_outer_count": len(history),
                "coarse_final_segment_intersections": final_segment_hits,
                "fine_spacing_mm": fine_args.sdf_max_spacing_mm,
            }
        return fine_x, fine_history, fine_topology, None

    _, _, source_direction = shape_system(
        x, data["source"], edge_start, edge_end
    )
    return x, history, (
        roots, edge_start, edge_end,
        collision_edge_start, collision_edge_end, source_direction,
        bend_prev, bend_mid, bend_next,
    ), None


def solve_stage(
    stage_name: str,
    data: dict,
    static_hessian: sp.csr_matrix,
    static_rhs: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    face_normals: np.ndarray,
    intersector,
    args,
    initial_x: np.ndarray | None = None,
    reusable_pseudonormal=None,
):
    if args.collision_standard == "sdf_projection":
        return solve_stage_sdf_projection(
            stage_name, data, static_hessian, static_rhs,
            body_vertices, body_faces, face_normals, intersector,
            args, initial_x, reusable_pseudonormal,
        )
    if initial_x is not None and initial_x.shape == data["init"].shape:
        x = np.asarray(initial_x, dtype=np.float64).copy()
    else:
        x = data["init"].copy()
    (
        roots,
        edge_start,
        edge_end,
        collision_edge_start,
        collision_edge_end,
        bend_prev,
        bend_mid,
        bend_next,
    ) = topology(data["starts"], data["lengths"])
    root_mask = np.zeros(len(x), dtype=bool)
    root_mask[roots] = True
    free = np.where(~root_mask)[0]
    fixed = roots
    x[fixed] = data["init"][fixed]
    point_strand = np.empty(len(x), dtype=np.int32)
    point_local = np.empty(len(x), dtype=np.int32)
    for strand, (start, length) in enumerate(
        zip(data["starts"], data["lengths"])
    ):
        start = int(start)
        length = int(length)
        point_strand[start: start + length] = strand
        point_local[start: start + length] = np.arange(length)
    collision_edge_strand = point_strand[collision_edge_start]
    collision_edge_local = point_local[collision_edge_start]
    adaptive_samples: list[AdaptiveCollisionSample] = []
    history = []
    regularity_hessian, regularity_rhs = regularity_system(
        data["init"],
        data["source"],
        edge_start,
        edge_end,
        bend_prev,
        bend_mid,
        bend_next,
        args.w_edge_length,
        args.w_bend_continuity,
    )

    for outer in range(args.outer_iters):
        outer_start = time.perf_counter()
        shape_hessian, shape_rhs, source_direction = shape_system(
            x, data["source"], edge_start, edge_end
        )
        hessian = (
            static_hessian + regularity_hessian + shape_hessian
        ).tocsr()
        rhs = static_rhs + regularity_rhs + shape_rhs
        hessian_free = hessian[free][:, free].tocsr()
        rhs_free = rhs[free] - hessian[free][:, fixed] @ x[fixed]

        collision, point_count, sample_kind, sample_edge = collision_matrix(
            len(x), roots, collision_edge_start, collision_edge_end,
            args.line_samples,
            [(sample.edge, sample.fraction) for sample in adaptive_samples],
        )
        collision_free = collision[:, free].tocsr()
        samples = np.asarray(collision @ x)
        band = args.collision_band_mm * 0.001
        margin = np.full(len(samples), args.line_margin_mm * 0.001)
        margin[sample_kind == "point"] = args.margin_mm * 0.001
        penalty_scale = np.full(len(samples), args.line_rho_scale)
        penalty_scale[sample_kind == "point"] = 1.0
        detection_start = time.perf_counter()
        signed, plane_points, plane_normals, active = collision_planes(
            samples, body_vertices, body_faces, face_normals,
            band, point_count,
            args.winding_point_constraints,
        )
        if args.collision_standard == "qp_point" and outer > 0:
            # Historical QP only filters its precomputed collision set in the
            # first stage. Later stages rebuild a nearest-face half-space for
            # every non-fixed vertex and solve all of those inequalities.
            active[:point_count] = True
        adaptive_rows = np.flatnonzero(sample_kind == "adaptive")
        if len(adaptive_rows) != len(adaptive_samples):
            raise RuntimeError("adaptive collision sample metadata mismatch")
        for row, sample in zip(adaptive_rows, adaptive_samples):
            plane_points[row] = sample.plane_point
            plane_normals[row] = sample.plane_normal
            signed[row] = float(
                (samples[row] - sample.plane_point) @ sample.plane_normal
            )
            active[row] = signed[row] < band
        nonroots = np.where(~root_mask)[0]
        sample_strand = np.empty(len(samples), dtype=np.int32)
        sample_local = np.empty(len(samples), dtype=np.float64)
        sample_strand[:point_count] = point_strand[nonroots]
        sample_local[:point_count] = point_local[nonroots]
        segment_rows = np.arange(point_count, len(samples))
        segment_edges = sample_edge[segment_rows]
        sample_strand[segment_rows] = collision_edge_strand[segment_edges]
        sample_local[segment_rows] = collision_edge_local[segment_edges] + 0.5
        for row, sample in zip(adaptive_rows, adaptive_samples):
            sample_local[row] = (
                collision_edge_local[sample.edge] + sample.fraction
            )
        suppressed_conflict_clusters, suppressed_collision_rows = (
            suppress_opposing_collision_rows(
                plane_normals,
                active,
                sample_kind,
                sample_strand,
                sample_local,
            )
        )
        detection_seconds = time.perf_counter() - detection_start

        if args.all_collision_rows:
            constraint_rows = np.arange(len(samples), dtype=np.int64)
        else:
            constraint_rows = np.flatnonzero(active)
        collision_admm = collision[constraint_rows].tocsr()
        collision_free_admm = collision_free[constraint_rows].tocsr()
        plane_points_admm = plane_points[constraint_rows]
        plane_normals_admm = plane_normals[constraint_rows]
        margin_admm = margin[constraint_rows]
        penalty_scale_admm = penalty_scale[constraint_rows]
        active_admm = active[constraint_rows]
        weighted_collision = collision_free_admm.multiply(
            penalty_scale_admm[:, None]
        )
        system = (
            hessian_free + args.rho * (
                collision_free_admm.T @ weighted_collision
            )
        ).tocsr()
        diagonal = system.diagonal()
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        preconditioner = sp.diags(1.0 / diagonal, format="csr")
        direct_solver = None
        if stage_name == "normal" and args.normal_direct:
            direct_solver = spla.factorized(system.tocsc())
        z = samples[constraint_rows].copy()
        u = np.zeros_like(z)
        primal_rms = None
        dual_rms = None
        dual_state_rms = None
        dual_relative = None
        dual_threshold = None
        residual_trace = []
        cg_retry_count = 0
        inner_start = time.perf_counter()

        for inner in range(args.inner_iters):
            previous_z = z.copy()
            iteration_count = inner + 1
            current_cg_tol = args.cg_tol
            if args.inexact_cg:
                if iteration_count <= args.cg_early_iters:
                    current_cg_tol = max(current_cg_tol, args.cg_tol_early)
                elif iteration_count < args.inner_iters - 1:
                    current_cg_tol = max(current_cg_tol, args.cg_tol_middle)
            admm_rhs = rhs_free + args.rho * (
                collision_free_admm.T
                @ (penalty_scale_admm[:, None] * (z - u))
            )
            if direct_solver is None:
                x_free, infos = solve_spd(
                    system, np.asarray(admm_rhs), x[free],
                    args.cg_iters, current_cg_tol, preconditioner,
                )
                retry_iters = args.cg_iters
                while (
                    any(info != 0 for info in infos)
                    and retry_iters < args.cg_retry_iters
                ):
                    retry_iters = min(2 * retry_iters, args.cg_retry_iters)
                    x_free, infos = solve_spd(
                        system, np.asarray(admm_rhs), x_free,
                        retry_iters, current_cg_tol, preconditioner,
                    )
                    cg_retry_count += 1
            else:
                x_free = np.column_stack([
                    direct_solver(np.asarray(admm_rhs)[:, coord])
                    for coord in range(3)
                ])
                infos = [0, 0, 0]
            if any(info != 0 for info in infos):
                raise RuntimeError(
                    f"{stage_name}: CG did not converge: {infos}"
                )
            x[free] = x_free
            x[fixed] = data["init"][fixed]
            cx = np.asarray(collision_admm @ x)
            q = cx + u
            z = q.copy()
            gap = np.einsum(
                "ij,ij->i", q - plane_points_admm, plane_normals_admm
            )
            correction = np.maximum(margin_admm - gap, 0.0)
            z[active_admm] += (
                correction[active_admm, None]
                * plane_normals_admm[active_admm]
            )
            u += cx - z

            check_residual = (
                iteration_count >= args.min_inner_iters
                and iteration_count % args.residual_check_every == 0
            )
            if check_residual:
                primal = cx - z
                dual = args.rho * (
                    collision_free_admm.T
                    @ (
                        penalty_scale_admm[:, None]
                        * (z - previous_z)
                    )
                )
                active_count = max(int(np.count_nonzero(active_admm)), 1)
                primal_rms = float(
                    np.linalg.norm(primal[active_admm])
                    / np.sqrt(3 * active_count)
                )
                dual_rms = float(
                    np.linalg.norm(dual)
                    / np.sqrt(3 * max(len(free), 1))
                )
                dual_state = args.rho * (
                    collision_free_admm.T
                    @ (penalty_scale_admm[:, None] * u)
                )
                dual_state_rms = float(
                    np.linalg.norm(dual_state)
                    / np.sqrt(3 * max(len(free), 1))
                )
                dual_relative = dual_rms / max(dual_state_rms, 1e-12)
                dual_threshold = (
                    args.dual_tol
                    if args.residual_mode == "absolute"
                    else args.dual_tol
                    + args.dual_rel_tol * dual_state_rms
                )
                residual_trace.append({
                    "iteration": iteration_count,
                    "primal_rms_mm": 1000.0 * primal_rms,
                    "dual_rms": dual_rms,
                    "dual_state_rms": dual_state_rms,
                    "dual_relative": dual_relative,
                    "dual_threshold": dual_threshold,
                })
                if (
                    primal_rms <= args.primal_tol_mm * 0.001
                    and dual_rms <= dual_threshold
                ):
                    break

        inner_seconds = time.perf_counter() - inner_start

        cx = np.asarray(collision @ x)
        local_gap = np.einsum("ij,ij->i", cx - plane_points, plane_normals)
        local_point_count = int(
            np.count_nonzero(local_gap[:point_count] < 0)
        )
        segment_start = time.perf_counter()
        segment_hits, new_samples, conflict_clusters = (
            coherent_segment_constraints(
                x,
                collision_edge_start,
                collision_edge_end,
                collision_edge_strand,
                collision_edge_local,
                intersector,
                face_normals,
                args.line_margin_mm * 0.001,
            )
        )
        if (
            args.collision_standard == "qp_point"
            and not args.qp_point_embree_feedback
        ):
            # Historical QP constrains colliding vertices only. Keep Embree
            # here as a post-hoc diagnostic, but never feed line hits back
            # into the next outer iteration.
            new_samples = []
            conflict_clusters = 0
        segment_seconds = time.perf_counter() - segment_start
        history.append({
            "outer": outer,
            "sample_count": int(len(samples)),
            "active_samples": int(np.count_nonzero(active)),
            "admm_constraint_rows": int(len(constraint_rows)),
            "inner_iterations": iteration_count,
            "cg_retry_count": cg_retry_count,
            "primal_rms_mm": (
                None if primal_rms is None else 1000.0 * primal_rms
            ),
            "dual_rms": dual_rms,
            "dual_state_rms": dual_state_rms,
            "dual_relative": dual_relative,
            "dual_threshold": dual_threshold,
            "residual_trace": residual_trace,
            "local_point_penetrations": local_point_count,
            "winding_point_penetrations": None,
            "segment_intersections": segment_hits,
            "adaptive_constraint_samples": len(new_samples),
            "adaptive_conflict_clusters": conflict_clusters,
            "suppressed_conflict_clusters": suppressed_conflict_clusters,
            "suppressed_collision_rows": suppressed_collision_rows,
            "timing_seconds": {
                "collision_detection": detection_seconds,
                "admm_inner": inner_seconds,
                "segment_intersections": segment_seconds,
                "outer_total": time.perf_counter() - outer_start,
            },
        })
        print(
            f"    {stage_name} outer={outer + 1} "
            f"inner={iteration_count} point=local:{local_point_count} "
            f"line={segment_hits}",
            flush=True,
        )
        stage_valid = local_point_count == 0 and segment_hits == 0
        if stage_valid:
            break
        adaptive_samples = new_samples

    _, _, source_direction = shape_system(
        x, data["source"], edge_start, edge_end
    )
    return x, history, (
        roots, edge_start, edge_end,
        collision_edge_start, collision_edge_end, source_direction,
        bend_prev, bend_mid, bend_next,
    ), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--guide_npz",
        default=None,
        help="Guide metadata used only by the undocumented compatibility path",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--w_fid", type=float, default=1000.0)
    parser.add_argument("--w_lap", type=float, default=3000.0)
    parser.add_argument("--w_lap_normal", type=float, default=30000.0)
    parser.add_argument(
        "--qp_objective_tensor_dir", type=Path,
        help=(
            "directory of frame_XXXX.npz bundles exported by QP; when set, "
            "Init, aligned Source, Guide selection, both KNN graphs, KNN "
            "weights, Laplacian targets, and local weights are consumed "
            "verbatim instead of recomputed"
        ),
    )
    parser.add_argument(
        "--qp_tensor_weight_mode", choices=("legacy", "corrected"),
        default="corrected",
        help="required local-weight semantics in QP tensor bundles",
    )
    parser.add_argument(
        "--compat_local_weight_mode", choices=("legacy", "corrected"),
        default="legacy",
        help=(
            "local-weight semantics when no QP tensor directory is supplied; "
            "legacy reproduces the historical QP root-overwrite behavior"
        ),
    )
    parser.add_argument("--w_edge_length", type=float, default=0.0)
    parser.add_argument("--w_bend_continuity", type=float, default=0.0)
    parser.add_argument(
        "--w_temporal_correction", type=float, default=0.0,
        help=(
            "quadratic weight on correction acceleration relative to the "
            "two preceding published frames"
        ),
    )
    parser.add_argument(
        "--temporal_target_dir", type=Path,
        help="optional per-frame NPZ targets supplied by temporal ADMM",
    )
    parser.add_argument(
        "--w_temporal_target", type=float, default=0.0,
        help="quadratic penalty for temporal_target_dir positions",
    )
    parser.add_argument(
        "--edge_length_constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include the uniform edge-length feasibility block Dx=y in ADMM",
    )
    parser.add_argument("--edge_constraint_ratio", type=float, default=3.0)
    parser.add_argument(
        "--edge_constraint_reference_mm", type=float, default=1.0
    )
    parser.add_argument("--edge_constraint_rho", type=float, default=1e5)
    parser.add_argument(
        "--temporal_gate_reference_dir", type=Path,
        help=(
            "optional frozen spatial-result sequence used only as the two "
            "neighbor references of the per-frame ADMM spike gate"
        ),
    )
    parser.add_argument("--temporal_gate_init_factor", type=float, default=1.5)
    parser.add_argument("--temporal_gate_rho", type=float, default=5e5)
    parser.add_argument("--rho", type=float, default=1e5)
    parser.add_argument("--line_rho_scale", type=float, default=10.0)
    parser.add_argument("--margin_mm", type=float, default=1.0)
    parser.add_argument("--line_margin_mm", type=float, default=2.0)
    parser.add_argument("--collision_band_mm", type=float, default=5.0)
    parser.add_argument("--line_samples", type=int, default=1)
    parser.add_argument(
        "--collision_standard",
        choices=("strict", "qp_point", "sdf_projection"),
        default="sdf_projection",
        help=(
            "sdf_projection is the formal unified exterior-SDF proximal "
            "operator; strict retains legacy tangent-plane constraints; "
            "qp_point retains historical QP-style vertex constraints"
        ),
    )
    parser.add_argument(
        "--qp_point_margin_mm", type=float, default=0.1,
        help="vertex half-space margin used by collision_standard=qp_point",
    )
    parser.add_argument(
        "--qp_point_embree_feedback", action="store_true",
        help=(
            "feed coherent residual Embree intersections into later outer "
            "iterations when collision_standard=qp_point"
        ),
    )
    parser.add_argument("--sdf_margin_mm", type=float, default=1.0)
    parser.add_argument("--sdf_max_spacing_mm", type=float, default=2.0)
    parser.add_argument(
        "--sdf_uniform_segment_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include deterministic interior segment samples in the unified "
            "SDF proximal operator; disable for a clean vertex-only ablation"
        ),
    )
    parser.add_argument(
        "--sdf_embree_feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "add deterministic samples around residual Embree intersections; "
            "new rows use the same SDF proximal operator, never fixed planes"
        ),
    )
    parser.add_argument(
        "--sdf_embree_refine_iters", type=int, default=2,
        help=(
            "maximum constraint-generation outer iterations used only while "
            "exact Embree feedback still finds intersections"
        ),
    )
    parser.add_argument(
        "--sdf_multiresolution_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "restart a stage at fine uniform resolution when its exact "
            "Embree feedback exceeds the coarse-resolution budget"
        ),
    )
    parser.add_argument("--sdf_fine_spacing_mm", type=float, default=1.0)
    parser.add_argument("--sdf_fine_refine_iters", type=int, default=20)
    parser.add_argument(
        "--sdf_stage_segment_budget", type=int, default=0,
        help=(
            "allowed exact Embree intersections per guide/normal stage; "
            "fine multiresolution fallback is used only above this budget"
        ),
    )
    parser.add_argument(
        "--sdf_active_set", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use conservative cached-distance lower bounds to query exact "
            "SDF only for rows that can enter sdf_active_band_mm"
        ),
    )
    parser.add_argument(
        "--sdf_active_band_mm", type=float, default=1.0,
        help="conservative exact-SDF refresh band for sdf_projection",
    )
    parser.add_argument(
        "--sdf_winding_feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "query point rows with winding number after each outer and "
            "persist sign-disagreement rows in the same SDF proximal"
        ),
    )
    parser.add_argument(
        "--sdf_winding_feedback_iters", type=int, default=2,
        help="maximum extra outer iterations for residual winding points",
    )
    parser.add_argument(
        "--sdf_candidate_band_mm", type=float, default=-1.0,
        help=(
            "outer active-set band: only rows inside this signed-distance "
            "band enter the ADMM system; adaptive Embree rows always remain"
        ),
    )
    parser.add_argument(
        "--sdf_projection_sign",
        choices=("pseudonormal", "fast_winding_number", "winding_number"),
        default="pseudonormal",
    )
    parser.add_argument("--outer_iters", type=int, default=4)
    parser.add_argument(
        "--shape_outer_iters", type=int, default=4,
        help=(
            "number of base QP-compatible shape majorization updates; "
            "collision/refinement outers also relinearize the shape system"
        ),
    )
    parser.add_argument("--inner_iters", type=int, default=10)
    parser.add_argument("--min_inner_iters", type=int, default=10)
    parser.add_argument("--primal_tol_mm", type=float, default=0.02)
    parser.add_argument("--dual_tol", type=float, default=1e-3)
    parser.add_argument("--dual_rel_tol", type=float, default=1e-3)
    parser.add_argument(
        "--residual_mode", choices=("absolute", "relative"),
        default="absolute",
    )
    parser.add_argument("--residual_check_every", type=int, default=2)
    parser.add_argument("--all_collision_rows", action="store_true")
    parser.add_argument("--warm_start_frames", action="store_true")
    parser.add_argument("--cg_iters", type=int, default=100)
    parser.add_argument("--cg_retry_iters", type=int, default=400)
    parser.add_argument("--cg_tol", type=float, default=1e-6)
    parser.add_argument("--inexact_cg", action="store_true")
    parser.add_argument(
        "--normal_direct", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cg_early_iters", type=int, default=0)
    parser.add_argument("--cg_tol_early", type=float, default=1e-4)
    parser.add_argument("--cg_tol_middle", type=float, default=1e-5)
    parser.add_argument("--winding_point_constraints", action="store_true")
    parser.add_argument(
        "--output_format", choices=("npz", "obj", "both"), default="npz"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.collision_standard == "qp_point":
        args.collision_band_mm = 0.0
        args.margin_mm = args.qp_point_margin_mm
        args.line_margin_mm = args.qp_point_margin_mm
        args.line_rho_scale = 1.0
        args.line_samples = 0
        args.winding_point_constraints = False

    if args.inner_iters < 1 or args.outer_iters < 1:
        parser.error("outer_iters and inner_iters must be positive")
    if not 1 <= args.shape_outer_iters <= args.outer_iters:
        parser.error("shape_outer_iters must be in [1, outer_iters]")
    if args.qp_point_margin_mm < 0:
        parser.error("qp_point_margin_mm must be non-negative")
    if args.w_temporal_correction < 0:
        parser.error("w_temporal_correction must be non-negative")
    if args.w_temporal_target < 0:
        parser.error("w_temporal_target must be non-negative")
    if args.sdf_margin_mm < 0:
        parser.error("sdf_margin_mm must be non-negative")
    if args.sdf_max_spacing_mm <= 0:
        parser.error("sdf_max_spacing_mm must be positive")
    if args.sdf_fine_spacing_mm <= 0:
        parser.error("sdf_fine_spacing_mm must be positive")
    if args.sdf_fine_spacing_mm > args.sdf_max_spacing_mm:
        parser.error("sdf_fine_spacing_mm must be <= sdf_max_spacing_mm")
    if args.sdf_fine_refine_iters < 0:
        parser.error("sdf_fine_refine_iters must be non-negative")
    if args.sdf_stage_segment_budget < 0:
        parser.error("sdf_stage_segment_budget must be non-negative")
    if args.sdf_embree_refine_iters < 0:
        parser.error("sdf_embree_refine_iters must be non-negative")
    if args.sdf_winding_feedback_iters < 0:
        parser.error("sdf_winding_feedback_iters must be non-negative")
    if args.sdf_active_band_mm < args.sdf_margin_mm:
        parser.error("sdf_active_band_mm must be >= sdf_margin_mm")
    if not 1 <= args.min_inner_iters <= args.inner_iters:
        parser.error("min_inner_iters must be in [1, inner_iters]")
    if args.residual_check_every < 1:
        parser.error("residual_check_every must be positive")
    if not 0 <= args.cg_early_iters < args.inner_iters:
        parser.error("cg_early_iters must be in [0, inner_iters)")
    if args.cg_retry_iters < args.cg_iters:
        parser.error("cg_retry_iters must be >= cg_iters")
    if (
        args.edge_constraint_ratio <= 0
        or args.edge_constraint_reference_mm <= 0
        or args.edge_constraint_rho <= 0
    ):
        parser.error("edge-length ADMM constraint parameters must be positive")
    if args.temporal_gate_init_factor <= 0 or args.temporal_gate_rho <= 0:
        parser.error("temporal-gate ADMM parameters must be positive")

    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    hair_dir = output_dir / "hair"
    metrics_dir = output_dir / "metrics"
    hair_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def parameter_snapshot():
        ignored = {"start_frame", "end_frame", "force", "output_dir"}
        result = {}
        for key, value in vars(args).items():
            if key in ignored:
                continue
            if isinstance(value, Path):
                result[key] = str(value.resolve())
            elif value is None or isinstance(value, (str, int, float, bool)):
                result[key] = value
        return result

    current_parameters = parameter_snapshot()

    init_paths = {
        frame_id(path): path
        for path in (case_dir / "training" / "init_transfer").glob("frame_*.npy")
    }
    source_paths = {
        frame_id(path): path
        for path in (case_dir / "source").glob("rods_*.obj")
    }
    tensor_paths = {}
    strict_qp_tensors = args.qp_objective_tensor_dir is not None
    if strict_qp_tensors:
        tensor_dir = args.qp_objective_tensor_dir.resolve()
        tensor_paths = {
            frame_id(path): path for path in tensor_dir.glob("frame_*.npz")
        }
        if not tensor_paths:
            parser.error(f"no frame_*.npz QP tensor bundles found in {tensor_dir}")
        first_tensor = to_padded_objective(load_qp_objective_bundle(
            tensor_paths[min(tensor_paths)],
            expected_local_weight_mode=args.qp_tensor_weight_mode,
        ), L_MAX)
        strand_count = int(first_tensor["strand_count"])
        knn_full = None
        guide_idx = None
        normal_idx = None
        ids = sorted(tensor_paths)
        print(
            "Strict QP tensor mode: objective inputs are loaded verbatim "
            f"from {tensor_dir}",
            flush=True,
        )
    else:
        if args.guide_npz is None:
            parser.error(
                "compatibility mode requires --guide_npz; the public entry "
                "point uses --qp_objective_tensor_dir instead"
            )
        knn_full, strand_count = load_knn(case_dir)
        if not init_paths:
            parser.error("no training/init_transfer/frame_*.npy files found")
        first_init = np.load(init_paths[min(init_paths)])
        guide_idx, _ = _load_guide_data(
            args.guide_npz, strand_count, first_init[:, 0]
        )
        guide_idx = np.asarray(guide_idx, dtype=np.int64)
        normal_idx = np.setdiff1d(np.arange(strand_count), guide_idx)
        ids = sorted(init_paths.keys() & source_paths.keys())
        print(
            "WARNING: compatibility mode recomputes objective tensors using "
            f"{args.compat_local_weight_mode} local weights; it does not "
            "guarantee tensor identity with the historical QP run.",
            flush=True,
        )
    ids = [
        fid for fid in ids
        if fid >= args.start_frame and (args.end_frame < 0 or fid <= args.end_frame)
    ]
    if not ids:
        parser.error("no frames remain after applying the requested frame range")

    def build_guide_knn_cache(frame: int) -> np.ndarray:
        """Build the per-process guide graph from the selected first frame."""
        init_shape = np.load(init_paths[frame], mmap_mode="r").shape
        input_length = int(init_shape[1])
        source_points, source_starts = _parse_hair_obj(source_paths[frame])
        source_padded, source_lengths = build_padded_rods(
            np.asarray(source_points, dtype=np.float64) * 0.01,
            source_starts,
        )
        subset_lengths = np.minimum(
            source_lengths[guide_idx].astype(np.int32), input_length
        )
        subset_starts = np.zeros(len(guide_idx), dtype=np.int64)
        if len(subset_starts) > 1:
            subset_starts[1:] = np.cumsum(subset_lengths[:-1])
        global_ids = np.concatenate([
            int(strand) * L_MAX + np.arange(int(length), dtype=np.int64)
            for strand, length in zip(guide_idx, subset_lengths)
        ])
        return cross_strand_knn(
            source_padded.reshape(-1, 3)[global_ids],
            subset_starts,
            subset_lengths,
            5,
        )

    temporal_gate_hair = None
    if args.temporal_gate_reference_dir is not None:
        temporal_gate_hair = args.temporal_gate_reference_dir.resolve()
        if (temporal_gate_hair / "hair").is_dir():
            temporal_gate_hair /= "hair"

    def padded_init(frame: int) -> np.ndarray:
        if strict_qp_tensors:
            if frame not in tensor_paths:
                raise FileNotFoundError(
                    f"missing QP tensor bundle for frame {frame}"
                )
            frame_tensor = to_padded_objective(load_qp_objective_bundle(
                tensor_paths[frame],
                expected_local_weight_mode=args.qp_tensor_weight_mode,
            ), L_MAX)
            if int(frame_tensor["strand_count"]) != strand_count:
                raise ValueError(
                    f"frame {frame} strand count differs from the first bundle"
                )
            return frame_tensor["initial_padded"]
        raw = np.asarray(np.load(init_paths[frame]), dtype=np.float64)
        padded = np.zeros((strand_count, L_MAX, 3), dtype=np.float64)
        count = min(raw.shape[1], L_MAX)
        padded[:, :count] = raw[:, :count]
        if count < L_MAX:
            padded[:, count:] = raw[:, count - 1:count]
        return padded

    def padded_reference(frame: int) -> np.ndarray:
        path = temporal_gate_hair / f"frame_{frame:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path)
        rods = np.asarray(payload["rods"], dtype=np.float64)
        padded = np.zeros((strand_count, L_MAX, 3), dtype=np.float64)
        count = min(rods.shape[1], L_MAX)
        padded[:, :count] = rods[:, :count]
        if count < L_MAX:
            padded[:, count:] = rods[:, count - 1:count]
        return padded

    guide_knn_cache = None
    previous_fid = None
    previous_guide_x = None
    previous_guide_init = None
    previous_normal_x = None
    previous_normal_init = None
    guide_correction_history = []
    normal_correction_history = []
    for index, fid in enumerate(ids):
        destination = hair_dir / f"frame_{fid:04d}.npz"
        obj_destination = hair_dir / f"frame_{fid:04d}.obj"
        metric_destination = metrics_dir / f"frame_{fid:04d}.json"
        frame_complete = destination.exists() and metric_destination.exists()
        if args.output_format in ("obj", "both"):
            frame_complete = frame_complete and obj_destination.exists()
        if frame_complete and strict_qp_tensors:
            try:
                previous_metrics = json.loads(metric_destination.read_text())
                previous_contract = previous_metrics.get(
                    "objective_tensor_contract", {}
                )
                frame_complete = (
                    previous_contract.get("strict") is True
                    and previous_contract.get("sha256")
                    == file_sha256(tensor_paths[fid])
                    and previous_metrics.get("parameters")
                    == current_parameters
                )
            except (OSError, ValueError, TypeError):
                frame_complete = False
        if frame_complete and not args.force:
            if not strict_qp_tensors and guide_knn_cache is None:
                guide_knn_cache = build_guide_knn_cache(fid)
            previous_fid = None
            previous_guide_x = None
            previous_guide_init = None
            previous_normal_x = None
            previous_normal_init = None
            guide_correction_history.clear()
            normal_correction_history.clear()
            continue
        print(f"[{index + 1}/{len(ids)}] frame {fid}", flush=True)

        frame_objective = None
        tensor_path_used = None
        tensor_sha256 = None
        if strict_qp_tensors:
            tensor_path_used = tensor_paths[fid]
            frame_objective = to_padded_objective(load_qp_objective_bundle(
                tensor_path_used,
                expected_local_weight_mode=args.qp_tensor_weight_mode,
            ), L_MAX)
            if int(frame_objective["strand_count"]) != strand_count:
                raise ValueError(
                    f"frame {fid} strand count differs from the first bundle"
                )
            for argument_name, tensor_name in (
                ("w_fid", "w_fid"),
                ("w_lap", "w_lap"),
                ("w_lap_normal", "w_lap_normal"),
            ):
                argument_value = float(getattr(args, argument_name))
                tensor_value = float(frame_objective[tensor_name])
                if not np.isclose(
                    argument_value, tensor_value, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        f"frame {fid}: --{argument_name}={argument_value} "
                        f"does not match QP tensor value {tensor_value}"
                    )
            disabled = [
                name for name in ("use_shape", "use_fit", "use_lap")
                if not frame_objective[name]
            ]
            if disabled:
                raise ValueError(
                    "strict QP tensor mode currently requires all three QP "
                    f"objective blocks; disabled in bundle: {disabled}"
                )
            n_strands = strand_count
            lengths = frame_objective["hair_lengths"].astype(np.int32)
            input_length = int(lengths.max())
            init_padded = frame_objective["initial_padded"]
            source_padded = frame_objective["source_padded"]
            fit_local_weights = frame_objective["fit_local_weights_padded"]
            lap_local_weights = frame_objective["lap_local_weights_padded"]
            knn_full = frame_objective["normal_knn_padded"]
            knn_weights = frame_objective["normal_knn_weights_padded"]
            ori_lap = frame_objective["normal_ori_laplacian_padded"]
            guide_idx = frame_objective["guide_strand_indices"]
            normal_idx = np.setdiff1d(
                np.arange(strand_count, dtype=np.int64), guide_idx
            )
            guide_knn_frame = frame_objective["guide_knn_compact"]
            guide_knn_weights_frame = frame_objective["guide_knn_weights"]
            guide_ori_lap_frame = frame_objective["guide_ori_laplacian"]
            tensor_sha256 = file_sha256(tensor_path_used)
        else:
            init_raw = np.asarray(np.load(init_paths[fid]), dtype=np.float64)
            n_strands, input_length, _ = init_raw.shape
            if input_length > L_MAX:
                raise ValueError(
                    f"frame {fid} input length {input_length} exceeds L_MAX={L_MAX}"
                )
            init_padded = np.zeros((n_strands, L_MAX, 3), dtype=np.float64)
            init_padded[:, :input_length] = init_raw
            if input_length < L_MAX:
                init_padded[:, input_length:] = init_raw[:, -1:, :]

            source_points, source_starts = _parse_hair_obj(source_paths[fid])
            source_padded, source_lengths = build_padded_rods(
                np.asarray(source_points, dtype=np.float64) * 0.01,
                source_starts,
            )
            lengths = np.minimum(
                source_lengths.astype(np.int32), input_length
            )
            knn_weights, ori_lap = _compute_invdist_weights_and_lap(
                source_padded.reshape(-1, 3), knn_full
            )
        x_init_flat = init_padded.reshape(-1, 3)
        x_source_flat = source_padded.reshape(-1, 3)
        temporal_target_padded = None
        if args.temporal_target_dir is not None:
            temporal_path = (
                args.temporal_target_dir / f"frame_{fid:04d}.npz"
            )
            if not temporal_path.exists():
                raise FileNotFoundError(temporal_path)
            temporal_data = np.load(temporal_path)
            temporal_rods = np.asarray(
                temporal_data["rods"], dtype=np.float64
            )
            temporal_target_padded = init_padded.copy()
            copy_length = min(input_length, temporal_rods.shape[1])
            temporal_target_padded[:, :copy_length] = (
                temporal_rods[:, :copy_length]
            )
            if copy_length < L_MAX:
                temporal_target_padded[:, copy_length:] = (
                    temporal_target_padded[:, copy_length - 1:copy_length]
                )
        if strict_qp_tensors:
            body_vertices = frame_objective["target_body_vertices"]
            body_faces = frame_objective["target_body_faces"].astype(np.int32)
            target_v_cm = None
        else:
            target_v_cm, body_faces = _fast_read_obj(
                case_dir / "target" / f"body_{fid}.obj"
            )
            body_vertices = np.asarray(target_v_cm, dtype=np.float64) * 0.01
        face_normals = _prepare_body_qpdetect(body_vertices, body_faces)
        reusable_pseudonormal = None
        if args.collision_standard == "sdf_projection":
            reusable_pseudonormal = ReusablePseudonormalSDF(
                body_vertices, body_faces, face_normals
            )
        intersector = RayMeshIntersector(
            trimesh.Trimesh(
                vertices=body_vertices, faces=body_faces, process=False
            )
        )
        if not strict_qp_tensors:
            if args.compat_local_weight_mode == "legacy":
                # Historical QP code overwrote both surface-root arrays with
                # the Init hair roots.  Reproduce that objective deliberately
                # when GT is treated as the authoritative old-QP baseline.
                source_roots = x_init_flat[
                    np.arange(n_strands) * L_MAX
                ].copy()
                target_roots = source_roots.copy()
            else:
                target_roots = _get_mesh_roots(
                    x_init_flat[np.arange(n_strands) * L_MAX],
                    target_v_cm, body_faces,
                )
                source_body_path = case_dir / "source" / f"body_{fid}.obj"
                if source_body_path.exists():
                    source_v_cm, source_faces = _fast_read_obj(
                        source_body_path
                    )
                    source_roots = _get_mesh_roots(
                        x_source_flat[np.arange(n_strands) * L_MAX],
                        source_v_cm, source_faces,
                    )
                else:
                    source_roots = x_source_flat[
                        np.arange(n_strands) * L_MAX
                    ]
            fit_local_weights = _compute_per_point_weights(
                x_init_flat, n_strands, source_roots, target_roots
            )
            lap_local_weights = fit_local_weights

        guide = compact_subset(
            guide_idx, init_padded, source_padded, lengths, fit_local_weights
        )
        guide_lap_w = lap_local_weights[guide["global_ids"]]
        if strict_qp_tensors and not np.array_equal(
            guide["global_ids"], frame_objective["guide_global_ids_padded"]
        ):
            raise ValueError(
                f"frame {fid}: compact Guide point ordering differs from QP"
            )
        gate_reference_current = None
        gate_reference_prev = None
        gate_reference_next = None
        gate_init_prev = None
        gate_init_next = None
        if (
            temporal_gate_hair is not None
            and fid - 1 in (tensor_paths if strict_qp_tensors else init_paths)
            and fid + 1 in (tensor_paths if strict_qp_tensors else init_paths)
            and (temporal_gate_hair / f"frame_{fid - 1:04d}.npz").exists()
            and (temporal_gate_hair / f"frame_{fid:04d}.npz").exists()
            and (temporal_gate_hair / f"frame_{fid + 1:04d}.npz").exists()
        ):
            gate_reference_current = padded_reference(fid)
            gate_reference_prev = padded_reference(fid - 1)
            gate_reference_next = padded_reference(fid + 1)
            gate_init_prev = padded_init(fid - 1)
            gate_init_next = padded_init(fid + 1)
            guide["temporal_gate_prev"] = gate_reference_prev.reshape(
                -1, 3
            )[guide["global_ids"]]
            guide["temporal_gate_next"] = gate_reference_next.reshape(
                -1, 3
            )[guide["global_ids"]]
            guide["temporal_gate_init_prev"] = gate_init_prev.reshape(
                -1, 3
            )[guide["global_ids"]]
            guide["temporal_gate_init_next"] = gate_init_next.reshape(
                -1, 3
            )[guide["global_ids"]]
        algorithm_start_time = time.perf_counter()
        if strict_qp_tensors:
            guide_knn_current = guide_knn_frame
            guide_knn_w = guide_knn_weights_frame
            guide_ori_lap = guide_ori_lap_frame
        else:
            if guide_knn_cache is None:
                guide_knn_cache = cross_strand_knn(
                    guide["source"], guide["starts"], guide["lengths"], 5
                )
            guide_knn_current = guide_knn_cache
            guide_knn_w, guide_ori_lap = _compute_invdist_weights_and_lap(
                guide["source"], guide_knn_current
            )
        guide_lap = sparse_laplacian(guide_knn_current, guide_knn_w)
        guide_w = guide["weights"]
        guide_W = sp.diags(args.w_lap * guide_lap_w, format="csr")
        guide_static_hessian = (
            sp.diags(args.w_fid * guide_w, format="csr")
            + guide_lap.T @ guide_W @ guide_lap
        ).tocsr()
        guide_static_rhs = (
            args.w_fid * guide_w[:, None] * guide["init"]
            + (guide_lap.T @ guide_W) @ guide_ori_lap
        )
        guide_external_temporal_target = None
        if temporal_target_padded is not None and args.w_temporal_target > 0:
            guide_external_temporal_target = (
                temporal_target_padded.reshape(-1, 3)[guide["global_ids"]]
            )
            guide_static_hessian = (
                guide_static_hessian
                + args.w_temporal_target
                * sp.eye(len(guide["init"]), format="csr")
            ).tocsr()
            guide_static_rhs += (
                args.w_temporal_target * guide_external_temporal_target
            )
        guide_temporal_target = None
        if (
            args.w_temporal_correction > 0
            and len(guide_correction_history) == 2
            and guide_correction_history[-2][0] + 1
            == guide_correction_history[-1][0]
            and guide_correction_history[-1][0] + 1 == fid
        ):
            guide_temporal_target = guide["init"] + (
                2.0 * guide_correction_history[-1][1]
                - guide_correction_history[-2][1]
            )
            guide_static_hessian = (
                guide_static_hessian
                + args.w_temporal_correction
                * sp.eye(len(guide["init"]), format="csr")
            ).tocsr()
            guide_static_rhs += (
                args.w_temporal_correction * guide_temporal_target
            )
        use_frame_warm_start = (
            args.warm_start_frames and previous_fid is not None
            and fid == previous_fid + 1
        )
        guide_initial_x = None
        if gate_reference_current is not None:
            guide_initial_x = gate_reference_current.reshape(-1, 3)[
                guide["global_ids"]
            ]
        if (
            guide_initial_x is None
            and use_frame_warm_start
            and previous_guide_x.shape == guide["init"].shape
        ):
            guide_initial_x = (
                guide["init"] + previous_guide_x - previous_guide_init
            )
        guide_solve_start = time.perf_counter()
        guide_x, guide_history, guide_topology, _ = solve_stage(
            "guide", guide, guide_static_hessian,
            np.asarray(guide_static_rhs), body_vertices,
            body_faces, face_normals, intersector, args,
            guide_initial_x, reusable_pseudonormal,
        )
        guide_solve_seconds = time.perf_counter() - guide_solve_start
        guide_system_build_seconds = guide_solve_start - algorithm_start_time
        normal_system_start = time.perf_counter()

        x_solution_padded = init_padded.copy().reshape(-1, 3)
        x_solution_padded[guide["global_ids"]] = guide_x
        valid_knn = (knn_full >= 0) & (knn_full < len(x_solution_padded))
        safe_knn = np.where(valid_knn, knn_full, 0)
        lap_target_padded = (
            np.sum(
                knn_weights[:, :, None] * x_solution_padded[safe_knn], axis=1
            )
            + ori_lap
        ).reshape(n_strands, L_MAX, 3)

        normal = compact_subset(
            normal_idx, init_padded, source_padded, lengths,
            fit_local_weights, lap_target_padded,
        )
        normal_lap_w = lap_local_weights[normal["global_ids"]]
        if gate_reference_current is not None:
            normal["temporal_gate_prev"] = gate_reference_prev.reshape(
                -1, 3
            )[normal["global_ids"]]
            normal["temporal_gate_next"] = gate_reference_next.reshape(
                -1, 3
            )[normal["global_ids"]]
            normal["temporal_gate_init_prev"] = gate_init_prev.reshape(
                -1, 3
            )[normal["global_ids"]]
            normal["temporal_gate_init_next"] = gate_init_next.reshape(
                -1, 3
            )[normal["global_ids"]]
        normal_w = normal["weights"]
        normal_static_hessian = sp.diags(
            args.w_fid * normal_w + args.w_lap_normal * normal_lap_w,
            format="csr",
        )
        normal_static_rhs = (
            args.w_fid * normal_w[:, None] * normal["init"]
            + args.w_lap_normal
            * normal_lap_w[:, None]
            * normal["lap_target"]
        )
        normal_external_temporal_target = None
        if temporal_target_padded is not None and args.w_temporal_target > 0:
            normal_external_temporal_target = (
                temporal_target_padded.reshape(-1, 3)[normal["global_ids"]]
            )
            normal_static_hessian = (
                normal_static_hessian
                + args.w_temporal_target
                * sp.eye(len(normal["init"]), format="csr")
            ).tocsr()
            normal_static_rhs += (
                args.w_temporal_target * normal_external_temporal_target
            )
        normal_temporal_target = None
        if (
            args.w_temporal_correction > 0
            and len(normal_correction_history) == 2
            and normal_correction_history[-2][0] + 1
            == normal_correction_history[-1][0]
            and normal_correction_history[-1][0] + 1 == fid
        ):
            normal_temporal_target = normal["init"] + (
                2.0 * normal_correction_history[-1][1]
                - normal_correction_history[-2][1]
            )
            normal_static_hessian = (
                normal_static_hessian
                + args.w_temporal_correction
                * sp.eye(len(normal["init"]), format="csr")
            ).tocsr()
            normal_static_rhs += (
                args.w_temporal_correction * normal_temporal_target
            )
        normal_initial_x = None
        if gate_reference_current is not None:
            normal_initial_x = gate_reference_current.reshape(-1, 3)[
                normal["global_ids"]
            ]
        if (
            normal_initial_x is None
            and use_frame_warm_start
            and previous_normal_x.shape == normal["init"].shape
        ):
            normal_initial_x = (
                normal["init"] + previous_normal_x - previous_normal_init
            )
        normal_solve_start = time.perf_counter()
        normal_system_build_seconds = normal_solve_start - normal_system_start
        normal_x, normal_history, normal_topology, _ = solve_stage(
            "normal", normal, normal_static_hessian,
            normal_static_rhs, body_vertices,
            body_faces, face_normals, intersector, args,
            normal_initial_x, reusable_pseudonormal,
        )
        normal_solve_seconds = time.perf_counter() - normal_solve_start
        algorithm_seconds = time.perf_counter() - algorithm_start_time
        x_solution_padded[normal["global_ids"]] = normal_x
        rods = x_solution_padded.reshape(n_strands, L_MAX, 3)[:, :input_length]

        (
            g_roots, g_es, g_ee, g_ces, g_cee, g_dir,
            g_bp, g_bm, g_bn,
        ) = guide_topology
        (
            n_roots, n_es, n_ee, n_ces, n_cee, n_dir,
            n_bp, n_bm, n_bn,
        ) = normal_topology
        guide_energy = qp_energy(
            guide_x, guide["init"], guide_w, args.w_fid,
            g_es, g_ee, g_dir, g_bp, g_bm, g_bn,
            "matrix", args.w_lap,
            w_edge=args.w_edge_length,
            w_bend=args.w_bend_continuity,
            lap_matrix=guide_lap,
            ori_lap=guide_ori_lap,
            lap_weights=guide_lap_w,
        )
        normal_energy = qp_energy(
            normal_x, normal["init"], normal_w, args.w_fid,
            n_es, n_ee, n_dir, n_bp, n_bm, n_bn,
            "target", args.w_lap_normal,
            w_edge=args.w_edge_length,
            w_bend=args.w_bend_continuity,
            lap_target=normal["lap_target"],
            lap_weights=normal_lap_w,
        )
        np.savez_compressed(
            destination, rods=rods.astype(np.float32),
            lengths=lengths.astype(np.int32),
        )
        if args.output_format in ("obj", "both"):
            write_hair_obj(obj_destination, rods, lengths)
        metrics = {
            "frame": fid,
            "objective_tensor_contract": {
                "strict": strict_qp_tensors,
                "path": (
                    None if tensor_path_used is None
                    else str(tensor_path_used.resolve())
                ),
                "sha256": tensor_sha256,
                "schema_version": (
                    None if frame_objective is None
                    else int(frame_objective["schema_version"])
                ),
                "local_weight_mode": (
                    None if frame_objective is None
                    else frame_objective["local_weight_mode"]
                ),
                "verbatim_fields": ([] if frame_objective is None else [
                    "initial_transfer_pos", "source_aligned",
                    "guide_strand_indices", "guide_knn_indices",
                    "guide_knn_weights", "guide_ori_laplacian",
                    "normal_knn_indices", "normal_knn_weights",
                    "normal_ori_laplacian", "fit_local_weights",
                    "lap_local_weights",
                    "target_body_vertices", "target_body_faces",
                ]),
            },
            "guide_energy": guide_energy,
            "normal_energy": normal_energy,
            "total_energy": guide_energy["total"] + normal_energy["total"],
            "temporal_correction_energy": 0.5 * args.w_temporal_correction * (
                (0.0 if guide_temporal_target is None else float(np.sum(
                    (guide_x - guide_temporal_target) ** 2
                )))
                + (0.0 if normal_temporal_target is None else float(np.sum(
                    (normal_x - normal_temporal_target) ** 2
                )))
            ),
            "temporal_target_energy": 0.5 * args.w_temporal_target * (
                (0.0 if guide_external_temporal_target is None else float(
                    np.sum((guide_x - guide_external_temporal_target) ** 2)
                ))
                + (0.0 if normal_external_temporal_target is None else float(
                    np.sum((normal_x - normal_external_temporal_target) ** 2)
                ))
            ),
            "guide_history": guide_history,
            "normal_history": normal_history,
            "published": True,
            "algorithm_seconds": algorithm_seconds,
            "algorithm_timing_seconds": {
                "guide_system_build": guide_system_build_seconds,
                "guide_admm": guide_solve_seconds,
                "normal_target_and_system_build": normal_system_build_seconds,
                "normal_admm": normal_solve_seconds,
                "total": algorithm_seconds,
            },
            "timing_scope": (
                "guide/normal system construction and spatial ADMM only; "
                "excludes data input, output, offline evaluation, energy statistics, "
                "and viewer preprocessing"
            ),
            "frame_warm_start": use_frame_warm_start,
            "parameters": current_parameters,
        }
        metric_destination.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        print(
            f"  E_qp={metrics['total_energy']:.6e} "
            f"published=True "
            f"algorithm_time={metrics['algorithm_seconds']:.2f}s",
            flush=True,
        )
        previous_fid = fid
        previous_guide_x = guide_x.copy()
        previous_guide_init = guide["init"].copy()
        previous_normal_x = normal_x.copy()
        previous_normal_init = normal["init"].copy()
        guide_correction_history.append((
            fid, (guide_x - guide["init"]).copy()
        ))
        normal_correction_history.append((
            fid, (normal_x - normal["init"]).copy()
        ))
        del guide_correction_history[:-2]
        del normal_correction_history[:-2]


if __name__ == "__main__":
    main()
