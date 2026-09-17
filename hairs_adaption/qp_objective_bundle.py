"""Canonical tensors shared by the QP and ADMM objective builders.

The QP pipeline writes this bundle immediately before optimization.  The
ADMM pipeline consumes it instead of independently rebuilding KNN graphs,
weights, source directions, or Laplacian targets.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
POINT_UNIT = "m"


def _scalar(payload, name, cast):
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return cast(value.reshape(-1)[0])


def _require_shape(name, value, shape):
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def validate_qp_objective_bundle(payload, expected_local_weight_mode=None):
    """Validate and normalize a loaded/in-memory objective tensor bundle."""
    required = {
        "schema_version", "point_unit", "local_weight_mode",
        "initial_transfer_pos", "source_aligned", "hair_starts",
        "hair_lengths", "guide_strand_indices", "guide_knn_indices",
        "guide_knn_weights", "guide_ori_laplacian",
        "normal_knn_indices", "normal_knn_weights",
        "normal_ori_laplacian", "fit_local_weights", "lap_local_weights",
        "source_mesh_roots", "target_body_vertices", "target_body_faces",
        "target_mesh_roots", "w_fid", "w_lap", "w_lap_normal",
        "use_shape", "use_fit", "use_lap",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"QP objective bundle is missing fields: {missing}")

    schema_version = _scalar(payload, "schema_version", int)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported QP objective schema {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    point_unit = _scalar(payload, "point_unit", str)
    if point_unit != POINT_UNIT:
        raise ValueError(f"point_unit must be {POINT_UNIT!r}, got {point_unit!r}")
    local_weight_mode = _scalar(payload, "local_weight_mode", str)
    if local_weight_mode not in ("legacy", "corrected"):
        raise ValueError(f"unknown local_weight_mode {local_weight_mode!r}")
    if (
        expected_local_weight_mode is not None
        and local_weight_mode != expected_local_weight_mode
    ):
        raise ValueError(
            "local weight semantics do not match: bundle uses "
            f"{local_weight_mode!r}, requested {expected_local_weight_mode!r}"
        )

    normalized = {
        "schema_version": schema_version,
        "point_unit": point_unit,
        "local_weight_mode": local_weight_mode,
        "initial_transfer_pos": np.asarray(
            payload["initial_transfer_pos"], dtype=np.float64
        ),
        "source_aligned": np.asarray(payload["source_aligned"], dtype=np.float64),
        "hair_starts": np.asarray(payload["hair_starts"], dtype=np.int64),
        "hair_lengths": np.asarray(payload["hair_lengths"], dtype=np.int64),
        "guide_strand_indices": np.asarray(
            payload["guide_strand_indices"], dtype=np.int64
        ),
        "guide_knn_indices": np.asarray(payload["guide_knn_indices"], dtype=np.int64),
        "guide_knn_weights": np.asarray(payload["guide_knn_weights"], dtype=np.float64),
        "guide_ori_laplacian": np.asarray(
            payload["guide_ori_laplacian"], dtype=np.float64
        ),
        "normal_knn_indices": np.asarray(payload["normal_knn_indices"], dtype=np.int64),
        "normal_knn_weights": np.asarray(payload["normal_knn_weights"], dtype=np.float64),
        "normal_ori_laplacian": np.asarray(
            payload["normal_ori_laplacian"], dtype=np.float64
        ),
        "fit_local_weights": np.asarray(
            payload["fit_local_weights"], dtype=np.float64
        ),
        "lap_local_weights": np.asarray(
            payload["lap_local_weights"], dtype=np.float64
        ),
        "source_mesh_roots": np.asarray(payload["source_mesh_roots"], dtype=np.float64),
        "target_mesh_roots": np.asarray(payload["target_mesh_roots"], dtype=np.float64),
        "target_body_vertices": np.asarray(
            payload["target_body_vertices"], dtype=np.float64
        ),
        "target_body_faces": np.asarray(
            payload["target_body_faces"], dtype=np.int64
        ),
        "w_fid": _scalar(payload, "w_fid", float),
        "w_lap": _scalar(payload, "w_lap", float),
        "w_lap_normal": _scalar(payload, "w_lap_normal", float),
        "use_shape": _scalar(payload, "use_shape", bool),
        "use_fit": _scalar(payload, "use_fit", bool),
        "use_lap": _scalar(payload, "use_lap", bool),
    }

    initial = normalized["initial_transfer_pos"]
    source = normalized["source_aligned"]
    if initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError(f"initial_transfer_pos must be (N, 3), got {initial.shape}")
    _require_shape("source_aligned", source, initial.shape)
    point_count = len(initial)

    starts = normalized["hair_starts"]
    lengths = normalized["hair_lengths"]
    if starts.ndim != 1 or lengths.ndim != 1 or starts.shape != lengths.shape:
        raise ValueError("hair_starts and hair_lengths must be equal-length vectors")
    if not len(starts) or starts[0] != 0 or np.any(lengths <= 0):
        raise ValueError("hair topology must start at zero and have positive lengths")
    expected_starts = np.r_[0, np.cumsum(lengths[:-1])]
    if not np.array_equal(starts, expected_starts) or int(lengths.sum()) != point_count:
        raise ValueError("hair_starts/hair_lengths are not contiguous over all points")
    strand_count = len(starts)

    guide_strands = normalized["guide_strand_indices"]
    if guide_strands.ndim != 1 or len(np.unique(guide_strands)) != len(guide_strands):
        raise ValueError("guide_strand_indices must be a unique vector")
    if np.any((guide_strands < 0) | (guide_strands >= strand_count)):
        raise ValueError("guide_strand_indices contains an out-of-range strand")
    guide_global_ids = np.concatenate([
        np.arange(starts[s], starts[s] + lengths[s], dtype=np.int64)
        for s in guide_strands
    ]) if len(guide_strands) else np.empty(0, dtype=np.int64)
    guide_knn = normalized["guide_knn_indices"]
    guide_weights = normalized["guide_knn_weights"]
    if guide_knn.ndim != 2:
        raise ValueError("guide_knn_indices must be a matrix")
    _require_shape("guide_knn_weights", guide_weights, guide_knn.shape)
    if guide_knn.shape[0] != len(guide_global_ids):
        raise ValueError("guide KNN row count does not match the guide point count")
    _require_shape(
        "guide_ori_laplacian", normalized["guide_ori_laplacian"],
        (len(guide_global_ids), 3),
    )
    valid_guide_knn = guide_knn >= 0
    if np.any(guide_knn[valid_guide_knn] >= point_count):
        raise ValueError("guide KNN contains an out-of-range point")
    guide_mask = np.zeros(point_count, dtype=bool)
    guide_mask[guide_global_ids] = True
    if np.any(~guide_mask[guide_knn[valid_guide_knn]]):
        raise ValueError("guide KNN contains a non-guide point")

    normal_knn = normalized["normal_knn_indices"]
    normal_weights = normalized["normal_knn_weights"]
    if normal_knn.ndim != 2 or normal_knn.shape[0] != point_count:
        raise ValueError("normal_knn_indices must have one row per hair point")
    _require_shape("normal_knn_weights", normal_weights, normal_knn.shape)
    _require_shape(
        "normal_ori_laplacian", normalized["normal_ori_laplacian"],
        (point_count, 3),
    )
    valid_normal_knn = normal_knn >= 0
    if np.any(normal_knn[valid_normal_knn] >= point_count):
        raise ValueError("normal KNN contains an out-of-range point")
    _require_shape(
        "fit_local_weights", normalized["fit_local_weights"], (point_count,)
    )
    _require_shape(
        "lap_local_weights", normalized["lap_local_weights"], (point_count,)
    )
    _require_shape(
        "source_mesh_roots", normalized["source_mesh_roots"],
        (strand_count, 3),
    )
    _require_shape(
        "target_mesh_roots", normalized["target_mesh_roots"],
        (strand_count, 3),
    )
    body_vertices = normalized["target_body_vertices"]
    body_faces = normalized["target_body_faces"]
    if body_vertices.ndim != 2 or body_vertices.shape[1] != 3:
        raise ValueError("target_body_vertices must be (V, 3)")
    if body_faces.ndim != 2 or body_faces.shape[1] != 3:
        raise ValueError("target_body_faces must be triangular (F, 3)")
    if np.any((body_faces < 0) | (body_faces >= len(body_vertices))):
        raise ValueError("target_body_faces contains an out-of-range vertex")
    if not all(np.isfinite(normalized[name]).all() for name in (
        "initial_transfer_pos", "source_aligned", "guide_knn_weights",
        "guide_ori_laplacian", "normal_knn_weights",
        "normal_ori_laplacian", "fit_local_weights", "lap_local_weights",
        "source_mesh_roots", "target_body_vertices",
        "target_mesh_roots",
    )):
        raise ValueError("QP objective bundle contains non-finite values")
    return normalized


def save_qp_objective_bundle(path, **payload):
    """Validate and atomically write a canonical objective bundle."""
    complete = dict(payload)
    complete.setdefault("schema_version", np.asarray(SCHEMA_VERSION, dtype=np.int32))
    complete.setdefault("point_unit", np.asarray(POINT_UNIT))
    normalized = validate_qp_objective_bundle(complete)
    serializable = {
        key: np.asarray(value) if key not in (
            "schema_version", "point_unit", "local_weight_mode",
            "w_fid", "w_lap", "w_lap_normal", "use_shape", "use_fit",
            "use_lap",
        ) else np.asarray(value)
        for key, value in normalized.items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **serializable)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_qp_objective_bundle(path, expected_local_weight_mode=None):
    with np.load(Path(path), allow_pickle=False) as payload:
        copied = {key: payload[key] for key in payload.files}
    return validate_qp_objective_bundle(copied, expected_local_weight_mode)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_padded_objective(bundle, l_max=40):
    """Map native variable-length QP point ids to ADMM padded point ids."""
    data = validate_qp_objective_bundle(bundle)
    starts = data["hair_starts"]
    lengths = data["hair_lengths"]
    if np.any(lengths > l_max):
        longest = int(lengths.max())
        raise ValueError(
            f"strict tensor mode cannot truncate a QP strand of length {longest} "
            f"to ADMM L_MAX={l_max}"
        )
    strand_count = len(starts)
    point_count = len(data["initial_transfer_pos"])
    flat_to_padded = np.full(point_count, -1, dtype=np.int64)
    initial_padded = np.zeros((strand_count, l_max, 3), dtype=np.float64)
    source_padded = np.zeros_like(initial_padded)
    fit_local_padded = np.zeros((strand_count, l_max), dtype=np.float64)
    lap_local_padded = np.zeros((strand_count, l_max), dtype=np.float64)
    for strand, (start, length) in enumerate(zip(starts, lengths)):
        start = int(start)
        length = int(length)
        rows = slice(start, start + length)
        padded_ids = strand * l_max + np.arange(length, dtype=np.int64)
        flat_to_padded[rows] = padded_ids
        initial_padded[strand, :length] = data["initial_transfer_pos"][rows]
        source_padded[strand, :length] = data["source_aligned"][rows]
        fit_local_padded[strand, :length] = data["fit_local_weights"][rows]
        lap_local_padded[strand, :length] = data["lap_local_weights"][rows]
        initial_padded[strand, length:] = initial_padded[strand, length - 1]
        source_padded[strand, length:] = source_padded[strand, length - 1]
        fit_local_padded[strand, length:] = fit_local_padded[strand, length - 1]
        lap_local_padded[strand, length:] = lap_local_padded[strand, length - 1]

    normal_knn_native = data["normal_knn_indices"]
    normal_k = normal_knn_native.shape[1]
    normal_knn_padded = np.full(
        (strand_count * l_max, normal_k), -1, dtype=np.int64
    )
    normal_weights_padded = np.zeros(
        (strand_count * l_max, normal_k), dtype=np.float64
    )
    normal_lap_padded = np.zeros((strand_count * l_max, 3), dtype=np.float64)
    valid = normal_knn_native >= 0
    mapped_normal = np.full_like(normal_knn_native, -1)
    mapped_normal[valid] = flat_to_padded[normal_knn_native[valid]]
    normal_knn_padded[flat_to_padded] = mapped_normal
    normal_weights_padded[flat_to_padded] = data["normal_knn_weights"]
    normal_lap_padded[flat_to_padded] = data["normal_ori_laplacian"]

    guide_strands = data["guide_strand_indices"]
    guide_global_native = np.concatenate([
        np.arange(starts[s], starts[s] + lengths[s], dtype=np.int64)
        for s in guide_strands
    ]) if len(guide_strands) else np.empty(0, dtype=np.int64)
    global_to_compact = np.full(point_count, -1, dtype=np.int64)
    global_to_compact[guide_global_native] = np.arange(
        len(guide_global_native), dtype=np.int64
    )
    guide_knn_native = data["guide_knn_indices"]
    guide_knn_compact = np.full_like(guide_knn_native, -1)
    valid = guide_knn_native >= 0
    guide_knn_compact[valid] = global_to_compact[guide_knn_native[valid]]
    if np.any(guide_knn_compact[valid] < 0):
        raise ValueError("guide KNN could not be mapped to compact guide ids")
    guide_lengths = lengths[guide_strands].astype(np.int32)
    guide_starts = np.r_[0, np.cumsum(guide_lengths[:-1])].astype(np.int64)
    guide_global_padded = flat_to_padded[guide_global_native]

    return {
        **data,
        "strand_count": strand_count,
        "initial_padded": initial_padded,
        "source_padded": source_padded,
        "fit_local_weights_padded": fit_local_padded.reshape(-1),
        "lap_local_weights_padded": lap_local_padded.reshape(-1),
        "normal_knn_padded": normal_knn_padded,
        "normal_knn_weights_padded": normal_weights_padded,
        "normal_ori_laplacian_padded": normal_lap_padded,
        "guide_global_ids_padded": guide_global_padded,
        "guide_starts": guide_starts,
        "guide_lengths": guide_lengths,
        "guide_knn_compact": guide_knn_compact,
    }
