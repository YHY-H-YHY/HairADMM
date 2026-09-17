"""Shared geometry, data, and I/O utilities for the production ADMM solver."""

from __future__ import annotations

import re
from pathlib import Path

import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


L_MAX = 40


def _parse_hair_obj(path):
    text = Path(path).read_text()
    vertex_lines = re.findall(
        r"^v ([\d\.\-eE]+ [\d\.\-eE]+ [\d\.\-eE]+)", text, re.MULTILINE
    )
    points = np.fromstring(
        "\n".join(vertex_lines), dtype=np.float32, sep=" "
    ).reshape(-1, 3)
    polyline_lines = re.findall(r"^l (.+)", text, re.MULTILINE)
    line_ends = [
        (int(part.split()[0]), int(part.split()[-1]))
        for part in polyline_lines
    ]
    starts = [0]
    for index in range(len(line_ends) - 1):
        if line_ends[index][1] != line_ends[index + 1][0]:
            starts.append(line_ends[index + 1][0] - 1)
    return points, np.asarray(starts, dtype=np.int32)


def build_padded_rods(points, hair_starts):
    strand_count = len(hair_starts)
    ends = np.empty(strand_count, dtype=np.int32)
    ends[:-1] = hair_starts[1:]
    ends[-1] = len(points)
    lengths = (ends - hair_starts).astype(np.int32)
    rods = np.zeros((strand_count, L_MAX, 3), dtype=np.float64)
    for strand in range(strand_count):
        start = int(hair_starts[strand])
        end = int(ends[strand])
        length = min(int(lengths[strand]), L_MAX)
        rods[strand, :length] = points[start : start + length]
        if length < L_MAX:
            rods[strand, length:] = points[start + length - 1]
    return rods, lengths


def write_hair_obj(out_path, rods, lengths):
    lines = []
    vertex_id = 1
    starts = []
    for strand in range(rods.shape[0]):
        length = int(lengths[strand])
        starts.append(vertex_id)
        for point in range(length):
            x, y, z = rods[strand, point] * 100.0
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            vertex_id += 1
    for strand in range(rods.shape[0]):
        length = int(lengths[strand])
        start = starts[strand]
        indices = " ".join(str(start + point) for point in range(length))
        lines.append(f"l {indices}\n")
    with open(str(out_path), "w") as output:
        output.writelines(lines)


def load_knn(case_dir):
    metadata = np.load(str(case_dir / "training" / "meta.npz"))
    knn = metadata["knn_of_pts_idx"].astype(np.int64)
    hair_starts = metadata["hair_starts"]
    strand_count = len(hair_starts)
    original_length = knn.shape[0] // strand_count if strand_count else L_MAX
    if original_length != L_MAX:
        neighbor_count = knn.shape[1]
        valid = knn >= 0
        remapped = np.where(
            valid,
            (knn // original_length) * L_MAX + (knn % original_length),
            np.int64(-1),
        )
        full = np.full(
            (strand_count * L_MAX, neighbor_count), -1, dtype=np.int64
        )
        original_rows = np.arange(strand_count * original_length)
        padded_rows = (
            (original_rows // original_length) * L_MAX
            + (original_rows % original_length)
        )
        full[padded_rows] = remapped
        knn = full
    return knn, strand_count


def _compute_invdist_weights_and_lap(source, knn):
    point_count, _ = knn.shape
    valid = (knn >= 0) & (knn < point_count)
    safe_indices = np.where(valid, knn, 0)
    neighbors = source[safe_indices]
    distances = np.linalg.norm(source[:, None, :] - neighbors, axis=2) + 1e-6
    inverse_distance = 1.0 / distances
    inverse_distance[~valid] = 0.0
    weight_sum = inverse_distance.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum < 1e-12, 1.0, weight_sum)
    weights = inverse_distance / weight_sum
    weighted_neighbors = np.sum(weights[:, :, None] * neighbors, axis=1)
    original_laplacian = source - weighted_neighbors
    return weights, original_laplacian


def _compute_per_point_weights(
    initial, strand_count, source_roots, target_roots
):
    weights = np.zeros(strand_count * L_MAX, dtype=np.float64)
    coefficient = 9.0 / (-np.log(0.2))
    for strand in range(strand_count):
        base = strand * L_MAX
        root = initial[base]
        distances = np.linalg.norm(
            initial[base : base + L_MAX] - root, axis=1
        )
        root_displacement = (
            np.linalg.norm(source_roots[strand] - target_roots[strand]) + 1e-7
        )
        sigma_squared = root_displacement * root_displacement * coefficient
        weights[base : base + L_MAX] = 1.0 - np.exp(
            -(distances**2) / sigma_squared
        )
    return weights


def _fast_read_obj(path):
    text = Path(path).read_text()
    vertex_lines = re.findall(
        r"^v ([\d\.\-eE]+ [\d\.\-eE]+ [\d\.\-eE]+)", text, re.MULTILINE
    )
    vertices = np.fromstring(
        "\n".join(vertex_lines), dtype=np.float64, sep=" "
    ).reshape(-1, 3)
    face_lines = re.findall(r"^f (.+)", text, re.MULTILINE)
    face_text = "\n".join(face_lines)
    raw_faces = (
        np.fromstring(face_text, dtype=np.int64, sep=" ")
        if "/" not in face_text
        else np.empty(0, dtype=np.int64)
    )
    if len(raw_faces) == 3 * len(face_lines):
        faces = (raw_faces.reshape(-1, 3) - 1).astype(np.int32)
    else:
        faces = np.asarray(
            [
                [int(token.split("/")[0]) - 1 for token in line.split()[:3]]
                for line in face_lines
            ],
            dtype=np.int32,
        )
    return vertices, faces


def _get_mesh_roots(roots_m, body_vertices_cm, body_faces):
    _, _, closest = igl.point_mesh_squared_distance(
        roots_m * 100.0,
        body_vertices_cm.astype(np.float64),
        body_faces.astype(np.int32),
    )
    return closest * 0.01


def _prepare_body_qpdetect(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths < 1e-12, 1.0, lengths)
    face_centroids = vertices[faces].mean(1)
    mesh_center = vertices.mean(0)
    if ((face_centroids - mesh_center) * normals).sum() < 0:
        normals = -normals
    return normals


def _load_guide_data(guide_npz, strand_count, root_positions=None):
    data = np.load(guide_npz, allow_pickle=True)
    guide_indices = data["cluster_hairguide_idx"].astype(np.int64)
    categories = data["category"].astype(np.int64)
    if len(categories) == strand_count:
        return guide_indices, categories

    from sklearn.cluster import KMeans

    guide_count = len(guide_indices)
    print(
        f"  guide NPZ N={len(categories)} != {strand_count}，"
        f"自动 KMeans 生成 {guide_count} 引导"
    )
    model = KMeans(
        n_clusters=guide_count, random_state=42, n_init=10
    ).fit(root_positions)
    new_categories = model.labels_.astype(np.int64)
    new_guide_indices = np.empty(guide_count, dtype=np.int64)
    for category in range(guide_count):
        members = np.where(new_categories == category)[0]
        distances = np.linalg.norm(
            root_positions[members] - model.cluster_centers_[category], axis=1
        )
        new_guide_indices[category] = members[np.argmin(distances)]
    return new_guide_indices, new_categories


def topology(starts: np.ndarray, lengths: np.ndarray):
    roots = starts.astype(np.int64)
    edge_start = []
    edge_end = []
    collision_edge_start = []
    collision_edge_end = []
    bend_prev = []
    bend_mid = []
    bend_next = []
    for start, length in zip(starts, lengths):
        start = int(start)
        length = int(length)
        if length >= 2:
            edge_start.extend(range(start, start + length - 1))
            edge_end.extend(range(start + 1, start + length))
        # The root and its incident edge are intentionally excluded.
        if length >= 3:
            collision_edge_start.extend(range(start + 1, start + length - 1))
            collision_edge_end.extend(range(start + 2, start + length))
            bend_prev.extend(range(start, start + length - 2))
            bend_mid.extend(range(start + 1, start + length - 1))
            bend_next.extend(range(start + 2, start + length))
    return tuple(
        np.asarray(values, dtype=np.int64)
        for values in (
            roots,
            edge_start,
            edge_end,
            collision_edge_start,
            collision_edge_end,
            bend_prev,
            bend_mid,
            bend_next,
        )
    )


def difference_matrix(point_count, start, end):
    rows = np.repeat(np.arange(len(start)), 2)
    columns = np.column_stack([start, end]).ravel()
    values = np.tile(np.asarray([-1.0, 1.0]), len(start))
    return sp.csr_matrix(
        (values, (rows, columns)), shape=(len(start), point_count), dtype=np.float64
    )


def bending_matrix(point_count, previous, middle, following):
    rows = np.repeat(np.arange(len(previous)), 3)
    columns = np.column_stack([previous, middle, following]).ravel()
    values = np.tile(np.asarray([1.0, -2.0, 1.0]), len(previous))
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(len(previous), point_count),
        dtype=np.float64,
    )


def collision_matrix(
    point_count,
    roots,
    segment_start,
    segment_end,
    line_samples,
    adaptive_samples,
):
    root_mask = np.zeros(point_count, dtype=bool)
    root_mask[roots] = True
    nonroots = np.where(~root_mask)[0]

    row_parts = [np.arange(len(nonroots), dtype=np.int64)]
    column_parts = [nonroots.astype(np.int64, copy=False)]
    value_parts = [np.ones(len(nonroots), dtype=np.float64)]
    sample_kind_parts = [np.full(len(nonroots), "point", dtype="<U8")]
    sample_edge_parts = [np.full(len(nonroots), -1, dtype=np.int64)]
    row = len(nonroots)

    if line_samples > 0 and len(segment_start):
        fractions = np.arange(1, line_samples + 1, dtype=np.float64)
        fractions /= line_samples + 1.0
        line_count = len(segment_start) * line_samples
        line_rows = np.arange(row, row + line_count, dtype=np.int64)
        line_start = np.repeat(segment_start, line_samples)
        line_end = np.repeat(segment_end, line_samples)
        line_fraction = np.tile(fractions, len(segment_start))
        row_parts.append(np.repeat(line_rows, 2))
        column_parts.append(np.column_stack([line_start, line_end]).ravel())
        value_parts.append(
            np.column_stack([1.0 - line_fraction, line_fraction]).ravel()
        )
        sample_kind_parts.append(np.full(line_count, "line", dtype="<U8"))
        sample_edge_parts.append(np.repeat(np.arange(len(segment_start)), line_samples))
        row += line_count

    seen = set()
    adaptive_edges = []
    adaptive_fractions = []
    for edge, fraction in adaptive_samples:
        key = (int(edge), int(round(float(fraction) * 10000)))
        if key in seen or not (0.0 < fraction < 1.0):
            continue
        seen.add(key)
        adaptive_edges.append(int(edge))
        adaptive_fractions.append(float(fraction))
    if adaptive_edges:
        adaptive_edges = np.asarray(adaptive_edges, dtype=np.int64)
        adaptive_fractions = np.asarray(adaptive_fractions, dtype=np.float64)
        adaptive_count = len(adaptive_edges)
        adaptive_rows = np.arange(row, row + adaptive_count, dtype=np.int64)
        adaptive_start = segment_start[adaptive_edges]
        adaptive_end = segment_end[adaptive_edges]
        row_parts.append(np.repeat(adaptive_rows, 2))
        column_parts.append(
            np.column_stack([adaptive_start, adaptive_end]).ravel()
        )
        value_parts.append(
            np.column_stack(
                [1.0 - adaptive_fractions, adaptive_fractions]
            ).ravel()
        )
        sample_kind_parts.append(
            np.full(adaptive_count, "adaptive", dtype="<U8")
        )
        sample_edge_parts.append(adaptive_edges)
        row += adaptive_count

    matrix = sp.csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(row, point_count),
    )
    return (
        matrix,
        len(nonroots),
        np.concatenate(sample_kind_parts),
        np.concatenate(sample_edge_parts),
    )


def solve_spd(matrix, rhs, initial, maxiter, rtol, preconditioner=None):
    if preconditioner is None:
        diagonal = matrix.diagonal()
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        preconditioner = sp.diags(1.0 / diagonal, format="csr")
    result = initial.copy()
    infos = []
    for coordinate in range(3):
        result[:, coordinate], info = spla.cg(
            matrix,
            rhs[:, coordinate],
            x0=result[:, coordinate],
            M=preconditioner,
            maxiter=maxiter,
            rtol=rtol,
        )
        infos.append(int(info))
    return result, infos


def collision_planes(
    samples,
    body_vertices,
    body_faces,
    face_normals,
    band,
    point_sample_count=0,
    winding_points=False,
):
    signed, face_ids, closest = igl.signed_distance(
        samples,
        body_vertices,
        body_faces,
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    )
    signed = np.asarray(signed, dtype=np.float64).reshape(-1)
    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
    normals = face_normals[face_ids]
    if winding_points and point_sample_count > 0:
        point_signed, point_faces, point_closest = igl.signed_distance(
            samples[:point_sample_count],
            body_vertices,
            body_faces,
            igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
        )
        point_signed = np.asarray(point_signed, dtype=np.float64).reshape(-1)
        point_faces = np.asarray(point_faces, dtype=np.int64).reshape(-1)
        point_closest = np.asarray(point_closest, dtype=np.float64).reshape(-1, 3)
        signed[:point_sample_count] = point_signed
        face_ids[:point_sample_count] = point_faces
        closest[:point_sample_count] = point_closest
        normals[:point_sample_count] = face_normals[point_faces]
    active = signed < band
    return signed, closest, normals, active
