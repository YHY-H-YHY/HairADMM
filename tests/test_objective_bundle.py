import tempfile
import unittest
from pathlib import Path

import numpy as np

from hairs_adaption.qp_objective_bundle import (
    load_qp_objective_bundle,
    save_qp_objective_bundle,
    to_padded_objective,
)


class ObjectiveBundleTest(unittest.TestCase):
    def test_round_trip_minimal_bundle(self):
        initial = np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.01],
            [0.01, 0.0, 0.0],
            [0.01, 0.0, 0.01],
        ])
        payload = dict(
            local_weight_mode=np.asarray("corrected"),
            initial_transfer_pos=initial,
            source_aligned=initial,
            hair_starts=np.asarray([0, 2]),
            hair_lengths=np.asarray([2, 2]),
            guide_strand_indices=np.asarray([0, 1]),
            guide_knn_indices=np.asarray([[2], [3], [0], [1]]),
            guide_knn_weights=np.ones((4, 1)),
            guide_ori_laplacian=np.zeros((4, 3)),
            normal_knn_indices=np.asarray([[2], [3], [0], [1]]),
            normal_knn_weights=np.ones((4, 1)),
            normal_ori_laplacian=np.zeros((4, 3)),
            fit_local_weights=np.ones(4),
            lap_local_weights=np.ones(4),
            source_mesh_roots=initial[[0, 2]],
            target_mesh_roots=initial[[0, 2]],
            target_body_vertices=np.asarray([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
            ]),
            target_body_faces=np.asarray([[0, 1, 2]]),
            w_fid=np.asarray(1000.0),
            w_lap=np.asarray(3000.0),
            w_lap_normal=np.asarray(30000.0),
            use_shape=np.asarray(True),
            use_fit=np.asarray(True),
            use_lap=np.asarray(True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_0000.npz"
            save_qp_objective_bundle(path, **payload)
            loaded = load_qp_objective_bundle(path, "corrected")
            padded = to_padded_objective(loaded)
            self.assertEqual(padded["strand_count"], 2)
            np.testing.assert_allclose(padded["initial_padded"][:, :2], initial.reshape(2, 2, 3))


if __name__ == "__main__":
    unittest.main()
