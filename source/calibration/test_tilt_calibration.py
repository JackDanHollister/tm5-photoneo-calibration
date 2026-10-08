"""Synthetic ground truth and failure gates for the offline direct fit."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyse_tilt_calibration as analysis
from check_translation_support import joint_fit


class DirectCalibrationTests(unittest.TestCase):
    def records(self, rotations):
        camera = analysis.fit.pose_matrix([-80, -100, 85, 15, -12, -80])
        board = analysis.fit.pose_matrix([500, 120, 20, 179, 2, -88])
        y, x = np.mgrid[0:8, 0:10]
        points = np.c_[x.ravel() * 20., y.ravel() * 20., np.zeros(x.size)]
        records = {}
        for index, (rx, ry) in enumerate(rotations):
            flange = analysis.fit.pose_matrix([410, -30, 585, 180 + rx, ry, 90])
            cb = np.linalg.inv(flange @ camera) @ board
            measured = points @ cb[:3, :3].T + cb[:3, 3]
            records[str(index)] = {'T_base_flange': flange.tolist(),
                                   'T_photoneo_board_metric': cb.tolist(),
                                   'photoneo_metric_points_mm': measured.tolist(),
                                   'nominal_board_points_mm': points.tolist()}
        return records, camera, board

    def test_recovers_known_transform_and_unseen_pose(self):
        records, camera, board = self.records([(0, 0), (-8, 0), (8, 0), (0, -8), (0, 8), (5, -5)])
        train = list(records)[:-1]
        actual, fixed_board, diagnostics = analysis.fit_records(records, train)
        self.assertTrue(diagnostics['meets_original_excitation_gate'])
        np.testing.assert_allclose(actual, camera, atol=1e-6)
        np.testing.assert_allclose(fixed_board, board, atol=1e-6)
        check = analysis.score_records(records, actual, fixed_board, train)['5']
        self.assertEqual(check['role'], 'held_out')
        self.assertLess(check['all_corner_error_mm']['max'], 1e-6)

    def test_translation_only_and_single_rotation_axis_rejected(self):
        for angles in [[(0, 0)] * 4, [(0, 0), (-8, 0), (8, 0), (4, 0)]]:
            records, _, _ = self.records(angles)
            # Translation changes cannot rescue missing rotational excitation.
            for i, record in enumerate(records.values()):
                record['T_base_flange'][0][3] += i * 10
            with self.assertRaisesRegex(ValueError, 'orientation diversity'):
                analysis.fit_records(records, list(records), exploratory=True)

    def test_limited_angles_require_explicit_exploratory_mode(self):
        records, camera, _ = self.records([(0, 0), (-8, 0), (8, 0), (0, -3), (0, 3)])
        with self.assertRaisesRegex(ValueError, 'orientation diversity'):
            analysis.fit_records(records, list(records))
        result, _, diagnostics = analysis.fit_records(records, list(records), exploratory=True)
        self.assertFalse(diagnostics['meets_original_excitation_gate'])
        np.testing.assert_allclose(result, camera, atol=1e-6)

    def test_raw_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw.json'
            raw.write_text('{"original": true}')
            view = root / 'views/origin'
            view.mkdir(parents=True)
            record = {'label': 'origin', 'input_files': {'test': {'path': 'raw.json', 'sha256': analysis.fit.sha256(raw)}}}
            (view / 'view.json').write_text(json.dumps(record))
            (view / 'manifest_sha256.json').write_text(json.dumps({'view.json': analysis.fit.sha256(view / 'view.json')}))
            analysis.load_views(root, root / 'views', ['origin'])
            raw.write_text('{"original": false}')
            with self.assertRaisesRegex(ValueError, 'Raw source changed'):
                analysis.load_views(root, root / 'views', ['origin'])

    def test_two_batches_allow_board_to_move(self):
        records, camera, board = self.records([(0, 0), (-8, 0), (8, 0), (0, -3), (0, 3)])
        for r in records.values():
            r['batch'] = 'tilts'
        other_board = analysis.fit.pose_matrix([490, 100, 24, 177, -1, -92])
        nominal = np.array(records['0']['nominal_board_points_mm'])
        for i, offset in enumerate([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]):
            flange = analysis.fit.pose_matrix([410+offset[0], -30+offset[1], 585+offset[2], 180, 0, 90])
            cb = np.linalg.inv(flange @ camera) @ other_board
            records['trans' + str(i)] = {'batch': 'translations', 'T_base_flange': flange.tolist(),
                                         'nominal_board_points_mm': nominal.tolist(),
                                         'photoneo_metric_points_mm': (nominal @ cb[:3, :3].T + cb[:3, 3]).tolist()}
        seed = camera @ analysis.fit.pose_matrix([1, -2, 3, .2, -.1, .3])
        actual, boards, residual = joint_fit(records, list(records), seed)
        np.testing.assert_allclose(actual, camera, atol=1e-6)
        np.testing.assert_allclose(boards['tilts'], board, atol=1e-6)
        np.testing.assert_allclose(boards['translations'], other_board, atol=1e-6)
        self.assertLess(residual['max'], 1e-6)


if __name__ == '__main__':
    unittest.main()
