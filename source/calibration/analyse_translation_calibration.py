#!/usr/bin/env python3
"""Offline analysis of the attended 2026-09-29 translation capture batch.

Uses the reviewed initial grid to predict point identities, not to constrain the
new camera transform. Splits physical captures before fitting; no device imports.
"""
import argparse
import json
import resource
from pathlib import Path

import cv2
import numpy as np

import fit_camera_alignment as fit


def read(path):
    return json.loads(path.read_text())


def pose(record):
    raw = record['feedback_trace'][0]['raw_controller_values']['Coord_Robot_Flange']
    return fit.pose_matrix([float(v) for v in raw.strip('{}').split(',')])


def detect_identify(image_path, predicted, board_ids):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # Coded cells sometimes prevent a large seed rectangle from being found.
    # Smaller seeds may grow with LARGER; every result uses the same identity
    # and integer-grid checks below, regardless of the seed rectangle.
    for shape in [(9, 6), (6, 9), (5, 4), (4, 5)]:
        for seed in (0, 1, 4):
            cv2.setRNGSeed(seed)
            ok, corners, meta = cv2.findChessboardCornersSBWithMeta(
                image, shape, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER)
            if not ok:
                continue
            pixels = corners.reshape(-1, 2).astype(float)
            distance = np.linalg.norm(pixels[:, None] - predicted[None], axis=2)
            index = distance.argmin(1)
            close = distance.min(1) < 8
            if close.sum() < max(40, 0.85 * len(pixels)):
                continue
            row, col = np.indices(meta.shape)
            grid = np.c_[row.ravel(), col.ravel(), np.ones(row.size)]
            affine = np.linalg.lstsq(grid[close], board_ids[index[close]], rcond=None)[0].T
            rounded = np.rint(affine).astype(int)
            assigned = grid @ rounded.T
            if not np.allclose(affine, rounded, atol=0.03):
                continue
            if not np.array_equal(rounded[:, :2] @ rounded[:, :2].T, np.eye(2)):
                continue
            agree = np.all(assigned[close] == board_ids[index[close]], axis=1)
            if agree.mean() < 0.99 or len(np.unique(assigned, axis=0)) != len(assigned):
                continue
            return {'detected_grid_shape': list(meta.shape), 'corners_image_px': pixels.tolist(),
                    'seed': seed, 'seed_pattern': list(shape),
                    'row_col_to_board_xy': rounded.tolist(), 'identity_match_count': int(close.sum()),
                    'identity_match_residual_px': fit.stats(distance.min(1)[close]),
                    'identity_basis': 'Projection of reviewed fixed-board IDs into recorded camera pose; nearest match gate 8 px, integer grid-affine agreement >=99 percent. Pose fit is unconstrained after assigning IDs.'}
    raise RuntimeError('Could not identify grid reliably: ' + str(image_path))


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    initial = root / 'board_calibration_initial'
    reference = read(initial / 'alignment_draft/reviewed_mapping.json')
    for key, record in reference['files'].items():
        path = (initial / record['path']).resolve()
        if not path.is_relative_to(initial.resolve()) or fit.sha256(path) != record['sha256']:
            raise ValueError('Reference input path/hash mismatch: ' + key)
    for name, digest in read(initial / 'alignment_draft_v1/manifest_sha256.json').items():
        if fit.sha256(initial / 'alignment_draft_v1' / name) != digest:
            raise ValueError('Reference fit artifact hash mismatch: ' + name)
    old = read(initial / 'alignment_draft_v1/alignment.json')
    calibration = read(initial / 'eih_calibration_readback.json')
    h = calibration['getHandEyeParameters']['HandEyeArray']
    flange_eih = fit.pose_matrix([h['handeye_' + k] for k in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
    intr = [v for v in calibration['getIntrinsics']['cam_intrinsics'] if v['FocusValue'] == 8 and v['ImageWidth'] == 2592][0]
    k = intr['CameraMatrix']
    ek = np.array([[k.get(f'matrix_{i}{j}', 0) for j in range(3)] for i in range(3)])
    ed = np.array([intr['DistortionCoefficients'].get(f'coefficient_{i}0', 0) for i in range(5)])
    grid = read(initial / reference['files']['eih_grid']['path'])
    row, col = np.indices(grid['detected_grid_shape'])
    ids = np.c_[row.ravel(), col.ravel(), np.ones(row.size)] @ np.array(reference['eih']['row_col_to_board_xy']).T
    eih_board, _, _ = fit.fit_pose(np.c_[ids * 20, np.zeros(len(ids))], np.array(grid['corners_image_px']), ek, ed)
    base_board = pose(read(initial / reference['files']['eih_pose']['path'])) @ flange_eih @ eih_board
    board_y, board_x = np.mgrid[0:12, 0:17]
    all_ids = np.c_[board_x.ravel(), board_y.ravel()]
    board_xyz = np.c_[all_ids * 20., np.zeros(len(all_ids))]
    # This split is fixed before new fits or quality metrics are observed.
    train = ['origin_start', 'x_plus', 'y_plus', 'z_away']
    test = ['x_minus', 'y_minus', 'z_toward', 'origin_end']
    save = lambda p, v: p.write_text(json.dumps(v, indent=2) + '\n')
    save(output / 'split.json', {'fit': train, 'held_out': test, 'fixed_before_new_fits': True})
    batches, per_view = {}, {}
    for label in train + test:
        capture = root / 'calibration_10mm/captures' / label
        pair = read(capture / 'pair.json')
        eih_dir = Path(pair['eih'])
        ep = read(eih_dir / 'coordinates.json')
        pp = read(capture / 'photoneo/scan_001.json')
        if not ep['accepted'] or not pp['pose_association_accepted']:
            raise RuntimeError('Rejected capture: ' + label)
        if np.linalg.norm(pose(ep)[:3, 3] - pose(pp)[:3, 3]) > 0.05:
            raise RuntimeError('Robot moved between paired images')
        prepared = output / 'prepared' / label
        prepared.mkdir(parents=True)
        pcam = pp['frame_info']['CurrentCamera']['PerspectiveSettings']
        pk = np.array(pcam['CameraMatrix']).reshape(3, 3)
        pd = np.array(pcam['DistortionCoefficients'])
        mapping = {'schema': 'workcell.reviewed_grid_correspondence.v1', 'mapping_reviewed': True,
                   'square_pitch_mm': 20, 'review_basis': 'IDs transferred from hash-bound, visually reviewed initial board using recorded poses and strict integer-grid consistency. Seed calibration only assigns identity; it does not constrain the fitted relative pose.'}
        image_paths = {'eih': eih_dir / 'image.png', 'photoneo': capture / 'photoneo/scan_001_texture.png'}
        for name, camera_pose, k, d in [('eih', pose(ep) @ flange_eih, ek, ed),
                                      ('photoneo', pose(pp) @ flange_eih @ np.array(old['T_eih_photoneo']), pk, pd)]:
            predicted = fit.project(board_xyz, np.linalg.inv(camera_pose) @ base_board, k, d)
            identified = detect_identify(image_paths[name], predicted, all_ids)
            save(prepared / (name + '_grid.json'), identified)
            mapping[name] = {'grid_shape': identified['detected_grid_shape'],
                             'row_col_to_board_xy': identified['row_col_to_board_xy']}
        files = {'eih_image': image_paths['eih'], 'eih_pose': eih_dir / 'coordinates.json',
                 'eih_calibration': initial / 'eih_calibration_readback.json',
                 'photoneo_image': image_paths['photoneo'], 'photoneo_metadata': capture / 'photoneo/scan_001.json',
                 'photoneo_cloud': capture / 'photoneo/scan_001.npz',
                 'eih_grid': prepared / 'eih_grid.json', 'photoneo_grid': prepared / 'photoneo_grid.json'}
        mapping['files'] = {key: {'path': str(path.relative_to(root)), 'sha256': fit.sha256(path)} for key, path in files.items()}
        save(prepared / 'mapping.json', mapping)
        destination = output / 'fits' / label
        report = fit.run(root, prepared / 'mapping.json', destination)
        correspondences = read(destination / 'correspondences.json')
        # Keep all identified corners here, including single-view residual outliers.
        xyz = np.array([r['photoneo_plane_xyz_mm'] for r in correspondences])
        uv = np.array([r['eih_uv'] for r in correspondences])
        batches[label] = (xyz, uv)
        per_view[label] = {'common_points': report['common_points'],
                           'single_view_fit': report['relative_fit'],
                           'transform': report['T_eih_photoneo'],
                           'plane': report['depth_plane'],
                           'flange_pose': ep['coordinates']['flange_in_robot_base']}
        print(label, 'identified', len(xyz), 'shared corners', flush=True)
    xyz = np.concatenate([batches[v][0] for v in train])
    uv = np.concatenate([batches[v][1] for v in train])
    transform, keep, metrics = fit.fit_pose(xyz, uv, ek, ed)
    validation = {}
    for label in train + test:
        xyz, uv = batches[label]
        error = np.linalg.norm(fit.project(xyz, transform, ek, ed) - uv, axis=1)
        validation[label] = {'role': 'fit' if label in train else 'held_out',
                             'all_identified_point_error_px': fit.stats(error),
                             'points_within_1_5_px': int((error <= 1.5).sum()),
                             'points_within_3_px': int((error <= 3).sum()),
                             'independent_pose_fit_difference': fit.transform_difference(transform, np.array(per_view[label]['transform']))}
    # Base-space agreement from measured planes is a separate consistency check.
    planes = []
    flange_photoneo = flange_eih @ transform
    for label in train + test:
        capture = root / 'calibration_10mm/captures' / label
        pp = read(capture / 'photoneo/scan_001.json')
        base_photo = pose(pp) @ flange_photoneo
        normal = base_photo[:3, :3] @ np.array(per_view[label]['plane']['normal_photoneo'])
        offset = per_view[label]['plane']['offset_mm'] + normal @ base_photo[:3, 3]
        planes.append({'label': label, 'normal_base': normal.tolist(), 'plane_offset_base_mm': float(offset)})
    reference_point = base_board[:3, 3]
    plane_distances = np.array([np.dot(p['normal_base'], reference_point) - p['plane_offset_base_mm'] for p in planes])
    summary = {'schema': 'workcell.translation_camera_alignment.v1', 'status': 'local_translation_validation_only',
               'accepted_for_specimen_pickup': False, 'applied_to_controller_or_runtime': False,
               'units': 'mm', 'transform_convention': 'T_A_B maps column vectors in B into A',
               'fit_labels': train, 'held_out_labels': test, 'fit_metrics': metrics,
               'validation': validation, 'T_eih_photoneo': transform.tolist(),
               'T_flange_photoneo_provisional': flange_photoneo.tolist(),
               'optical_axis_tilt_to_flange_z_deg': float(np.degrees(np.arccos(flange_photoneo[2, 2]))),
               'difference_from_initial_draft': fit.transform_difference(transform, np.array(old['T_eih_photoneo'])),
               'base_planes': planes,
               'base_plane_distance_at_reference_point_mm': plane_distances.tolist(),
               'base_plane_distance_peak_to_peak_mm': float(np.ptp(plane_distances)),
               'limit': 'All robot orientations fixed, only +/-10 mm translation neighbourhood. Relative camera transform fitted from paired metric 3D/2D data, not an independently solved robot hand-eye transform. Factory EIH intrinsics and flange transform systematic errors remain. Held-out images have independent acquisitions but share the target, intrinsics and detector. No collision/CoG/TCP qualification.',
               'script_sha256': fit.sha256(Path(__file__))}
    save(output / 'alignment_validation.json', summary)
    save(output / 'per_view.json', per_view)
    artifacts = [p for p in output.rglob('*') if p.is_file()]
    save(output / 'manifest_sha256.json', {str(p.relative_to(output)): fit.sha256(p) for p in artifacts})
    print(json.dumps({k: summary[k] for k in ['status', 'fit_metrics', 'validation',
                                             'T_flange_photoneo_provisional', 'base_plane_distance_peak_to_peak_mm']}, indent=2))


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    cv2.setNumThreads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
