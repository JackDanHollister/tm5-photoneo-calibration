#!/usr/bin/env python3
"""Offline direct Photoneo-to-flange calibration from the attended tilt batch.

Only Photoneo 3D board points and robot poses enter the direct fit. The old EIH
chain assigns reviewed printed board IDs and provides a comparison, not a prior.
"""
import argparse
import json
import resource
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import fit_camera_alignment as fit
from analyse_translation_calibration import detect_identify, pose, read


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def intrinsics(calibration):
    entry = [v for v in calibration['getIntrinsics']['cam_intrinsics']
             if v['FocusValue'] == 8 and v['ImageWidth'] == 2592 and v['ImageHeight'] == 1944][0]
    m = entry['CameraMatrix']
    k = np.array([[m.get(f'matrix_{i}{j}', 0) for j in range(3)] for i in range(3)])
    d = np.array([entry['DistortionCoefficients'].get(f'coefficient_{i}0', 0) for i in range(5)])
    return k, d


def detect_seeded_corners(path, predicted, ids):
    """Measure image corners near reviewed IDs; predictions are never measurements.

    Used explicitly for a cropped coded board that the grid detector cannot find.
    Alternating black/white quadrants reject dots, edges and absent board regions.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    lookup = {tuple(identity): point for identity, point in zip(ids, predicted)}
    pixels, identities, shifts, contrasts = [], [], [], []
    for identity, prediction in zip(ids, predicted):
        if not (12 < prediction[0] < width - 12 and 12 < prediction[1] < height - 12):
            continue
        directions = []
        for axis in range(2):
            other = identity.copy()
            step = 1 if tuple(identity + np.eye(2, dtype=int)[axis]) in lookup else -1
            other[axis] += step
            directions.append((lookup[tuple(other)] - prediction) / step)
        point = cv2.cornerSubPix(image, np.float32(prediction).reshape(1, 1, 2), (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, .001)).reshape(2)
        shift = float(np.linalg.norm(point - prediction))
        if shift > 6:
            continue
        quadrants = []
        for a, b in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            p = point + .13 * (a * directions[0] + b * directions[1])
            if not (2 < p[0] < width - 2 and 2 < p[1] < height - 2):
                break
            quadrants.append(float(cv2.getRectSubPix(image, (3, 3), tuple(p.astype(float))).mean()))
        if len(quadrants) != 4:
            continue
        q = np.array(quadrants)
        contrast = abs(float((q[0] + q[2] - q[1] - q[3]) / 2))
        if contrast < 35 or max(abs(q[0] - q[2]), abs(q[1] - q[3])) > .5 * contrast:
            continue
        pixels.append(point.tolist()); identities.append(identity.tolist())
        shifts.append(shift); contrasts.append(contrast)
    if len(pixels) < 40:
        raise ValueError('Too few measured checker corners after image-content checks')
    return {'method': 'predicted-ID seed; measured cornerSubPix 5px; alternating quadrants',
            'board_ids': identities, 'corners_image_px': pixels,
            'prediction_to_measurement_px': fit.stats(shifts),
            'quadrant_contrast': fit.stats(contrasts),
            'minimum_corners': 40, 'maximum_prediction_shift_px': 6,
            'minimum_quadrant_contrast': 35, 'predictions_used_as_measurements': False}


def prepare_view(root, label, output, photoneo_corner_method='sb'):
    source = root / 'calibration_10mm/analysis_v3'
    for relative, digest in read(source / 'manifest_sha256.json').items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or fit.sha256(path) != digest:
            raise ValueError('Reference analysis changed: ' + relative)
    registration = read(source / 'alignment_validation.json')
    reference_mapping = read(source / 'prepared/origin_start/mapping.json')
    for key, record in reference_mapping['files'].items():
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root.resolve()) or fit.sha256(path) != record['sha256']:
            raise ValueError('Reference hash mismatch: ' + key)
    co = read(source / 'fits/origin_start/correspondences.json')
    board_xyz = np.array([[*np.array(v['board_xy']) * 20., 0.] for v in co])
    photo_xyz = np.array([v['photoneo_plane_xyz_mm'] for v in co])
    photo_board, _ = fit.fit_rigid(board_xyz, photo_xyz)
    old_pose = pose(read(root / reference_mapping['files']['photoneo_metadata']['path']))
    flange_photo = np.array(registration['T_flange_photoneo_provisional'])
    base_board = old_pose @ flange_photo @ photo_board
    calibration = read(root / reference_mapping['files']['eih_calibration']['path'])
    h = calibration['getHandEyeParameters']['HandEyeArray']
    flange_eih = fit.pose_matrix([h['handeye_' + k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')])
    ek, ed = intrinsics(calibration)
    capture = root / 'calibration_tilts/captures' / label
    pair = read(capture / 'pair.json')
    eih_dir = Path(pair['eih']).resolve()
    if not eih_dir.is_relative_to(root.resolve()):
        raise ValueError('EIH capture is outside the evidence root')
    epose = read(eih_dir / 'coordinates.json')
    metadata_path = capture / 'photoneo/scan_001.json'
    photo = read(metadata_path)
    if fit.sha256(capture / 'photoneo/scan_001.npz') != photo['npz_sha256']:
        raise ValueError('Photoneo raw capture hash mismatch')
    if not epose['accepted'] or not photo['pose_association_accepted'] or photo['units'] != 'mm' or photo['coordinate_space'] != 'CameraSpace':
        raise ValueError('Capture rejected or wrong cloud convention')
    difference = fit.transform_difference(pose(epose), pose(photo))
    if difference['origin_difference_mm'] > 0.05 or difference['rotation_difference_deg'] > 0.02:
        raise ValueError('Robot moved between paired images')
    cam = photo['frame_info']['CurrentCamera']
    if not np.allclose(np.array(cam['WorldToCameraCoordinates']).reshape(3, 4), np.eye(4)[:3]):
        raise ValueError('Not primary-camera coordinates')
    pk = np.array(cam['PerspectiveSettings']['CameraMatrix']).reshape(3, 3)
    pd = np.array(cam['PerspectiveSettings']['DistortionCoefficients'])
    y, x = np.mgrid[0:12, 0:17]
    ids = np.c_[x.ravel(), y.ravel()]
    xyz = np.c_[ids * 20., np.zeros(len(ids))]
    data = {}
    files = {'photoneo_metadata': metadata_path, 'photoneo_cloud': capture / 'photoneo/scan_001.npz',
             'eih_pose': eih_dir / 'coordinates.json', 'pair': capture / 'pair.json'}
    for name, camera, k, d, path in (
            ('photoneo', pose(photo) @ flange_photo, pk, pd, capture / 'photoneo/scan_001_texture.png'),
            ('eih', pose(epose) @ flange_eih, ek, ed, eih_dir / 'image.png')):
        predicted = fit.project(xyz, np.linalg.inv(camera) @ base_board, k, d)
        detector = detect_seeded_corners if name == 'photoneo' and photoneo_corner_method == 'seeded' else detect_identify
        grid = detector(path, predicted, ids)
        if 'board_ids' in grid:
            identified = np.array(grid['board_ids'])
        else:
            row, col = np.indices(grid['detected_grid_shape'])
            identified = np.c_[row.ravel(), col.ravel(), np.ones(row.size)] @ np.array(grid['row_col_to_board_xy']).T
        pixels = np.array(grid['corners_image_px'])
        points = np.c_[identified * 20., np.zeros(len(identified))]
        pnp, keep, metrics = fit.fit_pose(points, pixels, k, d)
        data[name] = {'grid': grid, 'ids': identified, 'pixels': pixels,
                      'nominal_xyz': points, 'pnp': pnp, 'keep': keep, 'metrics': metrics}
        files[name + '_image'] = path
    p = data['photoneo']
    cloud = np.load(files['photoneo_cloud'])['PointCloud']
    measured, plane = fit.plane_points(cloud, p['pixels'], p['keep'], pk, pd)
    metric_pose, metric_fit = fit.fit_rigid(p['nominal_xyz'][p['keep']], measured[p['keep']])
    record = {'label': label, 'T_base_flange': pose(photo).tolist(),
              'T_photoneo_board_metric': metric_pose.tolist(),
              'T_eih_board_pnp': data['eih']['pnp'].tolist(),
              'photoneo_board_ids': p['ids'][p['keep']].astype(int).tolist(),
              'photoneo_metric_points_mm': measured[p['keep']].tolist(),
              'nominal_board_points_mm': p['nominal_xyz'][p['keep']].tolist(),
              'photoneo_pnp_metrics': p['metrics'], 'eih_pnp_metrics': data['eih']['metrics'],
              'metric_board_fit': metric_fit, 'depth_plane': plane,
              'input_files': {key: {'path': str(path.relative_to(root)), 'sha256': fit.sha256(path)} for key, path in files.items()},
              'reference_alignment_sha256': fit.sha256(source / 'alignment_validation.json'),
              'script_sha256': fit.sha256(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'view.json', record)
    for name, values in data.items():
        save(output / (name + '_grid.json'), values['grid'])
        image = cv2.imread(str(files[name + '_image']))
        for identity, pixel, keep in zip(values['ids'], values['pixels'], values['keep']):
            if keep and all(int(v) % 3 == 0 for v in identity):
                uv = tuple(np.rint(pixel).astype(int))
                cv2.circle(image, uv, 4, (0, 255, 255), -1)
                cv2.putText(image, ','.join(str(int(v)) for v in identity), (uv[0]+4, uv[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 80, 255), 1)
        cv2.imwrite(str(output / (name + '_identified.png')), image)
    save(output / 'manifest_sha256.json', {v.name: fit.sha256(v) for v in output.iterdir() if v.is_file()})
    print(json.dumps({'label': label, 'metric_corners': int(p['keep'].sum()),
                      'metric_board_fit': metric_fit, 'plane_rms_mm': plane['plane_residual_mm']['rms']}, indent=2), flush=True)


def unpack(parameters):
    return fit.rigid_matrix(Rotation.from_rotvec(parameters[:3]).as_matrix(), parameters[3:6])


def pack(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def load_views(root, views, names):
    records = {}
    for name in names:
        directory = (views / name).resolve()
        if not directory.is_relative_to(views.resolve()):
            raise ValueError('View outside requested directory')
        for rel, digest in read(directory / 'manifest_sha256.json').items():
            path = (directory / rel).resolve()
            if not path.is_relative_to(directory) or fit.sha256(path) != digest:
                raise ValueError('Prepared view changed: ' + name)
        records[name] = read(directory / 'view.json')
        if records[name]['label'] != name:
            raise ValueError('View label mismatch')
        for record in records[name]['input_files'].values():
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root.resolve()) or fit.sha256(path) != record['sha256']:
                raise ValueError('Raw source changed: ' + str(path))
    return records


def fit_records(records, train, exploratory=False):
    if len(train) < 3 or len(train) != len(set(train)):
        raise ValueError('Need at least three distinct poses')
    gs = [np.array(records[n]['T_base_flange']) for n in train]
    cs = [np.array(records[n]['T_photoneo_board_metric']) for n in train]
    excitation = np.vstack([g[:3, :3] - gs[0][:3, :3] for g in gs[1:]])
    singular_values = np.linalg.svd(excitation, compute_uv=False)
    # Keep the original capture-design gate. An explicit exploratory fit can
    # examine weaker data but cannot qualify it; exact degeneracy always fails.
    if singular_values.min() < (1e-4 if exploratory else .1):
        raise ValueError('Insufficient wrist orientation diversity')
    methods = {}
    for name, method in [('PARK', cv2.CALIB_HAND_EYE_PARK),
                         ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD)]:
        rotation, translation = cv2.calibrateHandEye([g[:3,:3] for g in gs], [g[:3,3] for g in gs],
                                                    [c[:3,:3] for c in cs], [c[:3,3] for c in cs], method=method)
        methods[name] = fit.rigid_matrix(rotation, translation)
    seed = methods['PARK']
    if not np.isfinite(seed).all() or np.linalg.norm(seed[:3, 3]) > 600:
        raise ValueError('Nonphysical direct seed solution')
    world, nominal = [], []
    for name, g in zip(train, gs):
        p = np.array(records[name]['photoneo_metric_points_mm'])
        world.append(p @ (g @ seed)[:3,:3].T + (g @ seed)[:3,3])
        nominal.append(np.array(records[name]['nominal_board_points_mm']))
    board_seed, _ = fit.fit_rigid(np.concatenate(nominal), np.concatenate(world))
    def residual(parameters):
        x, b = unpack(parameters[:6]), unpack(parameters[6:])
        values = []
        for name, g in zip(train, gs):
            points = np.array(records[name]['photoneo_metric_points_mm'])
            board = np.array(records[name]['nominal_board_points_mm'])
            gx = g @ x
            values.append((points @ gx[:3,:3].T + gx[:3,3] - board @ b[:3,:3].T - b[:3,3]).ravel())
        return np.concatenate(values)
    solution = least_squares(residual, np.r_[pack(seed), pack(board_seed)], loss='soft_l1', f_scale=0.3,
                             x_scale='jac', max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if not solution.success:
        raise ValueError('Joint direct calibration fit failed to converge')
    x, board = unpack(solution.x[:6]), unpack(solution.x[6:])
    if not np.isfinite(x).all() or np.linalg.norm(x[:3, 3]) > 600:
        raise ValueError('Nonphysical refined solution')
    return x, board, {
        'fit_point_residual_mm': fit.stats(np.linalg.norm(residual(solution.x).reshape(-1, 3), axis=1)),
        'orientation_excitation_singular_values': singular_values.tolist(),
        'original_design_minimum_singular_value': .1,
        'meets_original_excitation_gate': bool(singular_values.min() >= .1),
        'exploratory_limited_excitation_allowed': exploratory,
        'optimizer_evaluations': solution.nfev,
        'closed_form_methods': {k: {'transform': v.tolist(), 'difference_from_refined': fit.transform_difference(v, x)} for k, v in methods.items()}}


def score_records(records, x, board, train):
    validation = {}
    for name, r in records.items():
        g = np.array(r['T_base_flange'])
        pts = np.array(r['photoneo_metric_points_mm'])
        nominal = np.array(r['nominal_board_points_mm'])
        gx = g @ x
        errors = np.linalg.norm(pts @ gx[:3,:3].T + gx[:3,3] - nominal @ board[:3,:3].T - board[:3,3], axis=1)
        validation[name] = {'role': 'fit' if name in train else 'held_out', 'all_corner_error_mm': fit.stats(errors),
                             'board_pose_disagreement': fit.transform_difference(gx @ np.array(r['T_photoneo_board_metric']), board)}
    return validation


def solve(root, views, output, split_path=None, exploratory=False):
    # Default remains the original, prospective plan. A partial retrospective
    # analysis requires an explicit split file and is labelled as such.
    if split_path:
        split = read(split_path)
        train, test = split['fit_labels'], split['held_out_labels']
    else:
        plan = read(root / 'calibration_tilts/plan.json')
        train = ['origin_start'] + [s['label'] for s in plan['steps'] if not s['label'].startswith('return_') and s['role'] == 'fit']
        test = [s['label'] for s in plan['steps'] if not s['label'].startswith('return_') and s['role'] == 'held_out'] + ['origin_end']
        split = {'type': 'original_prospective_capture_plan'}
    if set(train) & set(test) or len(test) != len(set(test)):
        raise ValueError('Fit and check roles overlap or repeat')
    records = load_views(root, views, train + test)
    x, board, diagnostics = fit_records(records, train, exploratory)
    validation = score_records(records, x, board, train)
    old = read(root / 'calibration_10mm/analysis_v3/alignment_validation.json')
    report = {'schema': 'workcell.direct_photoneo_handeye.v1',
              'status': 'direct_fit_requires_review_of_held_out_results',
              'uses_eih_handeye_in_direct_fit': False, 'accepted_for_specimen_pickup': False,
              'applied_to_robot_or_runtime': False, 'units': 'mm',
              'convention': 'T_A_B maps B coordinates to A; use actual flange pose for each scan',
              'T_flange_photoneo': x.tolist(), 'T_base_board_training_fit': board.tolist(),
              'fit_labels': train, 'held_out_labels': test, 'validation': validation,
              **diagnostics,
              'split': split, 'split_sha256': fit.sha256(split_path) if split_path else None,
              'input_view_sha256': {n: fit.sha256(views / n / 'view.json') for n in records},
              'difference_from_eih_chain': fit.transform_difference(x, np.array(old['T_flange_photoneo_provisional'])),
              'excluded_methods': 'On exact synthetic poses, OpenCV TSAI rejected the small-angle design and ANDREFF failed rotation normalisation for fixed flange position. Neither is used as a calibration estimate or comparison. PARK and HORAUD recovered the known transform.',
              'fit_method': 'PARK seed then joint T_flange_camera / fixed T_base_board metric 3D least squares; soft_l1 0.3 mm, factory scanner intrinsics, nominal 20 mm board pitch. Held-out orientations never enter fit.',
              'limits': ['Same board and limited wrist-angle neighbourhood', 'No independent metrology of board pose, dimensions or robot absolute accuracy',
                         'Printed point identity seed uses old chained transform; direct numerical fit uses only Photoneo metric points and recorded flange poses',
                         'Optical calibration does not determine centre of mass or qualify gripper TCP'],
              'script_sha256': fit.sha256(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'direct_handeye.json', report)
    save(output / 'manifest_sha256.json', {p.name: fit.sha256(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    cv2.setNumThreads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['view', 'solve'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label')
    parser.add_argument('--views', type=Path)
    parser.add_argument('--photoneo-corner-method', choices=['sb', 'seeded'], default='sb')
    parser.add_argument('--split', type=Path)
    parser.add_argument('--exploratory-limited-excitation', action='store_true')
    args = parser.parse_args()
    if args.action == 'view': prepare_view(args.root.resolve(), args.label, args.output.resolve(), args.photoneo_corner_method)
    else: solve(args.root.resolve(), args.views.resolve(), args.output.resolve(), args.split, args.exploratory_limited_excitation)
