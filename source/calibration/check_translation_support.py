#!/usr/bin/env python3
"""Offline diagnostic: do the earlier straight translations support the tilt fit?

Two independent board poses avoid assuming the board stayed put between batches.
This is a sensitivity check, not an automatically promoted calibration profile.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import analyse_tilt_calibration as a
from analyse_partial_calibration import mapped_points


def joint_fit(records, names, seed):
    groups = sorted({records[n]['batch'] for n in names})
    boards = []
    for group in groups:
        subset = [records[n] for n in names if records[n]['batch'] == group]
        b, _ = a.fit.fit_rigid(np.concatenate([r['nominal_board_points_mm'] for r in subset]),
                              np.concatenate([mapped_points(r, seed) for r in subset]))
        boards.append(b)
    def residual(parameters):
        x = a.unpack(parameters[:6])
        bs = {g: a.unpack(parameters[6 + 6*i:12 + 6*i]) for i, g in enumerate(groups)}
        values = []
        for name in names:
            r = records[name]; b = bs[r['batch']]
            nominal = np.array(r['nominal_board_points_mm'])
            values.append((mapped_points(r, x) - nominal @ b[:3, :3].T - b[:3, 3]).ravel())
        return np.concatenate(values)
    result = least_squares(residual, np.concatenate([a.pack(seed)] + [a.pack(b) for b in boards]),
                           loss='soft_l1', f_scale=.3, x_scale='jac', max_nfev=300,
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if not result.success:
        raise ValueError('Combined diagnostic did not converge')
    return a.unpack(result.x[:6]), {g: a.unpack(result.x[6+6*i:12+6*i]) for i, g in enumerate(groups)}, a.fit.stats(np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1))


def run(root, evidence, output):
    partial = a.read(evidence / 'partial_analysis/offline_candidate.json')
    records = a.load_views(root, evidence / 'views', partial['fit_labels'])
    for r in records.values():
        r['batch'] = 'tilts'
    source = root / 'calibration_10mm/analysis_v3'
    for rel, digest in a.read(source / 'manifest_sha256.json').items():
        p = (source / rel).resolve()
        if not p.is_relative_to(source.resolve()) or a.fit.sha256(p) != digest:
            raise ValueError('Translation artifact hash mismatch')
    translation_names = ['origin_start', 'x_plus', 'y_plus', 'z_away', 'x_minus', 'y_minus', 'z_toward', 'origin_end']
    source_hashes = {}
    for name in translation_names:
        mapping = a.read(source / 'prepared' / name / 'mapping.json')
        for item in mapping['files'].values():
            p = (root / item['path']).resolve()
            if not p.is_relative_to(root.resolve()) or a.fit.sha256(p) != item['sha256']:
                raise ValueError('Translation raw source mismatch')
        path = source / 'fits' / name / 'correspondences.json'
        points = a.read(path); source_hashes[str(path.relative_to(root))] = a.fit.sha256(path)
        meta = a.read(root / mapping['files']['photoneo_metadata']['path'])
        nominal = np.array([[v['board_xy'][0]*20., v['board_xy'][1]*20., 0.] for v in points])
        measured = np.array([v['photoneo_plane_xyz_mm'] for v in points])
        # Recheck using only Photoneo pixels; ignore prior relative/EIH fit flags.
        cam = meta['frame_info']['CurrentCamera']['PerspectiveSettings']
        _, keep, _ = a.fit.fit_pose(nominal, np.array([v['photoneo_uv'] for v in points]),
                                   np.array(cam['CameraMatrix']).reshape(3, 3), np.array(cam['DistortionCoefficients']))
        cb, _ = a.fit.fit_rigid(nominal[keep], measured[keep])
        records['translation_' + name] = {'batch': 'translations', 'T_base_flange': a.pose(meta).tolist(),
                                          'T_photoneo_board_metric': cb.tolist(),
                                          'photoneo_metric_points_mm': measured[keep].tolist(),
                                          'nominal_board_points_mm': nominal[keep].tolist()}
    seed = np.array(partial['T_flange_photoneo'])
    names = list(records)
    train = a.read(evidence / 'split_v2.json')['fit_labels'] + ['translation_' + n for n in translation_names[:4]]
    check = [n for n in names if n not in train]
    sx, sb, sr = joint_fit(records, train, seed)
    checks = {n: a.score_records({n: records[n]}, sx, sb[records[n]['batch']], train)[n] for n in check}
    x, boards, rms = joint_fit(records, names, seed)
    loo = {}
    for n in names:
        cx, cb, cr = joint_fit(records, [v for v in names if v != n], x)
        shifts = np.concatenate([np.linalg.norm(mapped_points(r, cx)-mapped_points(r, x), axis=1) for r in records.values()])
        loo[n] = {'difference_from_full_data': a.fit.transform_difference(cx, x),
                  'registered_point_shift_mm': a.fit.stats(shifts),
                  'omitted_view': a.score_records({n: records[n]}, cx, cb[records[n]['batch']], [v for v in names if v != n])[n]}
    old = np.array(a.read(source / 'alignment_validation.json')['T_flange_photoneo_provisional'])
    result = {'status': 'additional_offline_sensitivity_check_not_promoted', 'accepted_for_specimen_pickup': False,
              'applied_to_robot_or_runtime': False, 'assumes_mount_unchanged_between_batches': True,
              'assumes_board_fixed_between_batches': False, 'units': 'mm',
              'T_flange_photoneo': x.tolist(), 'T_base_board_by_batch': {k: v.tolist() for k, v in boards.items()},
              'all_view_fit_residual_mm': rms, 'fit_labels': names,
              'retrospective_split': {'fit_labels': train, 'check_labels': check, 'T_flange_photoneo': sx.tolist(),
                                      'fit_residual_mm': sr, 'checks': checks},
              'leave_one_view_out': loo, 'difference_from_tilts_only': a.fit.transform_difference(x, seed),
              'difference_from_previous_eih_chain': a.fit.transform_difference(x, old),
              'old_chain_registered_point_difference_mm': a.fit.stats(np.concatenate([
                  np.linalg.norm(mapped_points(r, x)-mapped_points(r, old), axis=1) for r in records.values()])),
              'source_correspondence_sha256': source_hashes,
              'source_translation_manifest_sha256': a.fit.sha256(source / 'manifest_sha256.json'),
              'source_partial_analysis_sha256': a.fit.sha256(evidence / 'partial_analysis/offline_candidate.json'),
              'script_sha256': a.fit.sha256(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    a.save(output / 'translation_support.json', result)
    a.save(output / 'prepared_records.json', records)
    a.save(output / 'manifest_sha256.json', {p.name: a.fit.sha256(p) for p in output.iterdir() if p.is_file()})
    print('Full fit', rms, 'translation', x[:3, 3])
    print('Differences', result['difference_from_tilts_only'], result['difference_from_previous_eih_chain'])
    print('Old chain point difference', result['old_chain_registered_point_difference_mm'])
    print('Held-out RMS', {n: v['all_corner_error_mm']['rms'] for n, v in checks.items()})
    print('Max leave-one-out point shift RMS', max(v['registered_point_shift_mm']['rms'] for v in loo.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.evidence.resolve(), args.output.resolve())
