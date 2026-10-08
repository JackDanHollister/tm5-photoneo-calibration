#!/usr/bin/env python3
"""Offline sensitivity analysis of an explicitly selected incomplete board batch.

This script reads local files only. It never configures the robot or a runtime.
The full-data estimate is exploratory; its training residual is not validation.
"""
import argparse
import resource
from pathlib import Path

import cv2
import numpy as np

import analyse_tilt_calibration as analysis


def mapped_points(record, transform):
    gx = np.array(record['T_base_flange']) @ transform
    return np.array(record['photoneo_metric_points_mm']) @ gx[:3, :3].T + gx[:3, 3]


def run(root, evidence, output):
    split = analysis.read(evidence / 'split_v2.json')
    names = split['fit_labels'] + split['held_out_labels']
    records = analysis.load_views(root, evidence / 'views', names)
    x, board, diagnostics = analysis.fit_records(records, names, exploratory=True)
    loo = {}
    for omitted in names:
        train = [n for n in names if n != omitted]
        candidate, fixed_board, metrics = analysis.fit_records(records, train, exploratory=True)
        point_shifts = np.concatenate([np.linalg.norm(mapped_points(r, candidate) - mapped_points(r, x), axis=1)
                                       for r in records.values()])
        loo[omitted] = {
            'fit_labels': train, 'T_flange_photoneo': candidate.tolist(),
            'difference_from_full_data': analysis.fit.transform_difference(candidate, x),
            'registered_point_shift_from_full_data_mm': analysis.fit.stats(point_shifts),
            'omitted_view': analysis.score_records({omitted: records[omitted]}, candidate, fixed_board, train)[omitted],
            'fit_diagnostics': metrics}
    alternate = analysis.load_views(root, evidence / 'corner_method_check', [n for n in names if n != 'x_p8'])
    alternate['x_p8'] = records['x_p8']
    ax, ab, ad = analysis.fit_records(alternate, names, exploratory=True)
    old = analysis.read(root / 'calibration_10mm/analysis_v3/alignment_validation.json')
    old_x = np.array(old['T_flange_photoneo_provisional'])
    result = {
        'schema': 'workcell.offline_partial_handeye.v1',
        'status': 'exploratory_only_incomplete_validation',
        'accepted_for_specimen_pickup': False, 'applied_to_robot_or_runtime': False,
        'uses_eih_handeye_in_direct_numerical_fit': False,
        'uses_old_camera_chain_for_printed_point_identity_seeds': True,
        'units': 'mm', 'rotation': '3x3 orthonormal matrix; right-handed column vectors',
        'convention': 'T_A_B maps B into A; T_base_photoneo = T_base_flange @ T_flange_photoneo',
        'T_flange_photoneo': x.tolist(), 'T_base_board_fitted': board.tolist(),
        'fit_labels': names, 'fit_diagnostics': diagnostics,
        'training_views_not_independent_validation': analysis.score_records(records, x, board, names),
        'retrospective_check_report': 'retrospective_fit/direct_handeye.json',
        'retrospective_check_report_sha256': analysis.fit.sha256(evidence / 'retrospective_fit/direct_handeye.json'),
        'leave_one_view_out': loo,
        'all_seeded_corner_method_sensitivity': {
            'T_flange_photoneo': ax.tolist(), 'fit_diagnostics': ad,
            'difference_from_main': analysis.fit.transform_difference(ax, x),
            'registered_point_shift_mm': analysis.fit.stats(np.concatenate([
                np.linalg.norm(mapped_points(r, ax) - mapped_points(r, x), axis=1) for r in records.values()]))},
        'difference_from_previous_eih_chain': analysis.fit.transform_difference(x, old_x),
        'old_chain_registered_point_difference_mm': analysis.fit.stats(np.concatenate([
            np.linalg.norm(mapped_points(r, old_x) - mapped_points(r, x), axis=1) for r in records.values()])),
        'limits': split['limitations'] + [
            'Leave-one-out spread is an empirical sensitivity test, not an absolute-error bound or confidence interval',
            'Shared plane fits make corners correlated; hundreds of corners do not equal hundreds of independent poses',
            'Nominal printed board pitch 20 mm; no independent board dimensions, pose, scanner scale or robot metrology',
            'Seven-view estimate includes the two retrospective check views, whose errors validate only the earlier five-view fit',
            'No CoG, gripper TCP, collision geometry or real pin pickup qualification'],
        'input_view_sha256': {n: analysis.fit.sha256(evidence / 'views' / n / 'view.json') for n in names},
        'alternate_view_sha256': {n: analysis.fit.sha256(evidence / 'corner_method_check' / n / 'view.json') for n in names if n != 'x_p8'},
        'split_sha256': analysis.fit.sha256(evidence / 'split_v2.json'),
        'source_sha256': {p.name: analysis.fit.sha256(p) for p in [Path(__file__), Path(analysis.__file__), Path(analysis.fit.__file__)]}}
    output.mkdir(parents=True, exist_ok=False)
    analysis.save(output / 'offline_candidate.json', result)
    analysis.save(output / 'manifest_sha256.json', {p.name: analysis.fit.sha256(p) for p in output.iterdir() if p.is_file()})
    print('Full fit:', diagnostics['fit_point_residual_mm'])
    print('Translation mm:', x[:3, 3].tolist())
    print('Previous chain difference:', result['difference_from_previous_eih_chain'])
    for name, value in loo.items():
        print('Omit', name, value['difference_from_full_data'], 'omitted RMS', value['omitted_view']['all_corner_error_mm']['rms'],
              'mapped RMS', value['registered_point_shift_from_full_data_mm']['rms'])
    print('Corner method sensitivity:', result['all_seeded_corner_method_sensitivity']['difference_from_main'])


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    cv2.setNumThreads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.evidence.resolve(), args.output.resolve())
