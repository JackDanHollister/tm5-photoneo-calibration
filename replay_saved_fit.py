"""Replay my saved Photoneo hand-eye fit using original fitting functions, offline."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/calibration'))
import analyse_tilt_calibration as tilt
from check_translation_support import joint_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'replayed_fit.json')
    args = parser.parse_args()
    # Fitting uses only numerical records. No camera/controller modules are imported.
    def offline(event, _):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'):
            raise RuntimeError('Saved-data fit must remain offline')
    sys.addaudithook(offline)
    records = json.loads((ROOT/'data/prepared_records.json').read_text())
    tilt_names = [name for name, row in records.items() if row['batch']=='tilts']
    seed, _, seed_diagnostics = tilt.fit_records(records, tilt_names, exploratory=True)
    fitted, boards, residual = joint_fit(records, list(records), seed)
    reference = json.loads((ROOT/'results/photoneo_handeye_offline_candidate.json').read_text())
    difference = tilt.fit.transform_difference(fitted, np.array(reference['T_flange_photoneo']))
    if difference['origin_difference_mm'] > .001 or difference['rotation_difference_deg'] > .0001:
        raise ValueError('Saved calibration was not reproduced: '+str(difference))
    report = {
        'software': 'Custom Python; OpenCV PARK initialisation with HORAUD comparison, SciPy soft-L1 metric refinement',
        'manufacturer_robot_calibration_tool_used': False,
        'T_flange_photoneo': fitted.tolist(), 'units': 'mm',
        'convention': 'T_A_B maps homogeneous column-vector points from B into A',
        'registration': 'p_base = T_base_flange @ T_flange_photoneo @ p_primary_camera',
        'coordinate_space': 'Saved Photoneo CameraSpace / PrimaryCamera optical frame',
        'fit_residual_mm': residual, 'views': len(records),
        'T_base_board_by_batch': {name: value.tolist() for name, value in boards.items()},
        'difference_from_saved_estimate': difference,
        'tilt_excitation': seed_diagnostics,
        'physical_accuracy_validated': False,
        'limits': 'Reproduces original limited-excitation development fit; numerical residual is not absolute accuracy',
        'hardware_contact': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'views': len(records), 'fit_residual_mm': residual,
                      'difference_from_saved_estimate': difference, 'hardware_contact': False}, indent=2))


if __name__ == '__main__':
    main()
