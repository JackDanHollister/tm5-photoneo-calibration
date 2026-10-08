#!/usr/bin/env python3
"""Relate the nominal vendor housing mesh to saved Photoneo optical calibration.

Reads files only. The vendor sensor-link mesh placement and optical-window
checks support a nominal visual registration, not certified mounting metrology.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from preview_wrist_mount import PROJECT, WORKSPACE, digest, read_binary_stl


def checked_json(path):
    return json.loads(path.read_text())


def ray_intersections(vertices, faces, origin, direction):
    triangles = vertices[faces]
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    p = np.cross(direction, edge2)
    determinant = np.einsum('ij,ij->i', edge1, p)
    good = np.abs(determinant) > 1e-9
    inverse = np.zeros(len(determinant))
    inverse[good] = 1 / determinant[good]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum('ij,ij->i', s, p)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    t = inverse * np.einsum('ij,ij->i', edge2, q)
    valid = good & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return [{'distance_mm': float(value), 'point_mm': (origin + value * direction).tolist()}
            for value in sorted(set(np.round(t[valid], 8)))]


def main(root, output):
    out_cal = root / 'offline_calibration_20260930'
    profile_path = out_cal / 'photoneo_handeye_offline_candidate.json'
    calibration = checked_json(profile_path)
    if digest(profile_path) != checked_json(out_cal / 'manifest_sha256.json')[profile_path.name]:
        raise ValueError('Saved calibration changed')
    meta_path = root / 'calibration_tilts/captures/origin_start/photoneo/scan_001.json'
    expected = checked_json(out_cal / 'input_manifest_sha256.json')['origin_start'][str(meta_path.relative_to(root))]
    if digest(meta_path) != expected:
        raise ValueError('Saved camera metadata changed')
    meta = checked_json(meta_path)
    if meta['coordinate_space'] != 'CameraSpace' or meta['units'] != 'mm':
        raise ValueError('Unsupported source coordinate system')
    frame = meta['frame_info']
    if not np.allclose(np.array(frame['CurrentCamera']['WorldToCameraCoordinates']).reshape(3, 4), np.eye(4)[:3]):
        raise ValueError('Saved point cloud is not in primary-camera coordinates')
    vendor = WORKSPACE / 'projects/tm5s-robot-arm-gui/perception/phoxi_ws/src/PhoXi-ROS-API/phoxi_camera_description'
    mesh_path = vendor / 'meshes/visual/MotionCam-3D-Color-S.stl'
    xacro_path = vendor / 'urdf/phoxi_camera.urdf.xacro'
    model_path = vendor / 'urdf/models/MotionCam-3D-Color-S.xacro'
    link = ET.parse(xacro_path).getroot().find("link[@name='phoxi_camera_sensor']")
    origin = link.find('visual/origin')
    if any(float(v) for key in ['xyz', 'rpy'] for v in origin.attrib[key].split()):
        raise ValueError('Vendor visual no longer has identity sensor-link placement')
    if link.find('visual/geometry/mesh').get('scale') != '0.001 0.001 0.001':
        raise ValueError('Vendor mesh unit convention changed')
    if 'MotionCam-3D-Color-S.stl' not in model_path.read_text():
        raise ValueError('Wrong model selection')
    vertices, faces = read_binary_stl(mesh_path, 1.)
    low, high = vertices.min(0), vertices.max(0)
    if not np.allclose(high - low, [308., 67.90, 84.954], atol=.2):
        raise ValueError('Unexpected housing size')
    camera = np.array(calibration['T_flange_photoneo'])
    colour_inverse = np.eye(4)
    colour_inverse[:3] = np.array(frame['CurrentColorCamera']['WorldToCameraCoordinates']).reshape(3, 4)
    colour = np.linalg.inv(colour_inverse)
    colour_position = np.array([frame['ColorCameraPosition'][k] for k in ['x', 'y', 'z']])
    if not np.allclose(colour[:3, 3], colour_position):
        raise ValueError('RGB factory extrinsics disagree with frame position')
    hits = {'primary': ray_intersections(vertices, faces, np.zeros(3), np.array([0., 0., 1.])),
            'colour': ray_intersections(vertices, faces, colour[:3, 3], colour[:3, 2])}
    # Both forward axes must exit through the nominal front-window region.
    for name, values in hits.items():
        if len(values) != 2 or not all(5.5 < v['point_mm'][2] < 7.2 for v in values):
            raise ValueError('Optical axis/window check needs review: ' + name)
    centre = (low + high) / 2
    centre_flange = camera[:3, :3] @ centre + camera[:3, 3]
    mesh_flange_m = camera.copy()
    mesh_flange_m[:3, 3] /= 1000
    sources = {str(path): digest(path) for path in [profile_path, meta_path, mesh_path, xacro_path, model_path]}
    report = {
        'schema': 'workcell.photoneo_body_registration.v1', 'status': 'estimated_optical_pose_with_nominal_vendor_housing',
        'scope': 'Offline visual registration only; no controller, physical collision or centre-of-mass qualification',
        'hardware_commands': False, 'applied_to_robot_or_runtime': False,
        'vendor_commit': 'abc843c89e873182b0de69cffa2b9f4c880e7796',
        'mesh_origin_policy': 'Retain native vendor mesh coordinates; never replace optical origin with housing bbox centre',
        'nominal_T_primary_camera_mesh_mm': np.eye(4).tolist(),
        'T_flange_primary_camera_mm': camera.tolist(), 'T_flange_mesh_m': mesh_flange_m.tolist(),
        'T_primary_colour_camera_mm': colour.tolist(), 'T_flange_colour_camera_mm': (camera @ colour).tolist(),
        'native_mesh_bounds_mm': [low.tolist(), high.tolist()], 'native_bbox_centre_mm': centre.tolist(),
        'body_bbox_centre_flange_mm': centre_flange.tolist(),
        'camera_forward_flange': camera[:3, 2].tolist(),
        'camera_forward_tilt_from_flange_z_deg': float(np.degrees(np.arccos(camera[2, 2]))),
        'body_long_axis_in_flange': camera[:3, 0].tolist(),
        'body_clockwise_angle_from_flange_plus_y_deg': float(np.degrees(np.arctan2(-centre_flange[0], centre_flange[1])) % 360),
        'optical_axes_intersect_nominal_front_windows': hits,
        'basis': [
            'Saved CameraSpace / PrimaryCamera output has identity world-to-camera matrix',
            'Vendor xacro places native millimetre STL at identity in phoxi_camera_sensor link',
            'Primary and saved factory RGB optical rays intersect the corresponding nominal window surfaces',
            'Nominal housing dimensions agree with Color S drawings; manufacturer CAD is not unit-specific mechanical metrology'],
        'source_urls': {
            'vendor_model': 'https://github.com/photoneo/PhoXi-ROS-API/tree/abc843c89e873182b0de69cffa2b9f4c880e7796/phoxi_camera_description',
            'coordinate_manual': 'https://docs.photoneo.com/docs/3d_sensors/1.17/software/PXC/PXC_manual.html#coordinate-settings-photoneo-3d-sensor',
            'mechanical_manual': 'https://www.photoneo.com/files/dw/dw/devices/Photoneo3DSensors-UserManual05-2026-v1.0.pdf#page=43'},
        'limits': [
            'Optical pose remains the offline draft; largest measured leave-one-view-out point shift was 2.39 mm RMS, not an accuracy bound',
            'Native mesh-to-primary-camera identity is supported for nominal visualisation, not a certified per-device tolerance',
            'Housing bounding-box centre is not centre of mass, mounting-hole centre or TCP',
            'Projecting bracket geometry and exact fastener engagement are not recovered by this optical calibration',
            'Vendor demonstration base_link translation and generic inertial values are not mounting or mass measurements'],
        'source_hashes': sources, 'script_sha256': digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'body_registration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'body_bbox_centre_flange_mm': centre_flange.tolist(),
                      'tilt_deg': report['camera_forward_tilt_from_flange_z_deg'],
                      'clockwise_angle_deg': report['body_clockwise_angle_from_flange_plus_y_deg']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
