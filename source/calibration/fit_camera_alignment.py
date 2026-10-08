#!/usr/bin/env python3
"""Fit a provisional camera alignment from one saved, identified TM board pair.

Offline only: reads hash-bound files and writes a new output directory. No device
or robot imports, controller writes, runtime configuration, or motion interface.
The mapping is specific to reviewed images, not an automatic TM code decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def pose_matrix(pose_mm_deg: list[float]) -> np.ndarray:
    rx, ry, rz = np.radians(pose_mm_deg[3:])
    sx, sy, sz = np.sin([rx, ry, rz])
    cx, cy, cz = np.cos([rx, ry, rz])
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rigid_matrix(z @ y @ x, np.array(pose_mm_deg[:3]))


def project(xyz: np.ndarray, transform: np.ndarray, k: np.ndarray,
            distortion: np.ndarray) -> np.ndarray:
    return cv2.projectPoints(xyz, cv2.Rodrigues(transform[:3, :3])[0],
                             transform[:3, 3], k, distortion)[0].reshape(-1, 2)


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {"count": len(values), "rms": float(np.sqrt(np.mean(values**2))),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def fit_pose(xyz: np.ndarray, pixels: np.ndarray, k: np.ndarray,
             distortion: np.ndarray, threshold: float = 1.5) -> tuple:
    xyz, pixels = np.ascontiguousarray(xyz, dtype=float), np.ascontiguousarray(pixels, dtype=float)
    if len(xyz) < 12 or len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Need at least 12 distinct identified correspondences")
    if np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)[1][1] < 20:
        raise ValueError("Correspondences do not span a useful board area")
    cv2.setRNGSeed(0)
    ok, rv, tv, indices = cv2.solvePnPRansac(
        xyz, pixels, k, distortion, iterationsCount=1000,
        reprojectionError=threshold, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    # RANSAC returns its sampled consensus, which can be smaller than the set
    # supported by the refined pose. Require a majority seed, then enforce the
    # original 85 percent acceptance gate on measured, refined residuals.
    if not ok or indices is None or len(indices) < max(12, 0.5 * len(xyz)):
        raise ValueError("Insufficient RANSAC seed consensus")
    seed_count = len(indices)
    keep = np.zeros(len(xyz), dtype=bool)
    keep[indices.ravel()] = True
    for _ in range(3):
        rv, tv = cv2.solvePnPRefineLM(xyz[keep], pixels[keep], k, distortion, rv, tv)
        t = rigid_matrix(cv2.Rodrigues(rv)[0], tv)
        residual = np.linalg.norm(project(xyz, t, k, distortion) - pixels, axis=1)
        keep = residual < threshold
        if keep.sum() < max(12, 0.85 * len(xyz)):
            raise ValueError("Refined pose lost consensus")
    if np.min((xyz @ t[:3, :3].T + t[:3, 3])[:, 2]) <= 0:
        raise ValueError("Pose places board behind camera")
    return t, keep, {"ransac_seed_count": seed_count,
                     "required_final_consensus_fraction": 0.85,
                     "inlier_reprojection_px": stats(residual[keep]),
                     "all_reprojection_px": stats(residual),
                     "excluded_indices": np.flatnonzero(~keep).tolist()}


def plane_points(cloud: np.ndarray, pixels: np.ndarray, keep: np.ndarray,
                 k: np.ndarray, distortion: np.ndarray) -> tuple:
    mask = np.zeros(cloud.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pixels[keep].astype(np.int32)), 1)
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8))
    sampled = cloud[::2, ::2]
    valid = (mask[::2, ::2] > 0) & np.isfinite(sampled).all(2) & (sampled[:, :, 2] > 0)
    xyz = sampled[valid].astype(float)
    if len(xyz) < 1000:
        raise ValueError("Too little board depth")
    selected = np.ones(len(xyz), bool)
    for _ in range(7):
        centre = xyz[selected].mean(0)
        delta = xyz[selected] - centre
        _, vectors = np.linalg.eigh(delta.T @ delta / len(delta))
        normal = vectors[:, 0]
        if normal[2] < 0:
            normal *= -1
        residual = (xyz - centre) @ normal
        median = np.median(residual)
        mad = np.median(abs(residual - median))
        selected = abs(residual - median) < max(0.25, 4 * 1.4826 * mad)
    offset = normal @ centre
    rays = np.c_[cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, distortion).reshape(-1, 2),
                 np.ones(len(pixels))]
    points = rays * (offset / (rays @ normal))[:, None]
    if np.any(points[:, 2] <= 0):
        raise ValueError("Board plane intersects behind camera")
    # Verify the image/cloud coordinate convention from raw metric points.
    y, x = np.mgrid[0:cloud.shape[0]:20, 0:cloud.shape[1]:20]
    test_xyz = cloud[y, x].reshape(-1, 3).astype(float)
    test_uv = np.c_[x.ravel(), y.ravel()]
    valid = np.isfinite(test_xyz).all(1) & (test_xyz[:, 2] > 0)
    pixel_error = np.linalg.norm(project(test_xyz[valid], np.eye(4), k, distortion) - test_uv[valid], axis=1)
    if np.percentile(pixel_error, 95) > 0.1:
        raise ValueError("Cloud and texture camera coordinates disagree")
    return points, {"normal_photoneo": normal.tolist(), "offset_mm": float(offset),
                    "sampled_points": len(xyz), "retained_points": int(selected.sum()),
                    "plane_residual_mm": stats(abs(residual[selected])),
                    "cloud_pixel_consistency_px": stats(pixel_error)}


def fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple:
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    scale = float(np.sum(b * (a @ rotation.T)) / np.sum(a * a))
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rigid_matrix(rotation, translation), {"residual_mm": stats(residual),
                                                "diagnostic_scale_not_applied": scale}


def transform_difference(a: np.ndarray, b: np.ndarray) -> dict:
    angle = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2, -1, 1)))
    return {"origin_difference_mm": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            "rotation_difference_deg": float(angle)}


def run(root: Path, mapping_path: Path, output: Path) -> dict:
    mapping = json.loads(mapping_path.read_text())
    if mapping.get("mapping_reviewed") is not True:
        raise ValueError("Image-specific grid mapping must have been reviewed")
    paths = {}
    for key, record in mapping["files"].items():
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError("Input path/hash mismatch: " + key)
        paths[key] = path
    load = lambda key: json.loads(paths[key].read_text())
    ecal, meta, pose = load("eih_calibration"), load("photoneo_metadata"), load("eih_pose")
    if not pose["accepted"] or not meta["pose_association_accepted"]:
        raise ValueError("Rejected stationary capture")
    if meta["units"] != "mm" or meta["coordinate_space"] != "CameraSpace":
        raise ValueError("Expected raw millimetre camera cloud")
    focus = float(ecal["getCapturingSettings"]["Focus"]["CurrentValue"])
    size = ecal["getImageConfiguration"]
    intrinsics = [v for v in ecal["getIntrinsics"]["cam_intrinsics"] if
                  v["FocusValue"] == focus and v["ImageWidth"] == size["ImageWidth"] and
                  v["ImageHeight"] == size["ImageHeight"]]
    if len(intrinsics) != 1:
        raise ValueError("Missing unique EIH calibration for focus/resolution")
    em = intrinsics[0]["CameraMatrix"]
    ek = np.array([[em.get(f"matrix_{i}{j}", 0) for j in range(3)] for i in range(3)])
    ed = np.array([intrinsics[0]["DistortionCoefficients"].get(f"coefficient_{i}0", 0) for i in range(5)])
    current = meta["frame_info"]["CurrentCamera"]
    if not np.allclose(np.array(current["WorldToCameraCoordinates"]).reshape(3, 4), np.eye(4)[:3]):
        raise ValueError("Expected primary-camera coordinate space")
    pk = np.array(current["PerspectiveSettings"]["CameraMatrix"]).reshape(3, 3)
    pd = np.array(current["PerspectiveSettings"]["DistortionCoefficients"])
    data = {}
    for name, k, d in [("eih", ek, ed), ("photoneo", pk, pd)]:
        grid = load(name + "_grid")
        shape = grid["detected_grid_shape"]
        if shape != mapping[name]["grid_shape"]:
            raise ValueError("Unexpected grid layout")
        row, col = np.indices(shape)
        indices = np.c_[row.ravel(), col.ravel(), np.ones(row.size)]
        ids = indices @ np.array(mapping[name]["row_col_to_board_xy"]).T
        if not np.allclose(ids, np.round(ids)) or len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError("Invalid board identities")
        pixels = np.array(grid["corners_image_px"], dtype=float)
        xyz = np.c_[ids * mapping["square_pitch_mm"], np.zeros(len(ids))]
        t, keep, metrics = fit_pose(xyz, pixels, k, d)
        data[name] = {"ids": ids.astype(int), "pixels": pixels, "xyz": xyz,
                      "transform": t, "keep": keep, "metrics": metrics}
    ep, pp = data["eih"], data["photoneo"]
    cloud = np.load(paths["photoneo_cloud"])["PointCloud"]
    points, plane = plane_points(cloud, pp["pixels"], pp["keep"], pk, pd)
    depth_board, rigid_metrics = fit_rigid(pp["xyz"][pp["keep"]], points[pp["keep"]])
    index = {tuple(v): i for i, v in enumerate(pp["ids"])}
    matches = [(i, index[tuple(v)]) for i, v in enumerate(ep["ids"]) if
               tuple(v) in index and ep["keep"][i] and pp["keep"][index[tuple(v)]]]
    ei, pi = np.array(matches).T
    relative, relative_keep, relative_metrics = fit_pose(points[pi], ep["pixels"][ei], ek, ed)
    via_pnp = ep["transform"] @ np.linalg.inv(pp["transform"])
    via_depth_board = ep["transform"] @ np.linalg.inv(depth_board)
    # Same-image spatial holdouts test fit consistency; they are NOT new poses.
    spatial = []
    ids = ep["ids"][ei]
    for dimension in (0, 1):
        for side in (0, 1):
            train = (ids[:, dimension] < np.median(ids[:, dimension])) == bool(side)
            t, _, _ = fit_pose(points[pi][train], ep["pixels"][ei][train], ek, ed)
            error = np.linalg.norm(project(points[pi][~train], t, ek, ed) - ep["pixels"][ei][~train], axis=1)
            spatial.append({"split_axis": dimension, "train_lower_half": bool(side),
                            "held_out_reprojection_px": stats(error),
                            "difference_from_full_fit": transform_difference(t, relative)})
    h = ecal["getHandEyeParameters"]["HandEyeArray"]
    flange_eih = pose_matrix([h["handeye_" + v] for v in ("x", "y", "z", "rx", "ry", "rz")])
    raw = pose["feedback_trace"][0]["raw_controller_values"]["Coord_Robot_Flange"]
    base_flange = pose_matrix([float(v) for v in raw.strip("{}").split(",")])
    flange_photoneo = flange_eih @ relative
    report = {
        "schema": "workcell.provisional_camera_alignment.v1", "status": "single_pose_draft",
        "accepted_for_robot_motion": False, "applied_to_controller_or_runtime": False,
        "units": "mm", "transform_convention": "T_A_B maps column-vector points in B into A",
        "mapping_sha256": sha256(mapping_path), "script_sha256": sha256(Path(__file__)),
        "input_files": mapping["files"], "square_pitch_mm": mapping["square_pitch_mm"],
        "algorithm": {"opencv": cv2.__version__, "ransac_seed": 0,
                      "ransac_threshold_px": 1.5, "ransac_iterations": 1000,
                      "refinement": "solvePnPRefineLM; fixed factory intrinsics",
                      "depth_plane": "eroded board hull, stride 2, seven robust covariance fits",
                      "corner_3d": "undistorted corner rays intersect fitted metric board plane"},
        "board_fits": {"eih": ep["metrics"], "photoneo": pp["metrics"]},
        "depth_plane": plane, "nominal_grid_to_depth": rigid_metrics,
        "common_points": len(matches), "relative_fit": relative_metrics,
        "same_pose_spatial_holdouts": spatial,
        "comparison_two_board_pnp": transform_difference(relative, via_pnp),
        "comparison_depth_board_chain": transform_difference(relative, via_depth_board),
        "T_eih_photoneo": relative.tolist(), "T_flange_photoneo_provisional": flange_photoneo.tolist(),
        "T_base_photoneo_at_capture_provisional": (base_flange @ flange_photoneo).tolist(),
        "T_eih_photoneo_two_board_pnp_comparison": via_pnp.tolist(),
        "T_eih_photoneo_depth_board_comparison": via_depth_board.tolist(),
        "limitations": ["Only one physical board/robot pose; no independent pose validation",
                        "Spatial holdouts share images, target plane and intrinsics with training",
                        "EIH factory calibration errors propagate into arm registration",
                        "Cameras acquired sequentially; no hardware synchronisation",
                        "Detected grid IDs reviewed for these exact images; no general code decoder",
                        "Optical frame is not CAD origin, centre of mass, gripper TCP or collision model",
                        "Do not use for specimen pickup or motion until physical qualification"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    correspondence = [{"board_xy": ep["ids"][a].tolist(), "eih_uv": ep["pixels"][a].tolist(),
                       "photoneo_uv": pp["pixels"][b].tolist(), "photoneo_plane_xyz_mm": points[b].tolist(),
                       "relative_fit_inlier": bool(relative_keep[n])} for n, (a, b) in enumerate(matches)]
    (output / "correspondences.json").write_text(json.dumps(correspondence, indent=2) + "\n")
    # Scientific diagnostic: sparse identical point labels and reprojection marks.
    for name, camera, selected in [("eih", ep, ei), ("photoneo", pp, pi)]:
        image = cv2.imread(str(paths[name + "_image"]))
        for i in selected:
            x, y = camera["ids"][i]
            if x % 3 == 0 and y % 3 == 0:
                uv = tuple(np.round(camera["pixels"][i]).astype(int))
                cv2.circle(image, uv, 5 if name == "eih" else 3, (0, 255, 255), -1)
                cv2.putText(image, f"{x},{y}", (uv[0] + 5, uv[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 if name == "eih" else 0.4, (0, 80, 255), 2 if name == "eih" else 1)
        cv2.imwrite(str(output / (name + "_correspondence_diagnostic.png")), image)
    files = [p for p in output.iterdir() if p.is_file()]
    (output / "manifest_sha256.json").write_text(json.dumps({p.name: sha256(p) for p in files}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    cv2.setNumThreads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.capture_root, args.mapping, args.output)
    print(json.dumps({k: report[k] for k in ["status", "common_points", "relative_fit",
                                           "comparison_two_board_pnp", "comparison_depth_board_chain"]}, indent=2))
