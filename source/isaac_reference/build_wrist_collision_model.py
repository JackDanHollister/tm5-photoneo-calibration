#!/usr/bin/env python3
"""Derive offline URDF/XRDF geometry without changing a reference or device."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from preview_wrist_mount import PROJECT, WORKSPACE, digest, read_binary_stl, transform

DEMO = WORKSPACE / "projects/techman-jazzy-bringup/demos/pin_axis_3d_sim"
BASE_MODEL = DEMO / "generated/tool_profiles/watson_qc_fixed_workcell_cumotion"
BASE_USD = DEMO / "generated/isaac/6.0.1-watson-qc-10mm/tm5s_with_2fg7/tm5s_with_2fg7.usda"
BASE_REPORT = DEMO / "outputs/isaac_sim/6.0.1/watson_qc_10mm_import_report.json"


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-8)):
        raise ValueError("Expected a finite rigid transform in metres")
    return value


def sphere_cover(bounds, cell_size):
    """Every cell (including its interior) lies within its midpoint sphere."""
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not np.isfinite(cell_size)
            or cell_size <= 0):
        raise ValueError("Invalid box or cell size")
    counts = np.ceil((bounds[1] - bounds[0]) / cell_size).astype(int)
    if np.prod(counts) > 10000:
        raise ValueError("Sphere grid exceeds offline model budget")
    sizes = (bounds[1] - bounds[0]) / counts
    radius = float(np.linalg.norm(sizes) / 2 + 1e-9)
    centres = [bounds[0, i] + (np.arange(counts[i]) + .5) * sizes[i] for i in range(3)]
    spheres = [{"center": list(map(float, c)), "radius": radius} for c in itertools.product(*centres)]
    return spheres, {"counts": counts.tolist(), "cell_dimensions_m": sizes.tolist(),
                     "sphere_radius_m": radius, "coverage": "complete cuboid by cell circumspheres"}


def fk(urdf: Path, q):
    """Independent URDF tree evaluation, used for model/renderer checks."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected six finite joint positions")
    result, pending = {"base": np.eye(4)}, list(ET.parse(urdf).getroot().findall("joint"))
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in result]
        if not ready:
            raise ValueError("URDF is not a rooted tree")
        for joint in ready:
            o = joint.find("origin")
            xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else [0, 0, 1]
            name = joint.get("name")
            angle = q[int(name[-1]) - 1] if name in [f"joint_{i}" for i in range(1, 7)] else 0.
            result[joint.find("child").get("link")] = result[joint.find("parent").get("link")] @ transform(xyz, rpy, angle, axis)
            pending.remove(joint)
    return result


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError("Changed dependency: " + name)


def load_model(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "workcell.wrist_collision_model.v1" or value.get("physical_pick_allowed") is not False:
        raise ValueError("Expected offline wrist model")
    verify_hashes(value["source_hashes"])
    verify_hashes(value["output_hashes"])
    return value


def main():
    from scipy.spatial.transform import Rotation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--collision-config", type=Path, default=PROJECT / "config/wrist_collision.json")
    p.add_argument("--mount-config", type=Path, default=PROJECT / "config/wrist_mount.json")
    args = p.parse_args()
    registration = json.loads(args.registration.read_text())
    if registration.get("schema") != "workcell.photoneo_body_registration.v1":
        raise ValueError("Wrong registration schema")
    verify_hashes(registration["source_hashes"])
    camera = rigid(registration["T_flange_mesh_m"])
    cfg, mount = json.loads(args.collision_config.read_text()), json.loads(args.mount_config.read_text())
    if cfg.get("schema") != "workcell.wrist_collision_config.v1" or cfg.get("physical_pick_allowed") is not False:
        raise ValueError("Wrong collision configuration")
    extension = float(mount["adapter_and_cap_extension_m"])
    padding, radius = float(cfg["camera_padding_m"]), float(cfg["adapter_radius_m"])
    if not np.isfinite([extension, padding, radius]).all() or not 0 < extension < .1 or not 0 <= padding < .05 or not 0 < radius < .1:
        raise ValueError("Implausible proxy dimensions")
    mesh = WORKSPACE / mount["photoneo_visual"]["mesh_workspace_relative"]
    vertices, _ = read_binary_stl(mesh, .001)
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    if not np.allclose(bounds * 1000, registration["native_mesh_bounds_mm"], atol=1e-4):
        raise ValueError("Registration mesh bounds differ")
    components = [
        {"name": "photoneo_camera", "T_flange_link_m": camera.tolist(),
         "bounds_m": (bounds + np.array([[-padding], [padding]])).tolist(), "status": "padded nominal housing CAD"},
        {"name": "benano_adapter", "T_flange_link_m": np.eye(4).tolist(),
         "bounds_m": [[-radius, -radius, 0], [radius, radius, extension]], "status": "measured extension; assumed radial envelope"},
        {"name": "benano_support", "T_flange_link_m": np.eye(4).tolist(),
         "bounds_m": cfg["support_bounds_flange_m"], "status": cfg["support_status"]},
    ]
    source_urdf, source_xrdf = BASE_MODEL / "tm5s_with_2fg7.urdf", BASE_MODEL / "tm5s_with_2fg7.xrdf"
    tree = ET.parse(source_urdf)
    sources = {str(Path(k).resolve()): v for k, v in registration["source_hashes"].items()}
    source_paths = [args.registration, args.mount_config, args.collision_config, source_urdf, source_xrdf,
                    BASE_MODEL / "asset_manifest.json", BASE_REPORT, *BASE_USD.parent.rglob("*")]
    for path in source_paths:
        if path.is_file():
            sources[str(path.resolve())] = digest(path)
    for element in tree.findall(".//mesh"):
        filename = (source_urdf.parent / element.get("filename")).resolve()
        if not filename.is_file():
            raise ValueError("Missing source mesh: " + str(filename))
        sources[str(filename)] = digest(filename)
        element.set("filename", str(filename))
    source_report = json.loads(BASE_REPORT.read_text())
    if digest(source_urdf) != source_report["source_urdf_sha256"] or digest(BASE_USD) != source_report["output_usd_sha256"]:
        raise ValueError("Base planning and render models disagree")
    qc_origin = tree.find("./joint[@name='flange_to_onrobot_qc_robot_side']/origin")
    if not np.allclose(np.fromstring(qc_origin.get("xyz"), sep=" "), 0):
        raise ValueError("Base model already has a QC offset")
    qc_origin.set("xyz", f"0 0 {extension:.12g}")
    xrdf = yaml.safe_load(source_xrdf.read_text())
    group = xrdf["geometry"][xrdf["world_collision"]["geometry"]]["spheres"]
    for component in components:
        name, t = component["name"], rigid(component["T_flange_link_m"])
        link = ET.SubElement(tree.getroot(), "link", name=name)
        joint = ET.SubElement(tree.getroot(), "joint", name="flange_to_" + name, type="fixed")
        ET.SubElement(joint, "parent", link="flange")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, t[:3, 3])), rpy=" ".join(map(str, Rotation.from_matrix(t[:3, :3]).as_euler("xyz"))))
        lo, hi = np.array(component["bounds_m"])
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=" ".join(map(str, (lo + hi) / 2)), rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=" ".join(map(str, hi - lo)))
        if name == "photoneo_camera":
            visual = ET.SubElement(link, "visual")
            ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=str(mesh), scale="0.001 0.001 0.001")
        spheres, coverage = sphere_cover(component["bounds_m"], cfg["sphere_cell_size_m"])
        group[name] = spheres
        component.update(spheres=spheres, coverage=coverage)
    # Only intentional mounting interfaces. Camera/arm/finger checks stay active.
    exclusions = [("benano_adapter", "link_6"), ("benano_adapter", "onrobot_qc_robot_side_link"),
                  ("benano_adapter", "onrobot_2fg7_base_link"),
                  ("benano_adapter", "benano_support"), ("benano_support", "photoneo_camera"),
                  ("benano_support", "link_6"), ("benano_support", "onrobot_qc_robot_side_link"),
                  ("benano_support", "onrobot_2fg7_base_link")]
    for a, b in exclusions:
        xrdf["self_collision"].setdefault("ignore", {}).setdefault(a, []).append(b)
    xrdf["tool_frames"].append("photoneo_camera")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    urdf, xrdf_path = output / source_urdf.name, output / source_xrdf.name
    ET.indent(tree)
    tree.write(urdf, encoding="unicode")
    xrdf_path.write_text(yaml.safe_dump(xrdf, sort_keys=False))
    value = {"schema": "workcell.wrist_collision_model.v1", "physical_pick_allowed": False,
             "hardware_commands": False, "dynamics_validated": False, "config": cfg,
             "source_hashes": sources, "output_hashes": {str(p): digest(p) for p in (urdf, xrdf_path)},
             "urdf": str(urdf), "xrdf": str(xrdf_path), "source_urdf": str(source_urdf),
             "source_xrdf": str(source_xrdf), "source_usd": str(BASE_USD), "source_import_report": str(BASE_REPORT),
             "camera_mesh": str(mesh), "components": components, "adapter_extension_m": extension,
             "added_self_collision_exclusions": [{"pair": [a, b], "reason": "fixed mounting interface; proxy overlap"} for a, b in exclusions],
             "snapshot_pose_correction_applied": False,
             "limitations": ["Draft hand-eye estimate; 5 mm padding is not an accuracy bound",
                             "New spheres enclose their proxy cuboids; physical support enclosure is unverified",
                             "Inherited robot/gripper spheres are sample-audited, not certified enclosure",
                             "Planner fingers are fixed open; renderer finger animation is not full jaw collision qualification",
                             cfg["cable_routing_status"], "No mass, inertia, load or physical contact-response validation"]}
    verify_hashes(sources)
    (output / "model.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"model": str(output / "model.json"), "added_spheres": {c["name"]: len(c["spheres"]) for c in components}}))


if __name__ == "__main__":
    main()
