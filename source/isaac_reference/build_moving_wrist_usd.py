#!/usr/bin/env python3
"""Author a separate moving robot USD layer with matching wrist geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from build_wrist_collision_model import load_model, digest, fk
from preview_wrist_mount import read_binary_stl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    model = load_model(args.model)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    usd = output / "tm5s_photoneo.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    source = json.loads(Path(model["source_import_report"]).read_text())
    paths = source["expected_link_paths"].copy()
    root = paths["base"].split("/Geometry/")[0]
    robot = UsdGeom.Xform.Define(stage, root).GetPrim()
    robot.GetReferences().AddReference(model["source_usd"])
    robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    stage.SetDefaultPrim(robot)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.GetRootLayer().customLayerData = {"scope": "Offline kinematic wrist model; assumed bracket; unknown cables", "hardwareCommands": False}
    flange = paths["flange"]
    extension = model["adapter_extension_m"]
    qc = stage.GetPrimAtPath(paths["onrobot_qc_robot_side_link"])
    qc.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, extension))
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root + "/Physics/flange_to_onrobot_qc_robot_side"))
    if not fixed or fixed.GetBody0Rel().GetTargets() != [Sdf.Path(flange)]:
        raise ValueError("Unexpected QC fixed-joint parent")
    old_anchor = np.array(fixed.GetLocalPos0Attr().Get())
    if not np.allclose(old_anchor, 0, atol=1e-8):
        raise ValueError("QC anchor already shifted")
    fixed.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, extension))
    vertices, faces = read_binary_stl(Path(model["camera_mesh"]), .001)
    collider_paths = []
    for comp in model["components"]:
        path = flange + "/" + comp["name"]
        paths[comp["name"]] = path
        item = UsdGeom.Xform.Define(stage, path)
        item.AddTransformOp().Set(Gf.Matrix4d(np.array(comp["T_flange_link_m"]).T.tolist()))
        item.GetPrim().CreateAttribute("workcell:geometryStatus", Sdf.ValueTypeNames.String).Set(comp["status"])
        if comp["name"] == "photoneo_camera":
            mesh = UsdGeom.Mesh.Define(stage, path + "/Housing")
            mesh.CreatePointsAttr(vertices.tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.16, .19, .22)])
        lo, hi = np.array(comp["bounds_m"])
        cube = UsdGeom.Cube.Define(stage, path + "/CollisionEnvelope")
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        cube.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
        cube.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        collider_paths.append(str(cube.GetPath()))
        if comp["name"] != "photoneo_camera":
            # Clearly approximate support; same bounds as collision proxy.
            visible = UsdGeom.Cube.Define(stage, path + "/AssumedEnvelope")
            visible.CreateSizeAttr(1.)
            visible.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
            visible.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
            visible.CreateDisplayColorAttr([Gf.Vec3f(.27, .29, .32)])
    # Set the initial visible hierarchy from the derived URDF, including all
    # fixed children; PhysX will own it after articulation initialization.
    import xml.etree.ElementTree as ET
    for joint in ET.parse(model["urdf"]).getroot().findall("joint"):
        child = joint.find("child").get("link")
        if child not in paths: raise ValueError("Missing USD link " + child)
    zero = fk(Path(model["urdf"]), np.zeros(6))
    # Source asset is at zero joint positions. New local transforms were authored
    # above; check all composed zero-pose transforms before loading PhysX.
    initial_error = 0.
    for name, path in paths.items():
        actual = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        initial_error = max(initial_error, float(np.max(abs(actual - zero[name]))))
    if initial_error > 1e-6: raise ValueError(f"USD versus derived URDF mismatch: {initial_error}")
    stage.GetRootLayer().Save()
    # Audit the composed dependency graph, including referenced payloads/meshes.
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
    if unresolved: raise ValueError("Unresolved USD dependencies: " + str(unresolved))
    dependencies = {str(Path(layer.realPath).resolve()): digest(Path(layer.realPath)) for layer in layers if layer.realPath}
    for asset in assets:
        path = Path(asset).resolve()
        if not path.is_file(): raise ValueError("Missing USD dependency " + str(path))
        dependencies[str(path)] = digest(path)
    dependencies.update(model["source_hashes"])
    dependencies.update(model["output_hashes"])
    dependencies[str(args.model.resolve())] = digest(args.model)
    report = {"schema": "workcell.derived_isaac_asset.v1", "status": "authored_requires_runtime_validation",
              "output_usd_sha256": digest(usd), "source_urdf_sha256": digest(Path(model["urdf"])),
              "xrdf_sha256": digest(Path(model["xrdf"])), "expected_link_paths": paths,
              "dependency_sha256": dependencies, "physical_pick_allowed": False, "hardware_commands": False,
              "initial_urdf_usd_max_matrix_error": initial_error, "collider_paths": collider_paths,
              "physics_variant": "physx", "dynamics_validated": False,
              "wrist_attachment": {"camera_path": paths["photoneo_camera"], "flange_path": flange,
                  "planner_urdf": model["urdf"],
                  "T_flange_camera_m": model["components"][0]["T_flange_link_m"],
                  "T_flange_tcp_m": (np.linalg.inv(zero["flange"]) @ zero["pin_grasp_tcp"]).tolist(),
                  "model_json": str(args.model.resolve()), "snapshot_pose_correction_applied": False}}
    (output / "derived_asset.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"usd": str(usd), "dependency_count": len(dependencies), "initial_error": initial_error}))


if __name__ == "__main__": main()
