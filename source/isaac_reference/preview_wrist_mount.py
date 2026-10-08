#!/usr/bin/env python3
"""Build and render a static Isaac wrist-assembly draft from saved robot feedback.

Uses the existing TM5S/QC/2FG7 visual asset with its Physics variant disabled.
Writes a separate USD layer. No ROS or hardware-control interfaces are imported.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
REFERENCE = WORKSPACE / "projects/techman-jazzy-bringup/demos/pin_axis_3d_sim/reference/seven_pin"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_binary_stl(path: Path, scale: float):
    """Read the existing vendor visual mesh without importing its ROS runtime."""
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little")
    if len(data) != 84 + 50 * count:
        raise ValueError("Expected a complete binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"]
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    vertices = vertices.astype(float) * scale
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Non-finite camera mesh")
    return vertices, indices.reshape(-1, 3)


def transform(xyz, rpy, angle=0.0, axis=(0, 0, 1)):
    """URDF column-vector transform, including a revolute joint's rotation."""
    x, y, z = rpy
    cx, cy, cz, sx, sy, sz = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    rotation = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rotation = rotation @ (np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross))
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, xyz
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/wrist_mount.json")
    parser.add_argument("--snapshot", type=Path, required=True, help="Saved final_state.json; never reads the robot.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory outside source Git.")
    parser.add_argument("--interactive", action="store_true", help="Keep the static scene open after rendering.")
    parser.add_argument("--body-registration", type=Path,
                        help="Offline optical-to-housing audit JSON; preserves the vendor optical origin and full rotation.")
    parser.add_argument("--scan-registration", type=Path,
                        help="Saved registered point/surface reconstruction; requires matching capture snapshot and body registration.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    snapshot = json.loads(args.snapshot.read_text())["robot_snapshot"]
    positions = np.asarray(snapshot["feedback"]["joint_pos"], dtype=float)
    if positions.shape != (6,) or not np.all(np.isfinite(positions)):
        raise ValueError("Expected six finite saved joint positions")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    asset = REFERENCE / "isaac/tm5s_with_2fg7/tm5s_with_2fg7.usda"
    urdf = REFERENCE / "source/tm5s_with_2fg7.urdf"
    source_files = [urdf, *sorted(asset.parent.rglob("*.usd*"))]
    camera_config = config.get("photoneo_visual")
    camera_vertices = camera_faces = None
    registration = None
    if "photoneo_camera" in config["visual_components"]:
        mesh_path = WORKSPACE / camera_config["mesh_workspace_relative"]
        source_files.append(mesh_path)
        camera_vertices, camera_faces = read_binary_stl(mesh_path, camera_config["mesh_units_to_m"])
        if not np.allclose(np.ptp(camera_vertices, axis=0), camera_config["mesh_expected_extent_m"], atol=0.0002):
            raise ValueError("Camera mesh dimensions do not match the recorded vendor asset")
    if args.body_registration:
        registration = json.loads(args.body_registration.read_text())
        if registration["schema"] != "workcell.photoneo_body_registration.v1" or camera_vertices is None:
            raise ValueError("Body registration needs the expected audit and vendor mesh")
        for path, expected_hash in registration["source_hashes"].items():
            if digest(Path(path)) != expected_hash:
                raise ValueError("Body registration source changed: " + path)
            source_files.append(Path(path))
        source_files.append(args.body_registration)
        calibrated_camera = np.array(registration["T_flange_mesh_m"])
        if (calibrated_camera.shape != (4, 4) or not np.isfinite(calibrated_camera).all()
                or not np.allclose(calibrated_camera[3], [0, 0, 0, 1])
                or not np.allclose(calibrated_camera[:3, :3].T @ calibrated_camera[:3, :3], np.eye(3))
                or not np.isclose(np.linalg.det(calibrated_camera[:3, :3]), 1)):
            raise ValueError("Invalid registered camera transform")
    scan = scan_data = None
    if args.scan_registration:
        scan = json.loads(args.scan_registration.read_text())
        if (not registration or scan.get("schema") != "workcell.registered_scan.v1"
                or scan.get("status") != "passed_visual_reconstruction"
                or scan.get("coordinate_frame") != "robot_base" or scan.get("units") != "m"
                or scan.get("collision_geometry") is not False
                or scan.get("physical_pick_allowed") is not False
                or scan["snapshot_sha256"] != digest(args.snapshot)
                or scan["body_registration_sha256"] != digest(args.body_registration)):
            raise ValueError("Scan and displayed camera/capture pose do not match")
        for group in (scan["source_hashes"], scan["output_hashes"]):
            for path, expected in group.items():
                if digest(Path(path)) != expected:
                    raise ValueError("Scan dependency changed: " + path)
                source_files.append(Path(path))
        source_files.append(args.scan_registration)
        with np.load(scan["geometry_npz"], allow_pickle=False) as data:
            scan_data = {key: data[key].copy() for key in data.files}
        for key in ("points_base_m", "mesh_points_base_m"):
            if scan_data[key].ndim != 2 or scan_data[key].shape[1] != 3 or not np.isfinite(scan_data[key]).all():
                raise ValueError("Invalid registered scan points")
    source_hashes = {str(p): digest(p) for p in source_files}

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": not args.interactive, "hide_ui": not args.interactive,
        "width": 1440, "height": 1080, "window_width": 1440, "window_height": 1080,
        "renderer": "RaytracedLighting", "active_gpu": 0, "multi_gpu": False,
        "max_gpu_count": 1, "fast_shutdown": True,
    })
    try:
        import carb
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        from isaacsim.core.rendering_manager import ViewportManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        robot = UsdGeom.Xform.Define(stage, "/Watson").GetPrim()
        robot.GetReferences().AddReference(str(asset))
        robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
        stage.SetDefaultPrim(robot)
        stage.GetRootLayer().customLayerData = {"scope": config["scope"], "hardwareCommands": False}
        carb.settings.get_settings().set_bool("/persistent/app/hydra/displayPurpose/guide", False)
        omni.timeline.get_timeline_interface().stop()
        # Some source guide primitives carry collision APIs outside the Physics
        # variant. Remove these only in this new, stronger visual-draft layer.
        for prim in stage.Traverse():
            for api in (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI,
                        UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI,
                        UsdPhysics.MeshCollisionAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)

        def local_matrix(prim, matrix):
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))

        links = {}
        link_names = {link.attrib["name"] for link in ET.parse(urdf).getroot().findall("link")}
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/Watson/Geometry/") and prim.GetName() in link_names:
                if prim.GetName() in links:
                    raise ValueError(f"Ambiguous visual link: {prim.GetName()}")
                links[prim.GetName()] = prim
        joints = ET.parse(urdf).getroot().findall("joint")
        predicted = {"base": np.eye(4)}
        transforms = {}
        extension = float(config["adapter_and_cap_extension_m"])
        for joint in joints:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            angle = 0.0
            if joint.attrib["name"] in [f"joint_{i}" for i in range(1, 7)]:
                angle = positions[int(joint.attrib["name"].split("_")[-1]) - 1]
            if joint.attrib["name"] == "flange_to_onrobot_qc_robot_side":
                xyz[2] += extension
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.get("xyz"), sep=" ") if axis_node is not None else [0, 0, 1]
            matrix = transform(xyz, rpy, angle, axis)
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            transforms[child] = (parent, matrix)
            if child not in links:
                raise ValueError(f"Missing visual link: {child}")
            local_matrix(links[child], matrix)
        pending = dict(transforms)
        while pending:
            ready = [child for child, (parent, _) in pending.items() if parent in predicted]
            if not ready:
                raise ValueError("URDF topology is not a rooted tree")
            for child in ready:
                parent, matrix = pending.pop(child)
                predicted[child] = predicted[parent] @ matrix

        flange = links["flange"]
        saved_flange_matrix = transform(snapshot["feedback"]["tool0_pose"][:3], snapshot["feedback"]["tool0_pose"][3:])
        model_to_measured_flange = np.linalg.inv(predicted["flange"]) @ saved_flange_matrix
        cosine = (np.trace(predicted["flange"][:3, :3].T @ saved_flange_matrix[:3, :3]) - 1) / 2
        flange_rotation_error = float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
        measured_flange_path = str(flange.GetPath()) + "/MeasuredFlangePose"
        if registration:
            # The nominal URDF chain and physical controller pose differ slightly.
            # Preserve both: this snapshot-only child locates the optical model at
            # the recorded flange pose, without altering arm links or calibration.
            measured_flange = UsdGeom.Xform.Define(stage, measured_flange_path).GetPrim()
            local_matrix(measured_flange, model_to_measured_flange)
            measured_flange.CreateAttribute("workcell:snapshotOnly", Sdf.ValueTypeNames.Bool).Set(True)
        mount_path = str(flange.GetPath()) + "/BenanoAdapterAndCap"
        mount = UsdGeom.Xform.Define(stage, mount_path).GetPrim()
        mount.CreateAttribute("workcell:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
        mount.CreateAttribute("workcell:geometryStatus", Sdf.ValueTypeNames.String).Set(config["proxy_dimensions"]["status"])
        d = config["proxy_dimensions"]

        def colour(schema, rgb):
            schema.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])

        def ring(name, outer, inner, z_min, z_max, rgb):
            points = []
            segments = 72
            for z in (z_min, z_max):
                for radius in (outer, inner):
                    points.extend((radius * math.cos(i * 2 * math.pi / segments), radius * math.sin(i * 2 * math.pi / segments), z) for i in range(segments))
            faces = []
            for i in range(segments):
                j = (i + 1) % segments
                faces.extend([[i, j, j + 2 * segments, i + 2 * segments],
                              [i + segments, i + 3 * segments, j + 3 * segments, j + segments],
                              [i, i + segments, j + segments, j],
                              [i + 2 * segments, j + 2 * segments, j + 3 * segments, i + 3 * segments]])
            geom = UsdGeom.Mesh.Define(stage, mount_path + "/" + name)
            geom.CreatePointsAttr(points)
            geom.CreateFaceVertexCountsAttr([4] * len(faces))
            geom.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
            geom.CreateSubdivisionSchemeAttr("none")
            colour(geom, rgb)

        # The two individual thicknesses are not known. Show one combined
        # adapter/cap envelope rather than retaining the earlier assumed seam.
        ring("AdapterAndCapEnvelope", d["adapter_radius_m"], d["centre_hole_radius_m"],
             0, extension, (0.60, 0.65, 0.70))

        camera_report = None
        if camera_vertices is not None:
            support_path = (measured_flange_path if registration else mount_path) + "/CameraSupport"
            support = UsdGeom.Xform.Define(stage, support_path).GetPrim()
            # In the overhead view: EIH +Y is 12; flange -X is 3.
            support_matrix = transform([0, 0, 0], [0, 0, math.pi / 2])
            local_matrix(support, support_matrix)
            support.CreateAttribute("workcell:placementStatus", Sdf.ValueTypeNames.String).Set(camera_config["placement_status"])

            def block(name, centre, size):
                box = UsdGeom.Cube.Define(stage, support_path + "/" + name)
                box.CreateSizeAttr(1)
                box.AddTranslateOp().Set(Gf.Vec3d(*centre))
                box.AddScaleOp().Set(Gf.Vec3f(*size))
                colour(box, (0.24, 0.28, 0.32))

            bounds = np.array([camera_vertices.min(axis=0), camera_vertices.max(axis=0)])
            centre = bounds.mean(axis=0)
            extent = bounds[1] - bounds[0]
            radius = float(camera_config["body_bbox_centre_radius_m"])
            body_z = float(camera_config["body_bbox_centre_flange_z_m"])
            thickness = d["plate_thickness_m"]
            plate_y = radius - extent[1] / 2 - thickness / 2
            start_y, support_z = 0.020, 0.010
            for side, x in (("Left", -d["bracket_width_m"] / 2 + thickness / 2),
                            ("Right", d["bracket_width_m"] / 2 - thickness / 2)):
                if registration is None:
                    block("Rail" + side, [x, (start_y + plate_y) / 2, support_z],
                          [thickness, plate_y - start_y, thickness])
            if registration is None:
                block("CameraPlate", [0, plate_y, body_z],
                      [d["bracket_width_m"], thickness, d["bracket_plate_height_m"]])
            body = UsdGeom.Xform.Define(stage, support_path + "/PhotoneoBodyVisual").GetPrim()
            body_matrix = transform([0, radius, body_z], np.radians(camera_config["body_rotation_in_bracket_deg"]))
            if registration is not None:
                body_matrix = np.linalg.inv(support_matrix) @ calibrated_camera
                support.GetAttribute("workcell:placementStatus").Set(
                    "Camera follows estimated optical calibration; projecting bracket geometry unresolved and omitted")
            local_matrix(body, body_matrix)
            mesh = UsdGeom.Mesh.Define(stage, str(body.GetPath()) + "/VendorHousing")
            mesh.CreatePointsAttr((camera_vertices if registration else camera_vertices - centre).tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(camera_faces))
            mesh.CreateFaceVertexIndicesAttr(camera_faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            colour(mesh, (0.12, 0.15, 0.18))
            material = UsdShade.Material.Define(stage, "/Looks/PhotoneoGraphite")
            shader = UsdShade.Shader.Define(stage, "/Looks/PhotoneoGraphite/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.055, 0.065, 0.080))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.25)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            body_to_flange = support_matrix @ body_matrix
            body_centre_flange = (body_to_flange @ np.r_[centre, 1])[:3] if registration else body_to_flange[:3, 3]
            camera_report = {
                "source_extent_m": extent.tolist(), "source_bbox_centre_m": centre.tolist(),
                "body_bbox_centre_flange_m": body_centre_flange.tolist(),
                "body_rotation_flange": body_to_flange[:3, :3].tolist(),
                "placement_status": registration["status"] if registration else camera_config["placement_status"],
                "body_prim": str(body.GetPath()), "triangle_count": len(camera_faces),
            }
            if registration:
                camera_report.update({"registration": registration, "T_flange_mesh_m": body_to_flange.tolist(),
                                      "native_mesh_coordinates_preserved": True,
                                      "projecting_bracket_geometry": "omitted; optical calibration does not recover it"})

                def line(name, points, rgb, width=.0013):
                    curve = UsdGeom.BasisCurves.Define(stage, str(body.GetPath()) + "/" + name)
                    curve.CreateTypeAttr("linear")
                    curve.CreateCurveVertexCountsAttr([len(points)])
                    curve.CreatePointsAttr(points)
                    curve.CreateWidthsAttr([width])
                    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                    colour(curve, rgb)

                for axis, rgb in enumerate([(1., .15, .12), (.15, .9, .25), (.15, .55, 1.)]):
                    end = np.zeros(3); end[axis] = .045
                    line("PrimaryAxis" + "XYZ"[axis], [[0., 0., 0.], end.tolist()], rgb, .002)
                line("PrimaryViewingDirection", [[0., 0., .045], [0., 0., .15]], (.15, .55, 1.))
                colour_camera = np.array(registration["T_primary_colour_camera_mm"])
                start = colour_camera[:3, 3] / 1000
                end = start + .09 * colour_camera[:3, 2]
                line("ColourViewingDirection", [start.tolist(), end.tolist()], (1., .55, .12))

        scan_points = scan_surface = None
        if scan:
            expected_camera = saved_flange_matrix @ calibrated_camera
            if (not np.allclose(saved_flange_matrix, scan["T_base_flange_m"], atol=1e-10)
                    or not np.allclose(expected_camera, scan["T_base_camera_m"], atol=1e-10)):
                raise ValueError("Scan transform differs from displayed capture camera")
            # Base-frame points are already registered. Identity world parent is
            # deliberate; never put them beneath the wrist/camera transform.
            root = UsdGeom.Xform.Define(stage, "/CapturedScan")
            root.GetPrim().CreateAttribute("workcell:geometryStatus", Sdf.ValueTypeNames.String).Set("Measured visible surfaces; estimated registration; visual only")
            scan_points = UsdGeom.Points.Define(stage, "/CapturedScan/MeasuredPoints")
            scan_points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(scan_data["points_base_m"]))
            scan_points.CreateWidthsAttr([.00085])
            scan_points.SetWidthsInterpolation("constant")
            scan_points.CreateDisplayColorPrimvar("vertex").Set(Vt.Vec3fArray.FromNumpy(scan_data["colors"].astype(np.float32)/255.))
            scan_surface = UsdGeom.Mesh.Define(stage, "/CapturedScan/ObservedSurface")
            scan_surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(scan_data["mesh_points_base_m"]))
            faces = scan_data["mesh_triangles"]
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(scan_data["mesh_points_base_m"]):
                raise ValueError("Invalid measured surface triangles")
            scan_surface.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
            scan_surface.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
            scan_surface.CreateSubdivisionSchemeAttr("none")
            scan_surface.CreateDoubleSidedAttr(True)
            scan_surface.CreateDisplayColorPrimvar("vertex").Set(Vt.Vec3fArray.FromNumpy(scan_data["mesh_colors"].astype(np.float32)/255.))
            scan_points.CreateVisibilityAttr("invisible")
            scan_surface.CreateVisibilityAttr("inherited")
            # Non-emissive measured greyscale, read as a vertex primvar.
            material = UsdShade.Material.Define(stage, "/CapturedScan/LaserAppearance")
            shader = UsdShade.Shader.Define(stage, "/CapturedScan/LaserAppearance/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            reader = UsdShade.Shader.Define(stage, "/CapturedScan/LaserAppearance/Intensity")
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.9)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for geom in (scan_points, scan_surface):
                UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
            identity = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            if not np.array_equal(identity, np.eye(4)):
                raise RuntimeError("Registered scan would be transformed twice")

        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"Static preview unexpectedly contains physics: {prim.GetPath()}")
        cache = UsdGeom.XformCache()
        errors = {}
        for name, expected in predicted.items():
            actual = np.asarray(cache.GetLocalToWorldTransform(links[name])).T
            errors[name] = float(np.max(np.abs(actual - expected)))
        if max(errors.values()) > 1e-7:
            raise RuntimeError(f"USD hierarchy disagrees with URDF transforms: {errors}")
        saved_flange = np.asarray(snapshot["feedback"]["tool0_pose"][:3])
        flange_error = float(np.linalg.norm(predicted["flange"][:3, 3] - saved_flange))
        if flange_error > 0.003:
            raise RuntimeError(f"Saved robot flange and model disagree by {flange_error} m")
        flange_to_qc = np.linalg.inv(predicted["flange"]) @ predicted["onrobot_qc_robot_side_link"]
        if not np.allclose(flange_to_qc[:3, 3], [0, 0, extension], atol=1e-10):
            raise RuntimeError("QC stack extension is wrong")
        if camera_report is not None:
            camera_reference = saved_flange_matrix if registration else predicted["flange"]
            actual = np.linalg.inv(camera_reference) @ np.asarray(cache.GetLocalToWorldTransform(body)).T
            if not np.allclose(actual, support_matrix @ body_matrix, atol=1e-10):
                raise RuntimeError("Camera visual is not attached at the configured flange transform")
            direction = body_centre_flange[:2] / np.linalg.norm(body_centre_flange[:2])
            if registration:
                if not np.allclose(actual, calibrated_camera, atol=1e-10):
                    raise RuntimeError("USD camera transform differs from optical calibration")
                if not np.allclose(body_centre_flange * 1000, registration["body_bbox_centre_flange_mm"], atol=1e-7):
                    raise RuntimeError("Native mesh origin was not preserved")
                if direction[0] >= 0:
                    raise RuntimeError("Estimated housing conflicts with reported mounting side")
            elif not np.allclose(direction, config["camera_clocking"]["visual_flange_direction"][:2], atol=1e-10):
                raise RuntimeError("Camera visual is on the wrong clock side")

        dome = UsdLux.DomeLight.Define(stage, "/Lighting/Dome")
        dome.CreateIntensityAttr(1000)
        key = UsdLux.DistantLight.Define(stage, "/Lighting/Key")
        key.CreateIntensityAttr(1800)
        key.AddRotateXYZOp().Set(Gf.Vec3f(20, -35, -25))
        floor = UsdGeom.Cube.Define(stage, "/Floor")
        floor.CreateSizeAttr(1)
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.015))
        floor.AddScaleOp().Set(Gf.Vec3f(2, 2, 0.02))
        colour(floor, (0.10, 0.13, 0.17))
        ready, _ = ViewportManager.wait_for_viewport(max_frames=120)
        if not ready:
            raise RuntimeError("Isaac viewport did not become ready")
        f = predicted["flange"]
        screenshots = []
        views = [
            ("wrist-detail", [-0.48, -0.42, 0.23], [-0.060, 0, 0.045]),
            ("wrist-opposite", [-0.43, 0.46, 0.21], [-0.060, 0, 0.045]),
            ("wrist-clock-top", [-0.040, 0, -0.83], [-0.040, 0, 0]),
            ("whole-arm", None, None),
        ]
        if registration:
            views.append(("camera-face", None, None))
        if scan:
            views = [(label, None, None) for label in ("scan-overview", "scan-surface", "scan-points", "scan-top")]
        for label, eye_local, target_local in views:
            if label.startswith("scan-"):
                focus = np.array(scan["board_interior"]["centre_base_m"])
                surface_visible = label != "scan-points"
                scan_points.GetVisibilityAttr().Set("invisible" if surface_visible else "inherited")
                scan_surface.GetVisibilityAttr().Set("inherited" if surface_visible else "invisible")
                # Keep the wrist visible in the overview; hide only the robot in
                # scan detail views so its housing does not occlude the board.
                UsdGeom.Imageable(robot).GetVisibilityAttr().Set("inherited" if label == "scan-overview" else "invisible")
                if label == "scan-overview":
                    eye, target = [1.15, -.95, 1.0], [float(focus[0])*.65, float(focus[1]), .30]
                else:
                    target = focus.tolist()
                    eye = (focus + ([0., 0., .65] if label == "scan-top" else [.18, -.40, .40])).tolist()
            elif label == "camera-face":
                camera_world = saved_flange_matrix @ calibrated_camera
                eye = (camera_world @ np.r_[centre + [0., -.12, .55], 1])[:3].tolist()
                target = (camera_world @ np.r_[centre, 1])[:3].tolist()
            elif eye_local is None:
                eye, target = [1.28, -1.15, 1.05], [0.15, -0.05, 0.4]
            else:
                eye = (f @ np.r_[eye_local, 1])[:3].tolist()
                target = (f @ np.r_[target_local, 1])[:3].tolist()
            ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=eye, target=target)
            if label in ("wrist-clock-top", "camera-face"):
                view_camera = stage.GetPrimAtPath(str(get_active_viewport().camera_path))
                up = (camera_world if label == "camera-face" else f)[:3, :3] @ np.array([0.0, 1.0, 0.0])
                world_matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
                local_matrix(view_camera, np.asarray(world_matrix).T)
            for _ in range(90):
                app.update()
            path = output / (label + ".png")
            capture = capture_viewport_to_file(get_active_viewport(), file_path=str(path))
            future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
            capture_deadline = time.monotonic() + 12.0
            while not future.done() and time.monotonic() < capture_deadline:
                app.update()
                time.sleep(.01)
            if not future.done() or not future.result() or not path.exists():
                raise RuntimeError(f"Capture failed: {label}; future_done={future.done()}, file_exists={path.exists()}")
            screenshots.append({"path": str(path), "sha256": digest(path)})
        # Save the useful close view as the default when opening the USD.
        if scan:
            UsdGeom.Imageable(robot).GetVisibilityAttr().Set("inherited")
            scan_points.GetVisibilityAttr().Set("invisible")
            scan_surface.GetVisibilityAttr().Set("inherited")
            eye, target = [1.15, -.95, 1.0], [float(focus[0])*.65, float(focus[1]), .30]
        else:
            eye = (f @ np.r_[-0.48, -0.42, 0.23, 1])[:3].tolist()
            target = (f @ np.r_[-0.060, 0, 0.045, 1])[:3].tolist()
        ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=eye, target=target)
        scene_path = output / "watson_benano_mount.usda"
        stage.GetRootLayer().Export(str(scene_path))
        if any(digest(Path(p)) != value for p, value in source_hashes.items()):
            raise RuntimeError("A source asset changed during preview")
        report = {
            "status": "passed_visual_draft", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": config["scope"], "hardware_commands": False, "physics_variant": "none",
            "configuration": config, "snapshot_utc": snapshot["received_utc"],
            "snapshot_sha256": digest(args.snapshot), "config_sha256": digest(args.config),
            "source_hashes": source_hashes, "source_assets_unchanged": True,
            "scene": str(scene_path), "scene_sha256": digest(scene_path), "screenshots": screenshots,
            "max_usd_urdf_transform_error": max(errors.values()),
            "model_vs_saved_controller_flange_position_error_m": flange_error,
            "model_vs_saved_controller_flange_rotation_error_deg": flange_rotation_error,
            "flange_to_qc_translation_m": flange_to_qc[:3, 3].tolist(),
            "flange_to_2fg7_origin_m": (np.linalg.inv(f) @ predicted["onrobot_2fg7_origin"])[:3, 3].tolist(),
            "installed_camera_pose_verified": False, "mount_mass_verified": False,
            "photoneo_visual": camera_report,
            "body_registration_sha256": digest(args.body_registration) if args.body_registration else None,
            "snapshot_only_model_to_measured_flange_m": model_to_measured_flange.tolist() if registration else None,
            "camera_pose_reference": "recorded physical flange pose" if registration else "nominal URDF flange pose",
            "mount_mass_evidence": config.get("mount_mass_evidence"),
            "registered_scan": {"manifest_sha256": digest(args.scan_registration), "counts": scan["counts"],
                "scan_world_transform": "identity", "display_point_diameter_m": .00085,
                "collision_geometry": False, "capture_camera_pose_match": True} if scan else None,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
        if args.interactive:
            import omni.ui as ui
            panel = ui.Window("Wrist assembly draft", width=430, height=145)
            with panel.frame:
                with ui.VStack():
                    ui.Label("TM5S / Benano mount / Photoneo / Quick Changer / 2FG7")
                    ui.Label("Static saved pose. Robot disconnected.")
                    ui.Label("Adapter + cap: 23.15 mm combined visual envelope.")
                    ui.Label("Visual model only. No collision or load qualification.")
            while app.is_running():
                app.update()
        return 0
    except BaseException:
        failure = traceback.format_exc()
        (output / "failure.txt").write_text(failure)
        print(failure, flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
