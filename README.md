# Photoneo camera calibration and Isaac frame registration

This focused code package accompanies the installed-camera screenshot. The
transform was estimated using our own Python workflow, not the manufacturer's
robot-calibration tool. No SDK/controller binaries or specimen images are included.

The repository is private. The owner must grant your GitHub account access.
[Draft reply to the two software/code questions](DEVELOPER_REPLY.md).

## Method

The initial camera-to-camera fit used the official TM printed board seen by both
the robot EIH camera and Photoneo, with the existing EIH factory intrinsics/hand-eye.
The later direct sensor-to-flange fit used measured Photoneo board points and
recorded robot flange poses: seven usable tilt views plus eight translation views.
OpenCV PARK/HORAUD initial estimates were refined using SciPy metric 3D least
squares with a0.3mm soft-L1 loss. Separate fixed board poses were fitted for the
translation and tilt batches; the board did not have to remain in the same pose
between sessions. The camera mount was assumed unchanged.

The direct numerical fit uses Photoneo/robot observations. EIH observations helped
establish printed-point identities; EIH hand-eye was not a numerical pose prior
in the final direct fit.

For homogeneous column vectors in millimetres:

```
p_base = T_base_flange @ T_flange_photoneo @ p_primary_camera
```

`T_flange_photoneo` maps the saved **CameraSpace / PrimaryCamera optical frame**
into the robot flange frame. It is not a transform to the housing bounding-box
centre, mounting-hole centre or centre of mass. The origin is approximately
(-86.90, -97.60,80.58)mm in flange coordinates; full rotation is retained.

## Reproduce the saved fit

Tested using Python on Linux. Original modules use the Unix `resource` library.
Requirements retain the OpenCV4 Python API used by the original fitting code.
The OpenCV5.0.0.93 wheel selected by the first CI installation lacked
`cv2.calibrateHandEye`; see the [upstream issue](https://github.com/opencv/opencv/issues/29565).

```bash
python3 -m pip install -r requirements.txt
python3 replay_saved_fit.py
python3 -m unittest discover -s source/calibration -p 'test_tilt_calibration.py'
```

The replay uses the supplied numerical correspondences, computes an initial
tilt fit and runs the original two-batch joint fit. It compares its result with
the saved estimate and writes `replayed_fit.json`. It imports no controller or
camera acquisition modules and blocks network connections during fitting.

`source/calibration/` preserves the original fitting, corner/plane processing,
relative-camera alignment and sensitivity code byte-for-byte. The replay runs
without raw acquisition files. Original image-processing command-line entrypoints
require their original dataset layout; those raw images/point clouds are not in
this focused package. `data/provenance.json` describes the numerical export.

## Isaac/CAD relationship

The housing shape is the manufacturer's MotionCam-3D Color S CAD mesh. Its
native coordinates were retained. The nominal relationship between the vendor
sensor mesh and PrimaryCamera frame was checked against the vendor ROS
description, saved frame metadata and optical rays/front windows. CAD was not
recentred on its bounding-box centre.

The full calibrated flange-relative rotation places the camera at its actual
unusual mounting angle, approximately22.563° between optical +Z and flange +Z.
The later moving Isaac model attaches it to the wrist without the early static
preview's snapshot-only flange correction. URDF/XRDF/USD update the attached
camera consistently; QC/gripper stack extension is23.15mm.

`source/isaac_reference/` contains original CAD/frame-registration and moving
wrist-model scripts as reference. They retain workspace/Isaac/vendor-asset
dependencies and are not standalone launchers. Manufacturer mesh provenance and
source URLs are in `results/photoneo_body_registration.json`; CAD/robot assets
and Isaac binaries are not redistributed in this focused package.

## Results and limitations

The fifteen-view fit gives0.3321338mm RMS over2,104 points. The largest measured
leave-one-view-out mapped-point shift was2.390mm RMS. These are consistency and
sensitivity diagnostics, not absolute-accuracy guarantees. Y rotation diversity
was limited and the original combined-angle validation captures were incomplete.
Board pitch was nominal20mm; scanner scale, robot absolute calibration and exact
mount stability were not independently qualified.

The saved result is a development estimate and was not issued by the manufacturer
calibration tool or accepted for precision pickup. Projecting bracket geometry is
approximate. Original result JSON retains diagnostic source paths as provenance;
the supported saved-fit replay uses package-relative numerical input paths.

`source-identities.json` records original source hashes. `SHA256SUMS` verifies all
files in the bundle.
