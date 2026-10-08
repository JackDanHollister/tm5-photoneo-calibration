# Photoneo camera calibration and Isaac frame registration

I've been working on mounting a Photoneo camera to my TM5 arm and getting its
position into Isaac. I've put the calibration code, saved results and numerical
inputs here so others can see what I did and try the fit themselves.

I used a Python workflow built around OpenCV and SciPy. The saved fit can be
replayed without connecting to the arm or camera. I've also included the scripts
I used for the camera's CAD placement and wrist attachment in Isaac.

I've kept the repo private for now, so you'll need access through GitHub.

## How I calibrated it

I started with the official TM printed board underneath both cameras, using the
arm's existing eye-in-hand (EIH) factory calibration to line up the observations.
I then fitted the Photoneo-to-flange transform directly from measured board points
and recorded arm poses: seven usable tilted views and eight translation views.

I calculated PARK and HORAUD hand-eye estimates with OpenCV. I used PARK to start
the tilt fit and kept HORAUD for comparison, then refined the fit with SciPy using
metric 3D least squares and a 0.3 mm soft-L1 loss.
I fitted a separate board pose for each capture session, so moving the board
between sessions was fine. I assumed the camera mount stayed fixed.

The final direct fit uses the Photoneo measurements and robot poses. The EIH
observations helped identify the printed board points; I didn't use its hand-eye
transform as a numerical pose prior in that final fit.

For homogeneous column vectors in millimetres:

```
p_base = T_base_flange @ T_flange_photoneo @ p_primary_camera
```

`T_flange_photoneo` maps the saved **CameraSpace / PrimaryCamera optical frame**
into the robot flange frame. The optical origin comes out at roughly
(-86.90, -97.60, 80.58) mm in flange coordinates, with the full rotation included.
That origin is different from the housing centre, mounting holes and centre of mass.

## Run the saved fit

I've tested this with Python on Linux. The original modules use the Unix
`resource` library, and the requirements keep OpenCV on the 4.x Python API.

```bash
python3 -m pip install -r requirements.txt
python3 replay_saved_fit.py
python3 -m unittest discover -s source/calibration -p 'test_tilt_calibration.py'
```

The replay starts with the saved numerical observations, fits the tilted views,
then runs the combined two-session fit. It checks the result against my saved
estimate and writes `replayed_fit.json`. It doesn't load the controller or camera
acquisition modules, and it blocks network connections while fitting.

## What's in here

- `source/calibration/`: my original fitting, board-corner/plane processing,
  camera alignment and sensitivity code, along with the numerical tests.
- `data/`: the numerical inputs for the replay and a note on how I exported them.
- `results/`: the saved transform, diagnostics and camera-body registration.
- `source/isaac_reference/`: the CAD/frame-registration and moving wrist scripts.
- `config/`: the exact mounting and collision settings I used for the wrist model.

The replay works from the supplied numerical inputs. Running the original image
processing from scratch needs the original capture layout and raw images/point
clouds, which aren't included here. The Isaac reference scripts also need the
robot/vendor assets and Isaac environment they were written for.

I've kept the original source files unchanged. `source-identities.json` records
their hashes, and `SHA256SUMS` lets you check the files in this repo.

## How I placed the camera in Isaac

I used the manufacturer's MotionCam-3D Color S housing mesh and kept its native
coordinates. I checked the relationship between that mesh and the PrimaryCamera
frame against the vendor ROS description, saved frame metadata and the optical
rays through the front windows. I kept the optical origin rather than moving the
mesh origin to the centre of the housing.

The fitted rotation preserves the camera's unusual mounting angle: about
22.563° between optical +Z and flange +Z. In the moving model, the camera is
attached to the wrist and follows the arm. I didn't carry over the early static
preview's snapshot-only flange correction. The URDF/XRDF/USD descriptions include
the camera attachment and the 23.15 mm Quick Changer/gripper stack extension.

The manufacturer mesh references and source URLs are in
`results/photoneo_body_registration.json`. This repo contains the integration
code and registration records; the CAD/robot assets and Isaac installation are
separate.

The original settings are in `config/wrist_collision.json` and
`config/wrist_mount.json`. These match the configuration hashes recorded with
the moving model. Here are the main values and where they came from:

| Setting | Value | Basis |
|---|---|---|
| Adapter/cap stack extension | 23.15 mm | My approximate measurement in place, supported by the mounting documentation |
| Camera collision padding | 5 mm on each side | Development margin |
| Collision sphere cell size | 25 mm | Modelling setting |
| Adapter radius bound | 35 mm | Assumed envelope |
| Support bounds in flange coordinates | XYZ minimum (−105, −35, −5) mm; maximum (−25, 35, 50) mm | Assumed occupied box |

I haven't measured the bracket envelope or flexible cable path. The padding and
assumed bounds don't describe calibrated measurement accuracy.

`wrist_mount.json` also keeps the earlier static preview settings: a 141 mm
housing-centre radius, 35 mm flange-Z offset and zero optical tilt. Those were
illustrative placeholders. The calibrated moving camera uses the fitted
`T_flange_mesh_m` from `results/photoneo_body_registration.json` for its pose;
those old static placement fields aren't used for that pose.

## Results and what still needs checking

The 15-view fit gives **0.3321338 mm RMS over 2,104 points**. When I left out one
view at a time, the largest mapped-point shift was **2.390 mm RMS**. Those numbers
show how well the observations agree and how sensitive the estimate is. They
don't establish the camera's absolute positioning accuracy.

I had limited Y rotation, and the planned combined-angle validation captures
weren't finished. I used a nominal 20 mm board pitch; the board dimensions,
scanner scale, robot absolute calibration and mount stability still need
independent checks. The projecting bracket geometry is approximate too.

I'm treating this as a development estimate for now. I haven't validated it for
precision pickup. Some original result JSON still contains old file paths for
reference; the saved-fit replay uses paths within this repo.
