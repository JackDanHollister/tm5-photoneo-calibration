# Reply about calibration software and code

We did not use your robot-calibration tool to estimate the flange-to-sensor
transformation. We used a custom offline Python workflow.

Initially, we aligned the Photoneo and the robot's eye-in-hand camera using the
official TM calibration board and the existing EIH factory calibration. We then
estimated the Photoneo-to-flange transform directly from measured board points
and recorded robot flange poses: eight translation views and seven usable tilt
views. OpenCV hand-eye initialisation was followed by robust metric refinement
with SciPy, fitting separate fixed board poses for the two capture sessions.

The resulting transform maps the saved Photoneo PrimaryCamera optical coordinate
frame into the robot flange frame. For the Isaac model, we retained your CAD's
native coordinates and used the fitted transform to place the housing. The
housing centre is distinct from the calibrated sensor origin.

We can share the fitting and CAD/frame-registration code, the numerical input
records, the resulting transformation and a saved-data replay. They are provided
in this repository. The replay reproduces our estimate without a robot, camera
or Isaac connection. The original Isaac integration scripts are also included as
reference; those depend on our robot/vendor assets and Isaac environment.

The estimate remains a development calibration. Its0.332mm fitting RMS measures
agreement with the observations, rather than independently established absolute
accuracy. Limited rotational diversity and remaining sensitivity are documented.
