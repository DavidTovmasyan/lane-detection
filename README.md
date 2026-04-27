# Lane Detection and Autonomous Steering Simulator

A real-time lane detection and autonomous steering system built entirely with classical computer vision techniques and PID control theory. The simulator renders a first-person camera view of a closed-loop track, detects lane markings using a traditional CV pipeline (Canny + Hough Transform), and steers a kinematic bicycle-model vehicle to stay centered in the lane -- all without deep learning.

![FPV Camera View with Lane Detection](docs/screenshot.png)

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Technical Details](#technical-details)
  - [Track Generation](#track-generation)
  - [Vehicle Model](#vehicle-model)
  - [FPV Camera](#fpv-camera)
  - [Vision Pipeline](#vision-pipeline)
  - [PID Controller](#pid-controller)
- [Installation](#installation)
- [Usage and Controls](#usage-and-controls)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Performance](#performance)

---

## Features

- **Classical Computer Vision Lane Detection** -- Grayscale conversion, Gaussian blur, Canny edge detection, trapezoidal region-of-interest masking, probabilistic Hough line transform, and weighted line averaging.
- **PID Steering Controller** -- Proportional-integral-derivative control with anti-windup clamping that steers the vehicle to minimize lane center offset.
- **Bicycle Kinematic Vehicle Model** -- Physically grounded 2D vehicle simulation with wheelbase, steering angle limits, and heading updates.
- **First-Person Perspective Camera** -- Pinhole camera model with configurable field of view, pitch angle, and mounting height. Renders the track with perspective projection including road surface, solid lane boundaries, and dashed center lines.
- **3-Panel Real-Time UI** -- Built with Pygame, the interface includes:
  - **Map View** (top-left) -- Top-down view of the full track with the vehicle marker and trailing path.
  - **FPV Camera View** (right) -- Large panel showing the rendered camera feed with lane detection overlays, confidence indicator, and HUD.
  - **Dashboard** (bottom-left) -- Live gauges for speed, steering angle, lane offset, FPS, PID component breakdown (P/I/D bar chart), offset history sparkline, and a steering arc gauge.
- **Catmull-Rom Spline Track** -- Smooth closed-loop track generated from editable waypoints using Catmull-Rom interpolation.
- **Temporal Smoothing** -- Detected lane lines and center offset are smoothed across frames to reduce jitter.
- **Real-Time Metrics** -- Speed, steering angle, lane offset, detection confidence, FPS, and individual PID component contributions are displayed and updated every frame.
- **Interactive Controls** -- Pause/resume, reset simulation, and adjust vehicle speed at runtime.

---

## System Architecture

The simulator follows a closed-loop pipeline that executes once per frame:

```
+-------+     +---------+     +-----+     +---------+     +--------+
| Track | --> | FPV     | --> | CV  | --> | PID     | --> | Vehicle|
| Data  |     | Camera  |     | Lane|     | Control |     | Model  |
|       |     | Render  |     | Det.|     |         |     | Update |
+-------+     +---------+     +-----+     +---------+     +--------+
                  ^                                            |
                  |          +----------+                      |
                  +--------- |  UI      | <--------------------+
                             | Renderer |
                             +----------+
```

**Data flow for a single frame:**

1. **Camera Render** -- The `FPVCamera` projects the road geometry ahead of the vehicle into a 640x480 pixel image using a pinhole perspective model.
2. **Lane Detection** -- The `LaneDetector` processes the rendered image through the classical CV pipeline to identify left and right lane boundaries and compute the vehicle's lateral offset from lane center.
3. **PID Control** -- The `PIDController` takes the normalized offset as error input and produces a steering angle command.
4. **Vehicle Update** -- The `Vehicle` applies the steering command through the bicycle kinematic model to update position and heading.
5. **UI Render** -- The `UI` class composites all three panels (map, FPV with annotations, dashboard) into the final display.

---

## Technical Details

### Track Generation

The track is defined by a set of 12 waypoints (configurable in `config.py`) that form a closed loop. A **Catmull-Rom spline** interpolates between these waypoints to produce a smooth centerline with 60 samples per segment (720 total points).

From the centerline, the system computes:

- **Tangent vectors** -- Normalized direction of travel at each point.
- **Normal vectors** -- Perpendicular to tangents (90-degree CCW rotation), used to offset left and right boundaries.
- **Lane boundaries** -- Centerline offset by half the road width (50 px each side for a 100 px road).
- **Center dashes** -- Alternating on/off segments (20 px dash, 15 px gap) computed by walking cumulative arc length.

The `Track` class also provides spatial queries: nearest centerline index with signed offset, heading at an index, and road geometry ahead of a given position.

### Vehicle Model

The vehicle uses a **bicycle kinematic model**, a standard simplification in autonomous driving that treats the car as a single-track vehicle:

```
x(t+dt) = x(t) + v * cos(theta) * dt
y(t+dt) = y(t) + v * sin(theta) * dt
theta(t+dt) = theta(t) + (v / L) * tan(delta) * dt
```

| Symbol    | Description           | Default Value |
|-----------|-----------------------|---------------|
| `v`       | Forward speed         | 100 px/s      |
| `L`       | Wheelbase             | 25 px         |
| `delta`   | Steering angle        | Computed by PID |
| `theta`   | Heading angle         | Track-aligned at start |
| `MAX_STEERING_ANGLE` | Steering limit | 40 degrees    |

The steering angle is clamped to the physical limit on every update. The timestep `dt` is capped at 50 ms to prevent numerical instability.

### FPV Camera

The camera simulates a **pinhole projection model** mounted on the vehicle:

| Parameter         | Value   | Description                           |
|-------------------|---------|---------------------------------------|
| Image size        | 640x480 | Output resolution                     |
| Field of view     | 70 deg  | Horizontal FOV                        |
| Mounting height   | 50 px   | Height above the ground plane         |
| Pitch angle       | 18 deg  | Downward tilt                         |
| Look-ahead        | 350 px  | How far ahead road geometry is fetched|

**Projection steps:**

1. Translate world points to car-relative coordinates.
2. Rotate to align the vehicle's forward direction with the camera's +Z axis.
3. Apply pitch rotation around the X-axis (downward tilt).
4. Perspective divide: `u = f * (x_cam / z) + cx`, `v = f * (y_cam / z) + cy`.
5. Filter out points behind the camera (`z <= 2.0`).

The renderer paints a gradient sky, a grass-colored ground plane below the computed horizon line, fills the road surface polygon, and draws solid white lane boundaries and dashed yellow center lines.

### Vision Pipeline

The `LaneDetector` implements a six-stage classical computer vision pipeline:

```
RGB Image
    |
    v
1. Grayscale Conversion (cv2.cvtColor)
    |
    v
2. Gaussian Blur (5x5 kernel)
    |
    v
3. Canny Edge Detection (thresholds: 50 / 150)
    |
    v
4. Region of Interest Mask (trapezoid, top at 45% image height)
    |
    v
5. Probabilistic Hough Line Transform
   (rho=1, theta=1 deg, threshold=25, minLength=30, maxGap=40)
    |
    v
6. Line Classification and Averaging
   - Negative slope + left half  --> left lane
   - Positive slope + right half --> right lane
   - Weighted average by segment length
   - Extrapolate to image boundaries
```

**Post-processing:**

- **Temporal smoothing** -- If a lane is not detected in the current frame, the previous detection is reused. The center offset is smoothed with an exponential moving average (alpha = 0.4).
- **Confidence scoring** -- 1.0 when both lanes are detected, 0.5 for a single lane, 0.0 when no lanes are found.
- **Annotation** -- The annotated image overlays detected lane lines (green), the vehicle center reference (red), the detected lane center (blue), and a semi-transparent ROI region (yellow).

### PID Controller

The controller computes a steering angle from the normalized lane offset error (range -1 to +1):

```
steering = Kp * e(t) + Ki * integral(e) + Kd * de/dt
```

| Gain  | Value | Role                                              |
|-------|-------|----------------------------------------------------|
| `Kp`  | 1.2   | Proportional -- reacts to current offset           |
| `Ki`  | 0.008 | Integral -- eliminates steady-state drift          |
| `Kd`  | 0.5   | Derivative -- dampens oscillation                  |

**Anti-windup:** The integral term is clamped to +/-50.0 to prevent runaway accumulation during sustained offsets or detection loss.

The output is clamped to the vehicle's maximum steering angle (+/-40 degrees). Individual P, I, and D contributions are exposed for real-time visualization on the dashboard.

---

## Installation

### Prerequisites

- Python 3.9 or later
- A display environment capable of running Pygame (not a headless server)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/davtovmas/lane-detection.git
cd lane-detection
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

The project depends on three packages:

| Package           | Minimum Version | Purpose                        |
|-------------------|-----------------|--------------------------------|
| `numpy`           | 1.24            | Numerical computation          |
| `opencv-python`   | 4.8             | Image processing and CV pipeline|
| `pygame-ce`       | 2.5             | Window management and rendering|

4. Run the simulator:

```bash
python main.py
```

---

## Usage and Controls

When launched, the simulator opens a 1400x800 window divided into three panels. The vehicle begins at the first track waypoint, heading aligned with the road, and immediately starts driving and steering autonomously.

### Keyboard Controls

| Key          | Action                                      |
|--------------|---------------------------------------------|
| `Space`      | Pause / resume the simulation               |
| `R`          | Reset the vehicle to the starting position  |
| `Up Arrow`   | Increase speed by 20 px/s (max 300 px/s)   |
| `Down Arrow` | Decrease speed by 20 px/s (min 40 px/s)    |
| `Esc`        | Quit the simulator                          |

### UI Panels

- **Map View** (500x400, top-left) -- Displays the full track with road surface, lane boundaries, dashed center line, the vehicle's trailing path with fade-in alpha, and a triangular car marker oriented by heading.
- **FPV Camera** (900x800, right) -- Shows the perspective-projected camera feed scaled to fill the panel. A HUD overlay at the top displays "LANE DETECTION ACTIVE" and the current detection confidence. Detected lanes are drawn in green, the vehicle center reference in red, and the estimated lane center in blue.
- **Dashboard** (500x400, bottom-left) -- Contains six visualization components:
  - Speed gauge (px/s with fill bar)
  - Steering angle gauge (degrees with centered bar)
  - Lane offset display (pixels with color-coded severity)
  - FPS counter
  - PID component bar chart (P in red, I in green, D in blue with signed centered bars and numeric values)
  - Bottom section split between an offset history sparkline (200-sample rolling window) and a semicircular steering arc gauge with tick marks, needle, and L/R labels.

---

## Configuration

All tunable parameters are centralized in `config.py`. Below are the key parameters grouped by subsystem.

### Window and Layout

| Parameter       | Default | Description                     |
|-----------------|---------|---------------------------------|
| `WINDOW_WIDTH`  | 1400    | Total window width in pixels    |
| `WINDOW_HEIGHT` | 800     | Total window height in pixels   |
| `FPS_TARGET`    | 60      | Target frame rate               |

### Track

| Parameter                | Default | Description                              |
|--------------------------|---------|------------------------------------------|
| `ROAD_WIDTH`             | 100.0   | Road width in world pixels               |
| `LANE_DASH_LENGTH`       | 20.0    | Center line dash length                  |
| `LANE_DASH_GAP`          | 15.0    | Center line gap length                   |
| `TRACK_SAMPLES_PER_SEGMENT` | 60   | Spline interpolation density             |
| `TRACK_WAYPOINTS`        | 12 pts  | List of (x, y) tuples defining the track |

### Vehicle

| Parameter            | Default     | Description                     |
|----------------------|-------------|---------------------------------|
| `WHEELBASE`          | 25.0        | Distance between axles          |
| `MAX_STEERING_ANGLE` | 40 degrees  | Maximum steering lock           |
| `VEHICLE_SPEED`      | 100.0       | Initial speed in px/s           |
| `CAR_LENGTH`         | 20.0        | Visual car length               |
| `CAR_WIDTH`          | 10.0        | Visual car width                |

### Camera

| Parameter           | Default | Description                          |
|---------------------|---------|--------------------------------------|
| `CAMERA_IMG_W`      | 640     | Rendered image width                 |
| `CAMERA_IMG_H`      | 480     | Rendered image height                |
| `CAMERA_FOV_DEG`    | 70.0    | Horizontal field of view in degrees  |
| `CAMERA_HEIGHT`     | 50.0    | Camera mounting height               |
| `CAMERA_PITCH_DEG`  | 18.0    | Downward pitch angle in degrees      |
| `CAMERA_LOOK_AHEAD` | 350.0   | Road geometry fetch distance         |

### Vision Pipeline

| Parameter          | Default    | Description                            |
|--------------------|------------|----------------------------------------|
| `GAUSSIAN_KERNEL`  | (5, 5)     | Blur kernel size                       |
| `CANNY_LOW`        | 50         | Canny lower threshold                  |
| `CANNY_HIGH`       | 150        | Canny upper threshold                  |
| `HOUGH_THRESHOLD`  | 25         | Hough accumulator threshold            |
| `HOUGH_MIN_LINE_LEN` | 30      | Minimum detected line length           |
| `HOUGH_MAX_LINE_GAP`  | 40      | Maximum gap between line segments      |
| `ROI_TOP_RATIO`    | 0.45       | ROI top boundary as fraction of height |
| `SLOPE_MIN`        | 0.3        | Minimum absolute slope to accept a line|
| `SMOOTHING_ALPHA`  | 0.4        | Offset EMA smoothing factor            |

### PID Controller

| Parameter            | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `KP`                 | 1.2     | Proportional gain                    |
| `KI`                 | 0.008   | Integral gain                        |
| `KD`                 | 0.5     | Derivative gain                      |
| `PID_INTEGRAL_LIMIT` | 50.0    | Anti-windup integral clamp           |

---

## Project Structure

```
lane-detection/
├── main.py              # Entry point: simulation loop, event handling
├── config.py            # All constants and tunable parameters
├── track.py             # Track definition with Catmull-Rom spline interpolation
├── vehicle.py           # Bicycle kinematic vehicle model
├── camera.py            # FPV camera with pinhole perspective projection
├── vision.py            # Classical CV lane detection pipeline
├── controller.py        # PID steering controller with anti-windup
├── ui.py                # 3-panel Pygame UI (map, FPV, dashboard)
├── requirements.txt     # Python dependencies
├── description.md       # Original project specification
├── docs/
│   └── screenshot.png   # Screenshot of the simulator in action
└── README.md            # This file
```

---

## How It Works

Below is a step-by-step walkthrough of a single simulation frame.

### 1. Clock Tick

`pygame.time.Clock.tick(60)` limits the frame rate and returns the elapsed time `dt` in milliseconds. The value is converted to seconds and capped at 50 ms to prevent physics instability from lag spikes. FPS is tracked with an exponential moving average (95/5 blend).

### 2. Event Processing

Pygame events are polled for quit, pause toggle (`Space`), reset (`R`), speed adjustment (`Up`/`Down`), and exit (`Esc`). If paused, the UI is drawn without advancing the simulation state.

### 3. FPV Camera Render

The `FPVCamera.render()` method:

1. Copies the pre-computed sky gradient onto a blank 640x480 image.
2. Computes the horizon line from the camera pitch and fills everything below it with grass color.
3. Queries the track for road geometry (left boundary, right boundary, center line) ahead of the vehicle up to the look-ahead distance (350 px).
4. Projects all 3D ground-plane points into 2D image coordinates using the pinhole model with pitch rotation.
5. Fills the road surface polygon between the projected left and right boundaries.
6. Draws solid white polylines for both lane boundaries.
7. Draws dashed yellow segments for the center line.

### 4. Lane Detection

The `LaneDetector.detect()` method receives the rendered RGB image and:

1. Converts to grayscale.
2. Applies 5x5 Gaussian blur to suppress noise.
3. Runs Canny edge detection (thresholds 50/150).
4. Masks the edges with a trapezoidal ROI (bottom full-width, top narrowed to 35%-65% width at 45% height).
5. Runs the probabilistic Hough line transform to extract line segments.
6. Classifies each segment as left lane (negative slope, left half) or right lane (positive slope, right half), filtering by a minimum slope threshold of 0.3.
7. Computes a length-weighted average slope and intercept for each side, then extrapolates to the image boundaries.
8. Applies temporal smoothing: reuses the previous detection if a lane disappears, and smooths the center offset with an EMA.
9. Computes the lateral offset between the detected lane center and the image center.
10. Annotates the image with green lane lines, a red vehicle center reference, a blue lane center marker, and a semi-transparent ROI overlay.

### 5. PID Steering

The detected center offset is normalized to the range [-1, +1] by dividing by half the image width. The PID controller:

1. Computes the proportional term: `P = Kp * error`.
2. Accumulates the integral: `I = Ki * clamp(integral + error * dt)`.
3. Computes the derivative: `D = Kd * (error - prev_error) / dt`.
4. Sums P + I + D and clamps the result to the maximum steering angle.
5. Stores individual P, I, D values for dashboard visualization.

### 6. Vehicle Update

The bicycle kinematic model advances the vehicle state:

1. Clamps the steering command to the physical steering limit.
2. Updates position: `x += speed * cos(heading) * dt`, `y += speed * sin(heading) * dt`.
3. Updates heading: `heading += (speed / wheelbase) * tan(steering) * dt` (skipped for near-zero steering to avoid division artifacts).
4. Appends the new position to the trailing path buffer (400-point deque).

### 7. UI Render

The `UI.draw()` method composites all three panels onto the Pygame display surface:

- The **map panel** transforms world coordinates to screen space, draws the road and boundaries, renders the fading trail, and places the triangular vehicle marker.
- The **FPV panel** converts the annotated OpenCV image (NumPy array) to a Pygame surface via `surfarray.make_surface()`, scales it to fill the 900x800 panel, and overlays the HUD bar with detection status and confidence.
- The **dashboard** renders metric cards with animated value interpolation, centered bar gauges, the PID component breakdown, the offset history sparkline, and the semicircular steering arc gauge.

Finally, `pygame.display.flip()` presents the completed frame.

---

## Performance

The simulator is designed to run comfortably above 60 FPS on modern hardware. The frame rate target is set to 60 FPS via `pygame.time.Clock.tick()`.

Key performance characteristics:

- **Rendering** -- The FPV camera uses NumPy vectorized operations for coordinate transforms and OpenCV for polygon fills and line drawing, avoiding per-pixel Python loops.
- **Vision pipeline** -- All CV operations (blur, Canny, Hough) are backed by OpenCV's optimized C++ implementations.
- **Sky gradient** -- Pre-computed once at initialization and reused via array copy each frame.
- **Spline geometry** -- Track interpolation, tangents, normals, and boundaries are computed once at startup. Per-frame work is limited to spatial queries (nearest-index lookup and ahead-of-car slicing).
- **FPS tracking** -- An exponential moving average (5% new, 95% old) provides a stable FPS readout displayed on the dashboard.

Typical performance on a modern laptop exceeds 60 FPS with all three panels rendering simultaneously.

---

## License

This project was developed as a demonstration of classical computer vision and control theory applied to autonomous driving simulation.
