"""Centralized configuration for the lane detection simulator."""

import math

# ── Window ──────────────────────────────────────────────────────────
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS_TARGET = 60
WINDOW_TITLE = "Lane Detection & Autonomous Steering Simulator"

# ── Panel Layout ────────────────────────────────────────────────────
MAP_X, MAP_Y = 0, 0
MAP_W, MAP_H = 500, 400

DASH_X, DASH_Y = 0, 400
DASH_W, DASH_H = 500, 400

FPV_X, FPV_Y = 500, 0
FPV_W, FPV_H = 900, 800

# ── Colors (RGB) ────────────────────────────────────────────────────
BG_DARK = (18, 18, 24)
BG_PANEL = (24, 28, 36)
BG_CARD = (32, 38, 48)
SEPARATOR = (48, 54, 68)

ROAD_SURFACE = (64, 68, 74)
GRASS_COLOR = (34, 52, 30)
LANE_WHITE = (220, 225, 230)
LANE_DASH_COLOR = (200, 200, 80)
SKY_TOP = (25, 28, 40)
SKY_BOTTOM = (70, 85, 110)

CAR_BODY = (0, 180, 255)
CAR_OUTLINE = (255, 255, 255)
TRAIL_COLOR = (0, 140, 220)

LANE_DETECT_GREEN = (0, 255, 100)
CENTER_BLUE = (60, 140, 255)
VEHICLE_CENTER_RED = (255, 60, 60)
ROI_OVERLAY_COLOR = (255, 200, 0, 40)

TEXT_PRIMARY = (230, 235, 240)
TEXT_SECONDARY = (140, 150, 165)
TEXT_ACCENT = (0, 200, 255)
GAUGE_BG = (40, 46, 58)
GAUGE_FILL_NORMAL = (0, 200, 120)
GAUGE_FILL_WARN = (255, 180, 0)
GAUGE_FILL_DANGER = (255, 50, 50)
PID_P_COLOR = (255, 100, 80)
PID_I_COLOR = (80, 200, 120)
PID_D_COLOR = (80, 140, 255)

# ── Track ───────────────────────────────────────────────────────────
LANE_DASH_LENGTH = 20.0
LANE_DASH_GAP = 15.0
TRACK_SAMPLES_PER_SEGMENT = 60
DEFAULT_ROAD_WIDTH = 100.0

# Track definitions: name -> {waypoints, road_width, description, difficulty}
TRACKS = {
    "oval": {
        "description": "Simple Oval",
        "difficulty": 1,
        "road_width": 110.0,
        "speed": 100.0,
        "waypoints": [
            (350, 700), (700, 720), (950, 650), (1100, 500),
            (1150, 300), (1050, 150), (850, 80), (600, 100),
            (400, 130), (250, 220), (180, 380), (200, 550),
        ],
    },
    "stadium": {
        "description": "Stadium Circuit",
        "difficulty": 2,
        "road_width": 105.0,
        "speed": 95.0,
        "waypoints": [
            # Bottom straight
            (250, 700), (550, 720), (800, 700),
            # Right sweeping curve
            (1000, 620), (1100, 480), (1080, 320),
            # Top kink
            (1000, 200), (850, 120), (650, 100),
            # Left side long curve
            (420, 120), (250, 200), (180, 350),
            (170, 520), (200, 640),
        ],
    },
    "snake": {
        "description": "Winding Road",
        "difficulty": 3,
        "road_width": 100.0,
        "speed": 85.0,
        "waypoints": [
            # Bottom straight
            (250, 700), (550, 720), (800, 680),
            # Right side curve up
            (1000, 560), (1080, 380),
            # Top right
            (1020, 200), (850, 120),
            # Top straight
            (600, 100), (380, 130),
            # Left side curve down
            (230, 250), (180, 420),
            # Bottom left
            (200, 580),
        ],
    },
    "grand_prix": {
        "description": "Grand Prix Circuit",
        "difficulty": 4,
        "road_width": 100.0,
        "speed": 80.0,
        "waypoints": [
            # Long start/finish straight
            (200, 720), (500, 730), (800, 710),
            # Wide right-hander
            (1000, 640), (1100, 480),
            # Medium left
            (1050, 300), (920, 180),
            # Top straight
            (700, 100), (480, 110),
            # Left-hand return
            (300, 180), (210, 320),
            # Gentle S
            (240, 480), (210, 620),
        ],
    },
    "mountain": {
        "description": "Highland Rally",
        "difficulty": 5,
        "road_width": 95.0,
        "speed": 75.0,
        "waypoints": [
            (200, 720), (450, 740), (700, 700),
            # Right sweep uphill
            (920, 600), (1050, 440),
            # Wide summit bend
            (1080, 260), (980, 140), (800, 80),
            # Top traverse
            (580, 80), (380, 130),
            # Descent
            (250, 260), (200, 430),
            # Valley return
            (180, 590),
        ],
    },
    "redbull_ring": {
        "description": "Red Bull Ring",
        "difficulty": 5,
        "road_width": 120.0,
        "speed": 120.0,
        "adaptive_speed": True,
        "waypoints": [
            # Start/finish straight (bottom, car heads LEFT)
            (800, 720), (600, 700),
            # T1 -- right-hander uphill
            (420, 640), (310, 540),
            # T2 -- right uphill
            (230, 420),
            # T3 -- tight right hairpin (leftmost point)
            (140, 300), (160, 200),
            # Climb right toward T4
            (280, 120),
            # T4 -- sharp right at the peak
            (480, 60), (550, 80),
            # T5 kink / T6 descent
            (500, 170), (400, 290),
            # T7 -- sharp right (center)
            (400, 430), (470, 490),
            # T8 -- right onto back straight
            (560, 420), (680, 340),
            # Sector 3 back straight heading right
            (820, 260),
            # T9 -- right (top-right)
            (960, 200), (1040, 260),
            # T10 -- right hairpin (far right)
            (1070, 380), (1010, 480),
            # Straight back down to start/finish
            (920, 600), (860, 680),
        ],
    },
}

DEFAULT_TRACK = "oval"

# Legacy compat: used by code that reads ROAD_WIDTH directly
ROAD_WIDTH = DEFAULT_ROAD_WIDTH

# ── Vehicle ─────────────────────────────────────────────────────────
WHEELBASE = 25.0
MAX_STEERING_ANGLE = math.radians(40)
VEHICLE_SPEED = 100.0          # pixels/sec
CAR_LENGTH = 20.0
CAR_WIDTH = 10.0

# ── Camera ──────────────────────────────────────────────────────────
CAMERA_IMG_W = 640
CAMERA_IMG_H = 480
CAMERA_FOV_DEG = 70.0
CAMERA_HEIGHT = 50.0
CAMERA_PITCH_DEG = 18.0
CAMERA_LOOK_AHEAD = 350.0

# ── Vision Pipeline ─────────────────────────────────────────────────
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 40
ROI_TOP_RATIO = 0.45
SLOPE_MIN = 0.3
SMOOTHING_ALPHA = 0.4

# ── PID Controller ──────────────────────────────────────────────────
KP = 1.2
KI = 0.008
KD = 0.5
PID_INTEGRAL_LIMIT = 50.0

# ── Adaptive Speed Controller ───────────────────────────────────────
SPEED_MIN = 35.0               # minimum speed (px/s)
SPEED_CURVATURE_GAIN = 0.35    # lower = slower in curves
SPEED_LOOK_AHEAD_DIST = 300.0  # how far ahead to check curvature
SPEED_SMOOTHING = 0.08         # how fast speed adjusts (per frame blend)

# ── Map Panel ───────────────────────────────────────────────────────
MAP_PADDING = 15
MAP_TITLE_HEIGHT = 30
TRAIL_MAX_LENGTH = 400

# ── Fonts ───────────────────────────────────────────────────────────
FONT_MONO = "consolas"
FONT_SANS = "helvetica"
