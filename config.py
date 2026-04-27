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
ROAD_WIDTH = 100.0
LANE_DASH_LENGTH = 20.0
LANE_DASH_GAP = 15.0
TRACK_SAMPLES_PER_SEGMENT = 60

TRACK_WAYPOINTS = [
    (350, 700),
    (700, 720),
    (950, 650),
    (1100, 500),
    (1150, 300),
    (1050, 150),
    (850, 80),
    (600, 100),
    (400, 130),
    (250, 220),
    (180, 380),
    (200, 550),
]

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

# ── Map Panel ───────────────────────────────────────────────────────
MAP_PADDING = 15
MAP_TITLE_HEIGHT = 30
TRAIL_MAX_LENGTH = 400

# ── Fonts ───────────────────────────────────────────────────────────
FONT_MONO = "consolas"
FONT_SANS = "helvetica"
