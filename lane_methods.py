"""Lane detection methods compared in this project.

Three detectors are exposed with a common interface (``Detection`` /
``LaneDetectorBase``) so the simulator and the headless evaluator can plug
any of them in without code changes:

    1. CentroidDetector  -- simplest baseline. Threshold the road in HSV,
       compute the centroid of the lane region in a ROI, return its
       horizontal offset from the image centre.

    2. HoughDetector     -- classical baseline (Canny + ROI mask +
       probabilistic Hough). This is the method described in the project
       brief and originally implemented in ``vision.py``.

    3. PolyFitDetector   -- proposed improvement. An inverse-perspective
       (bird's-eye) warp followed by a histogram + sliding-window search
       and a 2nd-order polynomial fit per lane. Handles curved lanes,
       which the straight-line Hough detector cannot model.

All three detectors return a :class:`Detection` whose ``center_offset``
is in *pixels* (image-frame), positive when the lane centre is to the
right of the image centre. A predicted lane mask is also produced so the
evaluator can compute lane IoU against the ground-truth road polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

import config as cfg


# ──────────────────────────────────────────────────────────────────────
# Shared result object
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Output of a lane detector for a single frame.

    Attributes
    ----------
    center_offset : float
        Horizontal offset (pixels) of the detected lane centre relative
        to the image centre. Positive means the lane centre is to the
        right of the camera centre, so the car should steer right.
    confidence : float
        In [0, 1]. 1.0 means both lane boundaries detected, ~0.5 means
        only one detected (the other estimated), 0.0 means none.
    lookahead_offset : float
        Horizontal offset evaluated at the *look-ahead* line (a fixed
        fraction of the way up the image), not at the bottom. Equal to
        ``center_offset`` for line-based detectors but distinct for the
        polynomial detector (where it is informative on curves).
    annotated : np.ndarray
        BGR/RGB image with overlays for the UI.
    pred_mask : np.ndarray
        Binary mask (uint8 0/255) of the predicted drivable lane region,
        same size as the input frame. Used for IoU computation.
    method : str
        Identifier of the detector.
    """
    center_offset: float = 0.0
    confidence: float = 0.0
    lookahead_offset: float = 0.0
    annotated: Optional[np.ndarray] = None
    pred_mask: Optional[np.ndarray] = None
    method: str = ""
    # Optional extras kept so the UI can keep drawing the existing
    # straight-line overlay without changes.
    left_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    right_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    edges: Optional[np.ndarray] = None


class LaneDetectorBase:
    name = "base"

    def detect(self, img: np.ndarray) -> Detection:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────
# 1) Centroid baseline -- simplest possible method
# ──────────────────────────────────────────────────────────────────────

class CentroidDetector(LaneDetectorBase):
    """Baseline #1: thresholded-ROI centroid.

    Threshold the road surface in HSV (the simulator paints the road a
    uniform dark grey), restrict to a trapezoidal ROI, compute the
    centroid of the road region, and use its horizontal coordinate as
    the lane centre. No line fitting, no temporal filtering -- it is
    deliberately as simple as possible to give a fair lower bound.
    """

    name = "centroid"

    def __init__(self):
        self._prev_offset = 0.0

    def reset(self) -> None:
        self._prev_offset = 0.0

    def detect(self, img: np.ndarray) -> Detection:
        h, w = img.shape[:2]
        det = Detection(method=self.name)

        # Threshold the road colour: simulator road is RGB ~ (64,68,74),
        # a low-saturation dark grey. Pick low-V, low-S pixels.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        road = ((s < 40) & (v > 40) & (v < 110)).astype(np.uint8) * 255

        # ROI: trapezoid identical to the Hough detector for fairness.
        roi_top = int(h * cfg.ROI_TOP_RATIO)
        vertices = np.array([[
            (0, h),
            (int(w * 0.35), roi_top),
            (int(w * 0.65), roi_top),
            (w, h),
        ]], dtype=np.int32)
        roi_mask = np.zeros_like(road)
        cv2.fillPoly(roi_mask, vertices, 255)
        masked = cv2.bitwise_and(road, roi_mask)

        # Centroid via image moments.
        m = cv2.moments(masked, binaryImage=True)
        if m["m00"] > 200:
            cx = m["m10"] / m["m00"]
            offset = cx - w / 2
            det.confidence = float(np.clip(m["m00"] / (w * h * 0.25), 0.0, 1.0))
        else:
            cx = w / 2 + self._prev_offset
            offset = self._prev_offset
            det.confidence = 0.0

        # Light temporal smoothing.
        offset = 0.5 * offset + 0.5 * self._prev_offset
        self._prev_offset = offset
        det.center_offset = float(offset)
        det.lookahead_offset = float(offset)

        # Annotation
        ann = img.copy()
        overlay = ann.copy()
        cv2.fillPoly(overlay, vertices, (255, 200, 0))
        cv2.addWeighted(overlay, 0.08, ann, 0.92, 0, ann)
        # Show the segmented road as a tint
        tint = np.zeros_like(ann)
        tint[masked > 0] = (0, 200, 255)
        cv2.addWeighted(tint, 0.25, ann, 0.75, 0, ann)
        # Centre markers
        cv2.line(ann, (w // 2, h), (w // 2, int(h * 0.6)),
                 cfg.VEHICLE_CENTER_RED, 2, cv2.LINE_AA)
        cv2.line(ann, (int(cx), h), (int(cx), int(h * 0.65)),
                 cfg.CENTER_BLUE, 2, cv2.LINE_AA)
        cv2.circle(ann, (int(cx), int(h * 0.75)), 6, cfg.CENTER_BLUE, -1, cv2.LINE_AA)

        det.annotated = ann
        det.pred_mask = masked
        return det


# ──────────────────────────────────────────────────────────────────────
# 2) Hough baseline -- the classical pipeline from the brief
# ──────────────────────────────────────────────────────────────────────

class HoughDetector(LaneDetectorBase):
    """Baseline #2: Canny + ROI + probabilistic Hough.

    This is the canonical "classical CV" lane detector described in the
    project brief. It approximates each lane boundary as a single
    straight line. Works very well on straights, breaks down on tight
    curves (line of best fit underestimates curvature).
    """

    name = "hough"

    def __init__(self):
        self._prev_offset = 0.0
        self._prev_left = None
        self._prev_right = None

    def reset(self) -> None:
        self._prev_offset = 0.0
        self._prev_left = None
        self._prev_right = None

    def detect(self, img: np.ndarray) -> Detection:
        h, w = img.shape[:2]
        det = Detection(method=self.name)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, cfg.GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blurred, cfg.CANNY_LOW, cfg.CANNY_HIGH)
        det.edges = edges

        roi_top = int(h * cfg.ROI_TOP_RATIO)
        vertices = np.array([[
            (0, h),
            (int(w * 0.35), roi_top),
            (int(w * 0.65), roi_top),
            (w, h),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked,
            rho=cfg.HOUGH_RHO,
            theta=cfg.HOUGH_THETA,
            threshold=cfg.HOUGH_THRESHOLD,
            minLineLength=cfg.HOUGH_MIN_LINE_LEN,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP,
        )

        left_segs, right_segs = [], []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < cfg.SLOPE_MIN:
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                mid_x = (x1 + x2) / 2
                if slope < 0 and mid_x < w / 2:
                    left_segs.append((slope, x1, y1, x2, y2, length))
                elif slope > 0 and mid_x > w / 2:
                    right_segs.append((slope, x1, y1, x2, y2, length))

        left = _avg_line(left_segs, h, roi_top)
        right = _avg_line(right_segs, h, roi_top)

        if left is not None:
            self._prev_left = left
        elif self._prev_left is not None:
            left = self._prev_left
        if right is not None:
            self._prev_right = right
        elif self._prev_right is not None:
            right = self._prev_right

        det.left_line = left
        det.right_line = right

        if left and right:
            lx_b, rx_b = left[0][0], right[0][0]
            offset = (lx_b + rx_b) / 2 - w / 2
            lx_t, rx_t = left[1][0], right[1][0]
            la_offset = (lx_t + rx_t) / 2 - w / 2
            det.confidence = 1.0
        elif left:
            lx_b = left[0][0]
            offset = (lx_b + cfg.ROAD_WIDTH * 2) - w / 2
            la_offset = offset
            det.confidence = 0.5
        elif right:
            rx_b = right[0][0]
            offset = (rx_b - cfg.ROAD_WIDTH * 2) - w / 2
            la_offset = offset
            det.confidence = 0.5
        else:
            offset = self._prev_offset
            la_offset = offset
            det.confidence = 0.0

        alpha = cfg.SMOOTHING_ALPHA
        offset = alpha * offset + (1 - alpha) * self._prev_offset
        self._prev_offset = offset
        det.center_offset = float(offset)
        det.lookahead_offset = float(la_offset)

        # Annotate
        ann = img.copy()
        overlay = ann.copy()
        cv2.fillPoly(overlay, vertices, (255, 200, 0))
        cv2.addWeighted(overlay, 0.08, ann, 0.92, 0, ann)
        if left:
            cv2.line(ann, left[0], left[1],
                     cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)
        if right:
            cv2.line(ann, right[0], right[1],
                     cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)
        cv2.line(ann, (w // 2, h), (w // 2, int(h * 0.6)),
                 cfg.VEHICLE_CENTER_RED, 2, cv2.LINE_AA)
        if left and right:
            lcx = int((left[0][0] + right[0][0]) / 2)
            cv2.line(ann, (lcx, h), (lcx, int(h * 0.65)),
                     cfg.CENTER_BLUE, 2, cv2.LINE_AA)
            cv2.circle(ann, (lcx, int(h * 0.75)), 6, cfg.CENTER_BLUE, -1, cv2.LINE_AA)

        det.annotated = ann

        # Predicted lane mask: quadrilateral between the two lines.
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if left and right:
            poly = np.array([[left[0], left[1], right[1], right[0]]], dtype=np.int32)
            cv2.fillPoly(pred_mask, poly, 255)
        det.pred_mask = pred_mask
        return det


def _avg_line(segments, img_h, roi_top):
    if not segments:
        return None
    total = sum(s[5] for s in segments)
    if total < 1e-6:
        return None
    slope = sum(s[0] * s[5] for s in segments) / total
    intercept = sum(((s[2] - s[0] * s[1]) * s[5]) for s in segments) / total
    if abs(slope) < 1e-6:
        return None
    y_b, y_t = img_h - 1, roi_top
    x_b = int((y_b - intercept) / slope)
    x_t = int((y_t - intercept) / slope)
    return ((x_b, y_b), (x_t, y_t))


# ──────────────────────────────────────────────────────────────────────
# 3) Sliding-window polynomial-fit detector -- proposed improvement
# ──────────────────────────────────────────────────────────────────────

class PolyFitDetector(LaneDetectorBase):
    """Proposed method: bird's-eye sliding-window polynomial fit.

    Pipeline:
      1. Threshold lane markings (Sobel-x + value threshold).
      2. Inverse-perspective (IPM) warp to a top-down view.
      3. Bottom-half histogram to find the two lane bases.
      4. 9 vertical sliding windows tracking each base upwards;
         collect lane pixels and re-centre each window on their mean.
      5. Fit a 2nd-order polynomial ``x = a y^2 + b y + c`` per lane.
      6. Compute the lane centre at the *bottom* of the bird's-eye view
         (vehicle position) and at a *look-ahead* row.
      7. Warp the lane polygon back to the camera image for overlay and
         for the predicted-mask IoU.

    The polynomial fit lets the steering controller see the curve coming
    rather than only the straight-line tangent. The look-ahead offset is
    what the controller actually uses in the experiments below; it is
    the equivalent of a 1-frame predictor.
    """

    name = "polyfit"

    BEV_W = 320
    BEV_H = 320
    N_WINDOWS = 9
    WIN_MARGIN = 50
    MIN_PIX = 50
    LOOKAHEAD_RATIO = 0.45  # fraction of BEV height from the bottom

    def __init__(self):
        self._prev_left_fit = None
        self._prev_right_fit = None
        self._prev_offset = 0.0
        self._prev_la = 0.0
        self._M = None
        self._Minv = None

    def reset(self) -> None:
        self._prev_left_fit = None
        self._prev_right_fit = None
        self._prev_offset = 0.0
        self._prev_la = 0.0

    # ── perspective warp ────────────────────────────────────────────

    def _warp_matrices(self, h, w):
        if self._M is not None:
            return self._M, self._Minv
        # Source: trapezoid matching the ROI used by the other methods.
        src = np.float32([
            [w * 0.10, h - 1],
            [w * 0.35, h * cfg.ROI_TOP_RATIO],
            [w * 0.65, h * cfg.ROI_TOP_RATIO],
            [w * 0.90, h - 1],
        ])
        dst = np.float32([
            [self.BEV_W * 0.15, self.BEV_H - 1],
            [self.BEV_W * 0.15, 0],
            [self.BEV_W * 0.85, 0],
            [self.BEV_W * 0.85, self.BEV_H - 1],
        ])
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)
        return self._M, self._Minv

    # ── pixel mask ──────────────────────────────────────────────────

    def _lane_pixels(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sx = np.abs(sx)
        sx = (sx / max(sx.max(), 1e-3) * 255).astype(np.uint8)
        edge = (sx > 60).astype(np.uint8)

        # White lane markings: high V and low S.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)).astype(np.uint8)

        # Yellow centre dashes (in our sim ~ (200,200,80))
        yellow = ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40)
                  & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 130)).astype(np.uint8)

        return (edge | white | yellow) * 255

    # ── main entry ──────────────────────────────────────────────────

    def detect(self, img: np.ndarray) -> Detection:
        h, w = img.shape[:2]
        det = Detection(method=self.name)

        M, Minv = self._warp_matrices(h, w)
        mask = self._lane_pixels(img)
        bev = cv2.warpPerspective(mask, M, (self.BEV_W, self.BEV_H),
                                  flags=cv2.INTER_NEAREST)

        # Histogram of the bottom half to locate lane bases.
        bottom = bev[self.BEV_H // 2:, :]
        hist = np.sum(bottom, axis=0)
        mid = self.BEV_W // 2
        if hist.max() < 1:
            return self._fallback(img, det)

        leftx_base = int(np.argmax(hist[:mid]))
        rightx_base = int(np.argmax(hist[mid:]) + mid)

        win_h = self.BEV_H // self.N_WINDOWS
        nz = bev.nonzero()
        nzy, nzx = nz[0], nz[1]

        left_idx, right_idx = [], []
        leftx_cur, rightx_cur = leftx_base, rightx_base
        margin = self.WIN_MARGIN
        for wi in range(self.N_WINDOWS):
            y_lo = self.BEV_H - (wi + 1) * win_h
            y_hi = self.BEV_H - wi * win_h
            xl_lo, xl_hi = leftx_cur - margin, leftx_cur + margin
            xr_lo, xr_hi = rightx_cur - margin, rightx_cur + margin
            good_left = ((nzy >= y_lo) & (nzy < y_hi)
                         & (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            good_right = ((nzy >= y_lo) & (nzy < y_hi)
                          & (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]
            left_idx.append(good_left)
            right_idx.append(good_right)
            if len(good_left) > self.MIN_PIX:
                leftx_cur = int(nzx[good_left].mean())
            if len(good_right) > self.MIN_PIX:
                rightx_cur = int(nzx[good_right].mean())

        left_idx = np.concatenate(left_idx) if left_idx else np.array([], dtype=int)
        right_idx = np.concatenate(right_idx) if right_idx else np.array([], dtype=int)

        left_fit = None
        right_fit = None
        if len(left_idx) > 200:
            left_fit = np.polyfit(nzy[left_idx], nzx[left_idx], 2)
        if len(right_idx) > 200:
            right_fit = np.polyfit(nzy[right_idx], nzx[right_idx], 2)

        # Light temporal smoothing of the polynomial coefficients.
        if left_fit is not None and self._prev_left_fit is not None:
            left_fit = 0.5 * left_fit + 0.5 * self._prev_left_fit
        if right_fit is not None and self._prev_right_fit is not None:
            right_fit = 0.5 * right_fit + 0.5 * self._prev_right_fit
        if left_fit is None:
            left_fit = self._prev_left_fit
        if right_fit is None:
            right_fit = self._prev_right_fit
        self._prev_left_fit = left_fit
        self._prev_right_fit = right_fit

        if left_fit is None or right_fit is None:
            return self._fallback(img, det)

        # Evaluate at car position (bottom) and at look-ahead row.
        y_eval_bot = self.BEV_H - 1
        y_eval_la = int(self.BEV_H * (1 - self.LOOKAHEAD_RATIO))

        lx_bot = np.polyval(left_fit, y_eval_bot)
        rx_bot = np.polyval(right_fit, y_eval_bot)
        lx_la = np.polyval(left_fit, y_eval_la)
        rx_la = np.polyval(right_fit, y_eval_la)

        lane_cx_bot = (lx_bot + rx_bot) / 2
        lane_cx_la = (lx_la + rx_la) / 2

        # Offsets in bird's-eye-view pixel units; convert to camera-image
        # pixel units by scaling with the camera-image width (because the
        # PID gain was tuned in camera-image pixels for the Hough
        # baseline, this keeps controller gains comparable).
        scale_x = w / self.BEV_W
        offset_bot = (lane_cx_bot - self.BEV_W / 2) * scale_x
        offset_la = (lane_cx_la - self.BEV_W / 2) * scale_x

        # Light smoothing
        alpha = cfg.SMOOTHING_ALPHA
        offset_bot = alpha * offset_bot + (1 - alpha) * self._prev_offset
        offset_la = alpha * offset_la + (1 - alpha) * self._prev_la
        self._prev_offset = offset_bot
        self._prev_la = offset_la

        det.center_offset = float(offset_bot)
        det.lookahead_offset = float(offset_la)
        det.confidence = 1.0

        # Build lane polygon in BEV, warp back to camera frame.
        ploty = np.linspace(0, self.BEV_H - 1, 60)
        left_curve = np.polyval(left_fit, ploty)
        right_curve = np.polyval(right_fit, ploty)
        bev_pts_left = np.column_stack([left_curve, ploty])
        bev_pts_right = np.column_stack([right_curve, ploty])
        bev_poly = np.vstack([bev_pts_left, bev_pts_right[::-1]]).astype(np.float32)
        cam_poly = cv2.perspectiveTransform(bev_poly.reshape(-1, 1, 2), Minv).reshape(-1, 2)

        pred_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [cam_poly.astype(np.int32)], 255)
        det.pred_mask = pred_mask

        # Annotation
        ann = img.copy()
        overlay = np.zeros_like(ann)
        cv2.fillPoly(overlay, [cam_poly.astype(np.int32)], (0, 200, 100))
        cv2.addWeighted(overlay, 0.25, ann, 0.85, 0, ann)

        # Draw boundary polylines back in camera frame
        left_cam = cv2.perspectiveTransform(
            bev_pts_left.astype(np.float32).reshape(-1, 1, 2), Minv
        ).reshape(-1, 2).astype(np.int32)
        right_cam = cv2.perspectiveTransform(
            bev_pts_right.astype(np.float32).reshape(-1, 1, 2), Minv
        ).reshape(-1, 2).astype(np.int32)
        cv2.polylines(ann, [left_cam], False, cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)
        cv2.polylines(ann, [right_cam], False, cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)

        cv2.line(ann, (w // 2, h), (w // 2, int(h * 0.6)),
                 cfg.VEHICLE_CENTER_RED, 2, cv2.LINE_AA)

        # Project the look-ahead point back to the camera frame
        la_pt_bev = np.array([[lane_cx_la, y_eval_la]], dtype=np.float32).reshape(-1, 1, 2)
        la_pt_cam = cv2.perspectiveTransform(la_pt_bev, Minv).reshape(-1, 2)[0]
        cv2.circle(ann, (int(la_pt_cam[0]), int(la_pt_cam[1])),
                   8, cfg.CENTER_BLUE, -1, cv2.LINE_AA)

        det.annotated = ann
        # Also fill in left/right_line for UI compatibility (project the
        # polynomial endpoints to the camera frame).
        det.left_line = (tuple(left_cam[-1]), tuple(left_cam[0]))
        det.right_line = (tuple(right_cam[-1]), tuple(right_cam[0]))
        return det

    def _fallback(self, img, det):
        h, w = img.shape[:2]
        det.center_offset = self._prev_offset
        det.lookahead_offset = self._prev_la
        det.confidence = 0.0
        det.annotated = img.copy()
        det.pred_mask = np.zeros((h, w), dtype=np.uint8)
        return det


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

DETECTORS = {
    "centroid": CentroidDetector,
    "hough": HoughDetector,
    "polyfit": PolyFitDetector,
}


def make_detector(name: str) -> LaneDetectorBase:
    name = name.lower()
    if name not in DETECTORS:
        raise ValueError(f"unknown detector: {name!r}; pick from {list(DETECTORS)}")
    return DETECTORS[name]()
