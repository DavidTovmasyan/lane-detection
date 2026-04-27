"""Classical computer vision lane detection pipeline."""

import numpy as np
import cv2
import config as cfg


class DetectionResult:
    __slots__ = (
        "left_line", "right_line", "center_offset",
        "confidence", "annotated", "edges",
    )

    def __init__(self):
        self.left_line = None      # ((x1,y1),(x2,y2)) or None
        self.right_line = None
        self.center_offset = 0.0   # pixels, positive = lane center right of image center
        self.confidence = 0.0
        self.annotated = None      # image with overlays
        self.edges = None          # Canny output


class LaneDetector:
    def __init__(self):
        self._prev_offset = 0.0
        self._prev_left = None
        self._prev_right = None

    def detect(self, img):
        """Run the full lane detection pipeline on an RGB image."""
        h, w = img.shape[:2]
        result = DetectionResult()

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Gaussian blur
        blurred = cv2.GaussianBlur(gray, cfg.GAUSSIAN_KERNEL, 0)

        # 3. Canny edge detection
        edges = cv2.Canny(blurred, cfg.CANNY_LOW, cfg.CANNY_HIGH)
        result.edges = edges

        # 4. Region of Interest mask (trapezoid)
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

        # 5. Hough Line Transform
        lines = cv2.HoughLinesP(
            masked,
            rho=cfg.HOUGH_RHO,
            theta=cfg.HOUGH_THETA,
            threshold=cfg.HOUGH_THRESHOLD,
            minLineLength=cfg.HOUGH_MIN_LINE_LEN,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP,
        )

        # 6. Classify and average lines
        left_segments = []
        right_segments = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < cfg.SLOPE_MIN:
                    continue
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                mid_x = (x1 + x2) / 2
                if slope < 0 and mid_x < w / 2:
                    left_segments.append((slope, x1, y1, x2, y2, length))
                elif slope > 0 and mid_x > w / 2:
                    right_segments.append((slope, x1, y1, x2, y2, length))

        # Weighted average of segments -> single line per side
        left_line = self._average_line(left_segments, h, roi_top)
        right_line = self._average_line(right_segments, h, roi_top)

        # Temporal smoothing
        if left_line is not None:
            self._prev_left = left_line
        elif self._prev_left is not None:
            left_line = self._prev_left

        if right_line is not None:
            self._prev_right = right_line
        elif self._prev_right is not None:
            right_line = self._prev_right

        result.left_line = left_line
        result.right_line = right_line

        # 7. Compute center offset
        offset = 0.0
        detected = 0
        if left_line is not None:
            detected += 1
        if right_line is not None:
            detected += 1

        if left_line is not None and right_line is not None:
            # x positions at the bottom of the image
            left_x_bot = left_line[0][0]
            right_x_bot = right_line[0][0]
            lane_center = (left_x_bot + right_x_bot) / 2
            offset = lane_center - w / 2
            result.confidence = 1.0
        elif left_line is not None:
            left_x_bot = left_line[0][0]
            estimated_center = left_x_bot + cfg.ROAD_WIDTH * 2
            offset = estimated_center - w / 2
            result.confidence = 0.5
        elif right_line is not None:
            right_x_bot = right_line[0][0]
            estimated_center = right_x_bot - cfg.ROAD_WIDTH * 2
            offset = estimated_center - w / 2
            result.confidence = 0.5
        else:
            offset = self._prev_offset
            result.confidence = 0.0

        # Smooth offset
        alpha = cfg.SMOOTHING_ALPHA
        smoothed = alpha * offset + (1 - alpha) * self._prev_offset
        self._prev_offset = smoothed
        result.center_offset = smoothed

        # 8. Annotate
        annotated = img.copy()
        # Draw ROI region (semi-transparent)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, vertices, (255, 200, 0))
        cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)

        if left_line is not None:
            cv2.line(annotated, left_line[0], left_line[1],
                     cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)
        if right_line is not None:
            cv2.line(annotated, right_line[0], right_line[1],
                     cfg.LANE_DETECT_GREEN, 3, cv2.LINE_AA)

        # Vehicle center reference line
        cx = w // 2
        cv2.line(annotated, (cx, h), (cx, int(h * 0.6)),
                 cfg.VEHICLE_CENTER_RED, 2, cv2.LINE_AA)

        # Detected lane center marker
        if left_line is not None and right_line is not None:
            lane_cx = int((left_line[0][0] + right_line[0][0]) / 2)
            cv2.line(annotated, (lane_cx, h), (lane_cx, int(h * 0.65)),
                     cfg.CENTER_BLUE, 2, cv2.LINE_AA)
            cv2.circle(annotated, (lane_cx, int(h * 0.75)), 6,
                       cfg.CENTER_BLUE, -1, cv2.LINE_AA)

        result.annotated = annotated
        return result

    def _average_line(self, segments, img_h, roi_top):
        """Weighted average of line segments, extrapolated to img bounds."""
        if not segments:
            return None
        total_weight = sum(s[5] for s in segments)
        if total_weight < 1e-6:
            return None
        avg_slope = sum(s[0] * s[5] for s in segments) / total_weight
        # Average intercept: y = slope * x + b  =>  b = y - slope * x
        avg_intercept = sum(
            ((s[2] - s[0] * s[1]) * s[5]) for s in segments
        ) / total_weight

        if abs(avg_slope) < 1e-6:
            return None

        # Extrapolate: x = (y - b) / slope
        y_bot = img_h - 1
        y_top = roi_top
        x_bot = int((y_bot - avg_intercept) / avg_slope)
        x_top = int((y_top - avg_intercept) / avg_slope)
        return ((x_bot, y_bot), (x_top, y_top))
