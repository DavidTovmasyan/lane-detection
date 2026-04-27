"""FPV camera renderer using perspective projection."""

import math
import numpy as np
import cv2
import config as cfg


class FPVCamera:
    def __init__(self):
        self.w = cfg.CAMERA_IMG_W
        self.h = cfg.CAMERA_IMG_H
        fov_rad = math.radians(cfg.CAMERA_FOV_DEG)
        self.focal = (self.w / 2) / math.tan(fov_rad / 2)
        self.cx = self.w / 2
        self.cy = self.h / 2
        self.cam_h = cfg.CAMERA_HEIGHT
        self.pitch = math.radians(cfg.CAMERA_PITCH_DEG)
        self.look_ahead = cfg.CAMERA_LOOK_AHEAD

        # Pre-compute sky gradient
        self._sky = self._make_sky()

    def _make_sky(self):
        sky = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for row in range(self.h):
            t = row / self.h
            r = int(cfg.SKY_TOP[0] * (1 - t) + cfg.SKY_BOTTOM[0] * t)
            g = int(cfg.SKY_TOP[1] * (1 - t) + cfg.SKY_BOTTOM[1] * t)
            b = int(cfg.SKY_TOP[2] * (1 - t) + cfg.SKY_BOTTOM[2] * t)
            sky[row, :] = (r, g, b)
        return sky

    def _project(self, world_xy, car_x, car_y, heading):
        """Project world ground-plane points to image coordinates.

        Returns (uv, visible) where uv is (N,2) and visible is (N,) bool.
        """
        n = len(world_xy)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

        # Translate to car-relative
        dx = world_xy[:, 0] - car_x
        dy = world_xy[:, 1] - car_y

        # Rotate so car-forward (+heading direction) aligns with +Z
        c = math.cos(heading)
        s = math.sin(heading)
        forward = dx * c + dy * s       # distance ahead of car
        lateral = -dx * s + dy * c      # distance to the right of car

        # Camera frame before pitch: +X=right, +Y=down, +Z=forward
        # Road is cam_h below camera → positive Y (down direction)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        # Pitch rotation around X-axis (positive pitch = look down)
        x_cam = lateral
        y_cam = self.cam_h * cp - forward * sp
        z = self.cam_h * sp + forward * cp

        # Visibility: must be in front of camera
        visible = z > 2.0

        # Perspective projection
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * (x_cam / z) + self.cx
            v = self.focal * (y_cam / z) + self.cy

        # Clamp invisible points
        u = np.where(visible, u, -1)
        v = np.where(visible, v, -1)

        # In-frame check
        in_frame = visible & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)

        return np.column_stack([u, v]), in_frame

    def render(self, car_x, car_y, heading, track):
        """Render the FPV camera image. Returns an RGB numpy array."""
        img = self._sky.copy()

        # Paint grass below horizon
        # Horizon: as d→∞, v → -focal*tan(pitch) + cy
        sp = math.sin(self.pitch)
        cp = math.cos(self.pitch)
        horizon_v = max(0, int(-self.focal * sp / cp + self.cy))
        horizon_v = min(horizon_v, self.h - 1)
        img[horizon_v:, :] = cfg.GRASS_COLOR

        # Get road geometry ahead
        left_pts, right_pts, center_pts = track.get_road_ahead(
            car_x, car_y, self.look_ahead
        )

        if len(left_pts) < 2 or len(right_pts) < 2:
            return img

        # Project boundaries
        left_uv, left_vis = self._project(left_pts, car_x, car_y, heading)
        right_uv, right_vis = self._project(right_pts, car_x, car_y, heading)
        center_uv, center_vis = self._project(center_pts, car_x, car_y, heading)

        # Build road surface polygon
        left_valid = left_uv[left_vis].astype(np.int32)
        right_valid = right_uv[right_vis].astype(np.int32)

        if len(left_valid) > 1 and len(right_valid) > 1:
            road_poly = np.vstack([left_valid, right_valid[::-1]])
            cv2.fillPoly(img, [road_poly], cfg.ROAD_SURFACE)

        # Draw lane markings - left boundary (solid white)
        if len(left_valid) > 1:
            cv2.polylines(img, [left_valid], False, cfg.LANE_WHITE, 3, cv2.LINE_AA)

        # Right boundary (solid white)
        if len(right_valid) > 1:
            cv2.polylines(img, [right_valid], False, cfg.LANE_WHITE, 3, cv2.LINE_AA)

        # Center dashes (yellow)
        center_valid_mask = center_vis
        center_screen = center_uv.astype(np.int32)
        # Draw dashed center line
        dash_on = True
        dash_len_accum = 0.0
        threshold = cfg.LANE_DASH_LENGTH
        for i in range(1, len(center_screen)):
            if not (center_valid_mask[i] and center_valid_mask[i - 1]):
                continue
            p1 = center_screen[i - 1]
            p2 = center_screen[i]
            seg_len = np.linalg.norm(p2 - p1)
            dash_len_accum += seg_len
            if dash_on:
                cv2.line(img, tuple(p1), tuple(p2), cfg.LANE_DASH_COLOR, 2, cv2.LINE_AA)
            if dash_len_accum >= threshold:
                dash_on = not dash_on
                threshold = cfg.LANE_DASH_LENGTH if dash_on else cfg.LANE_DASH_GAP
                dash_len_accum = 0.0

        return img
