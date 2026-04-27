"""Track definition using Catmull-Rom spline interpolation."""

import numpy as np
import config as cfg


class Track:
    def __init__(self, waypoints=None, road_width=None):
        if waypoints is None:
            track_def = cfg.TRACKS[cfg.DEFAULT_TRACK]
            waypoints = track_def["waypoints"]
            road_width = track_def["road_width"]
        if road_width is None:
            road_width = cfg.DEFAULT_ROAD_WIDTH
        self.road_width = road_width
        wp = np.array(waypoints, dtype=np.float64)
        self.waypoints = wp
        self.centerline = self._interpolate(wp)
        self.tangents = self._compute_tangents(self.centerline)
        self.normals = self._compute_normals(self.tangents)
        # Normal points in screen-down direction (right of travel in screen coords)
        # So left boundary is centerline MINUS normal, right is PLUS
        self.left_boundary = self.centerline - self.normals * (self.road_width / 2)
        self.right_boundary = self.centerline + self.normals * (self.road_width / 2)
        self.center_dashes = self._compute_dashes(self.centerline)
        self.curvature = self._compute_curvature(self.centerline)
        self._bounds = self._compute_bounds()

    # ── Catmull-Rom spline ──────────────────────────────────────────

    def _interpolate(self, wp):
        """Catmull-Rom spline through closed-loop waypoints."""
        n = len(wp)
        pts = []
        samples = cfg.TRACK_SAMPLES_PER_SEGMENT
        for i in range(n):
            p0 = wp[(i - 1) % n]
            p1 = wp[i]
            p2 = wp[(i + 1) % n]
            p3 = wp[(i + 2) % n]
            for t_idx in range(samples):
                t = t_idx / samples
                tt = t * t
                ttt = tt * t
                q = 0.5 * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * ttt
                )
                pts.append(q)
        return np.array(pts, dtype=np.float64)

    def _compute_tangents(self, cl):
        tangents = np.zeros_like(cl)
        tangents[1:-1] = cl[2:] - cl[:-2]
        tangents[0] = cl[1] - cl[-1]
        tangents[-1] = cl[0] - cl[-2]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return tangents / norms

    def _compute_normals(self, tangents):
        # Rotate tangent 90 degrees CCW: (tx, ty) -> (-ty, tx)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        return normals

    def _compute_curvature(self, cl):
        """Compute curvature at each centerline point.

        Uses the cross-product formula on finite differences:
            kappa = |dx * d2y - dy * d2x| / (dx^2 + dy^2)^1.5
        Result is smoothed with a rolling window for stability.
        """
        n = len(cl)
        # First derivatives (central differences, wrapping)
        dx = np.zeros(n)
        dy = np.zeros(n)
        dx[1:-1] = cl[2:, 0] - cl[:-2, 0]
        dy[1:-1] = cl[2:, 1] - cl[:-2, 1]
        dx[0] = cl[1, 0] - cl[-1, 0]
        dy[0] = cl[1, 1] - cl[-1, 1]
        dx[-1] = cl[0, 0] - cl[-2, 0]
        dy[-1] = cl[0, 1] - cl[-2, 1]
        # Second derivatives
        d2x = np.zeros(n)
        d2y = np.zeros(n)
        d2x[1:-1] = cl[2:, 0] - 2 * cl[1:-1, 0] + cl[:-2, 0]
        d2y[1:-1] = cl[2:, 1] - 2 * cl[1:-1, 1] + cl[:-2, 1]
        d2x[0] = cl[1, 0] - 2 * cl[0, 0] + cl[-1, 0]
        d2y[0] = cl[1, 1] - 2 * cl[0, 1] + cl[-1, 1]
        d2x[-1] = cl[0, 0] - 2 * cl[-1, 0] + cl[-2, 0]
        d2y[-1] = cl[0, 1] - 2 * cl[-1, 1] + cl[-2, 1]

        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.maximum(denom, 1e-8)
        kappa = np.abs(dx * d2y - dy * d2x) / denom

        # Smooth with a rolling average (window ~15 points)
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        kappa_smooth = np.convolve(kappa, kernel, mode="same")
        return kappa_smooth

    def get_max_curvature_ahead(self, car_x, car_y, look_ahead):
        """Return the max curvature in the road ahead of the car."""
        idx, _ = self.get_nearest_index(car_x, car_y)
        n = len(self.centerline)
        count = 0
        cum = 0.0
        for i in range(1, n):
            j = (idx + i) % n
            j_prev = (idx + i - 1) % n
            cum += np.linalg.norm(self.centerline[j] - self.centerline[j_prev])
            count += 1
            if cum >= look_ahead:
                break
        indices = [(idx + i) % n for i in range(count + 1)]
        if not indices:
            return 0.0
        return float(np.max(self.curvature[indices]))

    def _compute_dashes(self, cl):
        """Return list of (start_idx, end_idx) for dash-on segments."""
        dashes = []
        cum_dist = 0.0
        dash_on = True
        seg_start = 0
        threshold = cfg.LANE_DASH_LENGTH if dash_on else cfg.LANE_DASH_GAP
        for i in range(1, len(cl)):
            d = np.linalg.norm(cl[i] - cl[i - 1])
            cum_dist += d
            if cum_dist >= threshold:
                if dash_on:
                    dashes.append((seg_start, i))
                dash_on = not dash_on
                threshold = cfg.LANE_DASH_LENGTH if dash_on else cfg.LANE_DASH_GAP
                cum_dist = 0.0
                seg_start = i
        return dashes

    def _compute_bounds(self):
        all_pts = np.vstack([self.left_boundary, self.right_boundary])
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        return (min_xy[0], min_xy[1], max_xy[0], max_xy[1])

    @property
    def bounds(self):
        return self._bounds

    # ── Queries ─────────────────────────────────────────────────────

    def get_nearest_index(self, x, y):
        """Return (index, signed_offset) of nearest centerline point."""
        pos = np.array([x, y])
        diffs = self.centerline - pos
        dists_sq = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(dists_sq))
        # Signed offset: positive = right of centerline
        to_pos = pos - self.centerline[idx]
        # right direction is -normal (since normal points left)
        signed = -np.dot(to_pos, self.normals[idx])
        return idx, signed

    def get_heading_at(self, idx):
        """Return heading angle (radians) at centerline index."""
        t = self.tangents[idx % len(self.tangents)]
        return float(np.arctan2(t[1], t[0]))

    def get_road_ahead(self, car_x, car_y, look_ahead):
        """Return arrays of left, right, center points ahead of car."""
        idx, _ = self.get_nearest_index(car_x, car_y)
        n = len(self.centerline)

        # Walk forward until cumulative distance exceeds look_ahead
        count = 0
        cum = 0.0
        for i in range(1, n):
            j = (idx + i) % n
            j_prev = (idx + i - 1) % n
            cum += np.linalg.norm(self.centerline[j] - self.centerline[j_prev])
            count += 1
            if cum >= look_ahead:
                break

        indices = [(idx + i) % n for i in range(count + 1)]
        left_pts = self.left_boundary[indices]
        right_pts = self.right_boundary[indices]
        center_pts = self.centerline[indices]
        return left_pts, right_pts, center_pts

    def get_dash_segments_ahead(self, car_x, car_y, look_ahead):
        """Return list of (pt_start, pt_end) arrays for dashes ahead."""
        idx, _ = self.get_nearest_index(car_x, car_y)
        n = len(self.centerline)
        # Determine the range of indices ahead
        count = 0
        cum = 0.0
        for i in range(1, n):
            j = (idx + i) % n
            j_prev = (idx + i - 1) % n
            cum += np.linalg.norm(self.centerline[j] - self.centerline[j_prev])
            count += 1
            if cum >= look_ahead:
                break
        ahead_max = idx + count

        segments = []
        for s, e in self.center_dashes:
            # Check if dash overlaps with ahead range (handling wrapping)
            if s >= idx and s <= ahead_max:
                seg_s = self.centerline[s % n]
                seg_e = self.centerline[min(e, ahead_max) % n]
                segments.append((seg_s, seg_e))
            elif s < idx and s + n <= ahead_max:
                seg_s = self.centerline[s % n]
                seg_e = self.centerline[min(e, ahead_max - n + s) % n]
                segments.append((seg_s, seg_e))
        return segments
