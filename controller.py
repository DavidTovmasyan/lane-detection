"""PID steering controller."""

import numpy as np
import config as cfg


class PIDController:
    def __init__(self, kp=cfg.KP, ki=cfg.KI, kd=cfg.KD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def compute(self, error, dt):
        """Compute steering command from normalized error (-1..+1).

        Returns steering angle in radians.
        """
        if dt <= 0:
            return 0.0

        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = float(np.clip(
            self.integral,
            -cfg.PID_INTEGRAL_LIMIT,
            cfg.PID_INTEGRAL_LIMIT,
        ))
        i = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        self.prev_error = error
        self.last_p = p
        self.last_i = i
        self.last_d = d

        output = p + i + d
        return float(np.clip(output, -cfg.MAX_STEERING_ANGLE, cfg.MAX_STEERING_ANGLE))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0


class SpeedController:
    """Curvature-based adaptive speed controller.

    Looks ahead on the track, finds the sharpest curve, and computes
    a target speed: fast on straights, slow before sharp turns.
    """

    def __init__(self, max_speed, min_speed=cfg.SPEED_MIN,
                 curvature_gain=cfg.SPEED_CURVATURE_GAIN,
                 look_ahead=cfg.SPEED_LOOK_AHEAD_DIST,
                 smoothing=cfg.SPEED_SMOOTHING):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.curvature_gain = curvature_gain
        self.look_ahead = look_ahead
        self.smoothing = smoothing
        self.current_speed = max_speed
        self.target_speed = max_speed

    def update(self, track, car_x, car_y, dt):
        """Compute the target speed and smoothly adjust current speed."""
        max_curv = track.get_max_curvature_ahead(car_x, car_y, self.look_ahead)

        # target = gain / curvature, clamped to [min, max]
        if max_curv > 1e-6:
            raw = self.curvature_gain / max_curv
            self.target_speed = float(np.clip(raw, self.min_speed, self.max_speed))
        else:
            self.target_speed = self.max_speed

        # Smooth: blend toward target (asymmetric -- brake faster than accelerate)
        if self.target_speed < self.current_speed:
            blend = min(1.0, self.smoothing * 3)  # brake 3x faster
        else:
            blend = self.smoothing
        self.current_speed += (self.target_speed - self.current_speed) * blend
        return self.current_speed

    def reset(self, max_speed=None):
        if max_speed is not None:
            self.max_speed = max_speed
        self.current_speed = self.max_speed
        self.target_speed = self.max_speed
