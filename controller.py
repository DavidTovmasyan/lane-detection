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
