"""Bicycle kinematic vehicle model."""

import math
import numpy as np
import config as cfg


class Vehicle:
    def __init__(self, track):
        # Initialize at first centerline point, heading aligned with track
        idx = 0
        self.x = float(track.centerline[idx, 0])
        self.y = float(track.centerline[idx, 1])
        self.heading = track.get_heading_at(idx)
        self.speed = cfg.VEHICLE_SPEED
        self.steering = 0.0
        self.wheelbase = cfg.WHEELBASE

    def update(self, dt, steering_command):
        """Advance vehicle state using bicycle kinematic model."""
        self.steering = float(np.clip(
            steering_command,
            -cfg.MAX_STEERING_ANGLE,
            cfg.MAX_STEERING_ANGLE,
        ))
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        if abs(self.steering) > 1e-6:
            self.heading += (self.speed / self.wheelbase) * math.tan(self.steering) * dt
        # Normalize heading to [-pi, pi]
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "speed": self.speed,
            "steering": self.steering,
        }
