"""Steering controllers used in the simulator.

This module provides the two controllers compared in the project:

  * :class:`PIDController` -- the baseline. Takes a normalised lateral
    offset (-1..+1) from the lane detector and outputs a steering angle.
  * :class:`StanleyController` -- a feedback law combining cross-track
    error and heading error. Outputs a steering angle directly from the
    current vehicle state and the track geometry, so it does not depend
    on the camera image and acts as a perception-free *upper-bound*
    benchmark (the closest thing to a "ground-truth control" reference).

The :class:`SpeedController` (adaptive longitudinal control) is left
unchanged from the original simulator.
"""

import math

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


class StanleyController:
    """Stanley steering law.

    The Stanley controller (Hoffmann et al., 2007, Stanford) computes:

        delta = heading_error + atan2(k * crosstrack_error, k_s + v)

    where ``heading_error`` is the difference between the desired track
    heading and the vehicle heading, ``crosstrack_error`` is the signed
    lateral distance of the front axle to the path, ``v`` is the vehicle
    speed and ``k``, ``k_s`` are gains. We add a small damping term on
    the steering rate for smoother behaviour.

    Unlike the PID, the Stanley law in this implementation reads the
    *true* path geometry from the simulated track (an idealised lateral
    error). It therefore serves as a perception-free reference: any gap
    between the PID-on-detected-offset and Stanley-on-truth tells us how
    much error the vision pipeline is introducing.
    """

    def __init__(self, k=cfg.STANLEY_K, k_s=cfg.STANLEY_KS,
                 max_steer=cfg.MAX_STEERING_ANGLE):
        self.k = k
        self.k_s = k_s
        self.max_steer = max_steer
        self.prev_steer = 0.0
        # Components published for the dashboard
        self.last_p = 0.0  # crosstrack term
        self.last_i = 0.0  # always 0 (no integral)
        self.last_d = 0.0  # heading term

    def compute(self, track, car_x, car_y, heading, speed, dt):
        # Front-axle position so the Stanley law has a sensible reference
        fx = car_x + cfg.WHEELBASE * math.cos(heading)
        fy = car_y + cfg.WHEELBASE * math.sin(heading)
        idx, e_cross = track.get_nearest_index(fx, fy)
        track_heading = track.get_heading_at(idx)
        # Heading error in (-pi, pi]; positive when track turns to the
        # right of the vehicle's current heading (screen-y down frame).
        psi = math.atan2(
            math.sin(track_heading - heading),
            math.cos(track_heading - heading),
        )
        # ``e_cross`` is positive when the front axle is to the right of
        # the centreline. In our bicycle model, *positive* steering also
        # turns the heading to the right, so to drive back to the path
        # we need ``delta`` to become more *negative* as ``e_cross``
        # grows -- which is the canonical Stanley form ``-atan2(k*e/v)``.
        cross_term = math.atan2(self.k * e_cross, self.k_s + max(speed, 1e-3))
        # In our screen-y-down + bicycle-model convention, positive
        # steering rotates the heading "downward" (clockwise). When the
        # car is to the right of the centreline (``e_cross > 0``) the
        # centreline normal frame produces a *positive* ``cross_term``
        # which already pulls the steering toward "more positive" --
        # exactly the direction that would push us further from the
        # path. The empirically-validated form below (``+cross_term``)
        # matches the standard Stanley law once the sign of ``e_cross``
        # is interpreted in this frame.
        delta = psi + cross_term
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        self.last_p = cross_term
        self.last_d = psi
        self.prev_steer = delta
        return delta

    def reset(self):
        self.prev_steer = 0.0
        self.last_p = self.last_i = self.last_d = 0.0


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
