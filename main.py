"""Lane Detection & Autonomous Steering Simulator - Entry Point."""

import math
from collections import deque
import pygame
import config as cfg
from track import Track
from vehicle import Vehicle
from camera import FPVCamera
from vision import LaneDetector
from controller import PIDController
from ui import UI


def main():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption(cfg.WINDOW_TITLE)
    clock = pygame.time.Clock()

    # Create simulation objects
    track = Track()
    vehicle = Vehicle(track)
    camera = FPVCamera()
    detector = LaneDetector()
    pid = PIDController()
    ui = UI()
    trail = deque(maxlen=cfg.TRAIL_MAX_LENGTH)

    running = True
    paused = False
    fps_ema = 60.0

    while running:
        dt = clock.tick(cfg.FPS_TARGET) / 1000.0
        dt = min(dt, 0.05)  # Cap dt to prevent physics explosion

        # Smooth FPS
        if dt > 0:
            fps_ema = fps_ema * 0.95 + (1.0 / dt) * 0.05

        # ── Event handling ──────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset
                    vehicle = Vehicle(track)
                    pid.reset()
                    trail.clear()
                elif event.key == pygame.K_UP:
                    vehicle.speed = min(vehicle.speed + 20, 300)
                elif event.key == pygame.K_DOWN:
                    vehicle.speed = max(vehicle.speed - 20, 40)

        if paused:
            # Still render but don't update sim
            metrics = _build_metrics(vehicle, pid, detector, fps_ema)
            ui.draw(screen, track, vehicle, trail, None, metrics)
            # Draw pause indicator
            pause_font = pygame.font.SysFont(cfg.FONT_MONO, 36, bold=True)
            pause_surf = pause_font.render("PAUSED", True, cfg.GAUGE_FILL_WARN)
            pw = pause_surf.get_width()
            screen.blit(pause_surf, (cfg.WINDOW_WIDTH // 2 - pw // 2, 15))
            pygame.display.flip()
            continue

        # ── Simulation step ─────────────────────────────────────

        # 1. Render FPV image
        fpv_image = camera.render(vehicle.x, vehicle.y, vehicle.heading, track)

        # 2. Lane detection
        detection = detector.detect(fpv_image)

        # 3. PID steering
        normalized_offset = detection.center_offset / (cfg.CAMERA_IMG_W / 2)
        steering_cmd = pid.compute(normalized_offset, dt)

        # 4. Update vehicle
        vehicle.update(dt, steering_cmd)
        trail.append((vehicle.x, vehicle.y))

        # 5. Collect metrics
        metrics = _build_metrics(vehicle, pid, detection, fps_ema)

        # 6. Render UI
        ui.draw(screen, track, vehicle, trail, detection.annotated, metrics)
        pygame.display.flip()

    pygame.quit()


def _build_metrics(vehicle, pid, detection_or_detector, fps):
    conf = 0.0
    offset = 0.0
    if hasattr(detection_or_detector, "confidence"):
        conf = detection_or_detector.confidence
        offset = detection_or_detector.center_offset
    return {
        "speed": vehicle.speed,
        "steering": vehicle.steering,
        "offset": offset,
        "confidence": conf,
        "fps": fps,
        "pid_p": pid.last_p,
        "pid_i": pid.last_i,
        "pid_d": pid.last_d,
    }


if __name__ == "__main__":
    main()
