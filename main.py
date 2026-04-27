"""Lane Detection & Autonomous Steering Simulator - Entry Point."""

import argparse
import math
from collections import deque

import pygame

import config as cfg
from track import Track
from vehicle import Vehicle
from camera import FPVCamera
from vision import LaneDetector
from controller import PIDController, SpeedController
from ui import UI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lane Detection & Autonomous Steering Simulator",
    )
    track_names = list(cfg.TRACKS.keys())
    parser.add_argument(
        "--map",
        type=str,
        choices=track_names,
        default=None,
        metavar="NAME",
        help=f"Track to load: {', '.join(track_names)} (default: show selector)",
    )
    return parser.parse_args()


# ── Track selection screen ──────────────────────────────────────────

def track_select_screen(screen, clock):
    """Show a track selection menu. Returns the chosen track key or None to quit."""
    title_font = pygame.font.SysFont(cfg.FONT_SANS, 34, bold=True)
    sub_font = pygame.font.SysFont(cfg.FONT_SANS, 15)
    name_font = pygame.font.SysFont(cfg.FONT_SANS, 18, bold=True)
    diff_font = pygame.font.SysFont(cfg.FONT_MONO, 13)
    hint_font = pygame.font.SysFont(cfg.FONT_MONO, 13)

    track_keys = list(cfg.TRACKS.keys())
    n = len(track_keys)

    # Layout: cards in a row
    card_w, card_h = 220, 300
    gap = 25
    total_w = n * card_w + (n - 1) * gap
    start_x = (cfg.WINDOW_WIDTH - total_w) // 2
    start_y = 220

    # Pre-render track thumbnails
    thumbs = {}
    for key in track_keys:
        tdef = cfg.TRACKS[key]
        mini_track = Track(tdef["waypoints"], tdef["road_width"])
        thumbs[key] = _render_track_thumbnail(mini_track, card_w - 30, 160)

    hovered = None

    while True:
        clock.tick(30)
        mx, my = pygame.mouse.get_pos()
        hovered = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                # Number keys 1-N
                num = event.key - pygame.K_1
                if 0 <= num < n:
                    return track_keys[num]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, key in enumerate(track_keys):
                    cx = start_x + i * (card_w + gap)
                    rect = pygame.Rect(cx, start_y, card_w, card_h)
                    if rect.collidepoint(mx, my):
                        return key

        # ── Draw ────────────────────────────────────────────────
        screen.fill(cfg.BG_DARK)

        # Title
        t = title_font.render("SELECT TRACK", True, cfg.TEXT_ACCENT)
        screen.blit(t, (cfg.WINDOW_WIDTH // 2 - t.get_width() // 2, 60))

        sub = sub_font.render(
            f"Click a track or press 1-{n} to start  |  ESC to quit",
            True, cfg.TEXT_SECONDARY,
        )
        screen.blit(sub, (cfg.WINDOW_WIDTH // 2 - sub.get_width() // 2, 110))

        # Cards
        for i, key in enumerate(track_keys):
            tdef = cfg.TRACKS[key]
            cx = start_x + i * (card_w + gap)
            rect = pygame.Rect(cx, start_y, card_w, card_h)

            is_hover = rect.collidepoint(mx, my)
            if is_hover:
                hovered = key

            # Card background
            bg = cfg.BG_CARD if not is_hover else (45, 52, 65)
            pygame.draw.rect(screen, bg, rect, border_radius=8)
            border_color = cfg.TEXT_ACCENT if is_hover else cfg.SEPARATOR
            pygame.draw.rect(screen, border_color, rect, width=2, border_radius=8)

            # Track number
            num_surf = diff_font.render(f"{i + 1}", True, cfg.TEXT_SECONDARY)
            screen.blit(num_surf, (cx + 10, start_y + 8))

            # Track name
            nm = name_font.render(tdef["description"], True, cfg.TEXT_PRIMARY)
            screen.blit(nm, (cx + card_w // 2 - nm.get_width() // 2, start_y + 8))

            # Difficulty stars
            diff = tdef["difficulty"]
            stars = "+" * diff + "-" * (5 - diff)
            diff_color = (
                cfg.GAUGE_FILL_NORMAL if diff <= 2
                else cfg.GAUGE_FILL_WARN if diff <= 3
                else cfg.GAUGE_FILL_DANGER
            )
            ds = diff_font.render(f"Difficulty: {stars}", True, diff_color)
            screen.blit(ds, (cx + card_w // 2 - ds.get_width() // 2, start_y + 32))

            # Thumbnail
            thumb = thumbs[key]
            tx = cx + (card_w - thumb.get_width()) // 2
            ty = start_y + 55
            screen.blit(thumb, (tx, ty))

            # Speed info
            spd = diff_font.render(
                f"Speed: {tdef['speed']:.0f} px/s | Width: {tdef['road_width']:.0f}",
                True, cfg.TEXT_SECONDARY,
            )
            screen.blit(spd, (cx + card_w // 2 - spd.get_width() // 2, start_y + card_h - 40))

            # Waypoint count
            wpc = diff_font.render(
                f"{len(tdef['waypoints'])} waypoints",
                True, cfg.TEXT_SECONDARY,
            )
            screen.blit(wpc, (cx + card_w // 2 - wpc.get_width() // 2, start_y + card_h - 22))

        # Footer hint
        foot = hint_font.render(
            "M = return to this menu during simulation",
            True, cfg.TEXT_SECONDARY,
        )
        screen.blit(foot, (cfg.WINDOW_WIDTH // 2 - foot.get_width() // 2, cfg.WINDOW_HEIGHT - 50))

        pygame.display.flip()


def _render_track_thumbnail(track, width, height):
    """Render a small track preview onto a Pygame surface."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    bx0, by0, bx1, by1 = track.bounds
    bw = max(bx1 - bx0, 1)
    bh = max(by1 - by0, 1)
    margin = 8
    sx = (width - 2 * margin) / bw
    sy = (height - 2 * margin) / bh
    scale = min(sx, sy)
    ox = margin + (width - 2 * margin - bw * scale) / 2
    oy = margin + (height - 2 * margin - bh * scale) / 2

    def w2s(wx, wy):
        return (int(ox + (wx - bx0) * scale),
                int(oy + (wy - by0) * scale))

    # Road surface
    cl = track.centerline
    road_w = max(2, int(track.road_width * scale))
    step = max(1, len(cl) // 80)
    for i in range(0, len(cl) - step, step):
        p1 = w2s(cl[i, 0], cl[i, 1])
        p2 = w2s(cl[min(i + step, len(cl) - 1), 0],
                  cl[min(i + step, len(cl) - 1), 1])
        pygame.draw.line(surf, (55, 58, 64), p1, p2, road_w)

    # Lane boundaries
    lb = track.left_boundary
    rb = track.right_boundary
    step_b = max(1, len(lb) // 60)
    left_pts = [w2s(lb[i, 0], lb[i, 1]) for i in range(0, len(lb), step_b)]
    right_pts = [w2s(rb[i, 0], rb[i, 1]) for i in range(0, len(rb), step_b)]
    if len(left_pts) > 1:
        pygame.draw.lines(surf, (180, 185, 190), True, left_pts, 1)
    if len(right_pts) > 1:
        pygame.draw.lines(surf, (180, 185, 190), True, right_pts, 1)

    # Start marker
    sp = w2s(cl[0, 0], cl[0, 1])
    pygame.draw.circle(surf, cfg.GAUGE_FILL_NORMAL, sp, 4)

    return surf


# ── Simulation loop ─────────────────────────────────────────────────

def run_simulation(screen, clock, track_key):
    """Run the simulation for a given track. Returns 'menu' or 'quit'."""
    tdef = cfg.TRACKS[track_key]
    track = Track(tdef["waypoints"], tdef["road_width"])
    vehicle = Vehicle(track)
    vehicle.speed = tdef["speed"]
    camera = FPVCamera()
    detector = LaneDetector()
    pid = PIDController()
    use_adaptive = tdef.get("adaptive_speed", False)
    speed_ctrl = SpeedController(tdef["speed"]) if use_adaptive else None
    ui = UI()
    ui.track_name = tdef["description"]
    trail = deque(maxlen=cfg.TRAIL_MAX_LENGTH)

    paused = False
    fps_ema = 60.0

    while True:
        dt = clock.tick(cfg.FPS_TARGET) / 1000.0
        dt = min(dt, 0.05)

        if dt > 0:
            fps_ema = fps_ema * 0.95 + (1.0 / dt) * 0.05

        # ── Events ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_m:
                    return "menu"
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    vehicle = Vehicle(track)
                    vehicle.speed = tdef["speed"]
                    pid.reset()
                    detector = LaneDetector()
                    if speed_ctrl:
                        speed_ctrl.reset(tdef["speed"])
                    trail.clear()
                    ui.offset_history.clear()
                elif event.key == pygame.K_UP:
                    if speed_ctrl:
                        speed_ctrl.max_speed = min(speed_ctrl.max_speed + 20, 300)
                    else:
                        vehicle.speed = min(vehicle.speed + 20, 300)
                elif event.key == pygame.K_DOWN:
                    if speed_ctrl:
                        speed_ctrl.max_speed = max(speed_ctrl.max_speed - 20, 40)
                    else:
                        vehicle.speed = max(vehicle.speed - 20, 40)

        if paused:
            metrics = _build_metrics(vehicle, pid, detector, fps_ema, speed_ctrl)
            ui.draw(screen, track, vehicle, trail, None, metrics)
            pause_font = pygame.font.SysFont(cfg.FONT_MONO, 36, bold=True)
            pause_surf = pause_font.render("PAUSED", True, cfg.GAUGE_FILL_WARN)
            pw = pause_surf.get_width()
            screen.blit(pause_surf, (cfg.WINDOW_WIDTH // 2 - pw // 2, 15))
            pygame.display.flip()
            continue

        # ── Sim step ────────────────────────────────────────────
        # Adaptive speed
        if speed_ctrl:
            vehicle.speed = speed_ctrl.update(track, vehicle.x, vehicle.y, dt)

        fpv_image = camera.render(vehicle.x, vehicle.y, vehicle.heading, track)
        detection = detector.detect(fpv_image)

        normalized_offset = detection.center_offset / (cfg.CAMERA_IMG_W / 2)
        steering_cmd = pid.compute(normalized_offset, dt)

        vehicle.update(dt, steering_cmd)
        trail.append((vehicle.x, vehicle.y))

        metrics = _build_metrics(vehicle, pid, detection, fps_ema, speed_ctrl)
        ui.draw(screen, track, vehicle, trail, detection.annotated, metrics)
        pygame.display.flip()


def _build_metrics(vehicle, pid, detection_or_detector, fps, speed_ctrl=None):
    conf = 0.0
    offset = 0.0
    if hasattr(detection_or_detector, "confidence"):
        conf = detection_or_detector.confidence
        offset = detection_or_detector.center_offset
    return {
        "speed": vehicle.speed,
        "max_speed": speed_ctrl.max_speed if speed_ctrl else vehicle.speed,
        "target_speed": speed_ctrl.target_speed if speed_ctrl else vehicle.speed,
        "adaptive": speed_ctrl is not None,
        "steering": vehicle.steering,
        "offset": offset,
        "confidence": conf,
        "fps": fps,
        "pid_p": pid.last_p,
        "pid_i": pid.last_i,
        "pid_d": pid.last_d,
    }


# ── Entry point ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption(cfg.WINDOW_TITLE)
    clock = pygame.time.Clock()

    if args.map:
        # Direct launch: skip selector
        result = run_simulation(screen, clock, args.map)
        if result == "menu":
            # Fall through to selector loop
            pass
        else:
            pygame.quit()
            return

    # Selection loop
    while True:
        chosen = track_select_screen(screen, clock)
        if chosen is None:
            break
        result = run_simulation(screen, clock, chosen)
        if result == "quit":
            break
        # result == "menu": loop back to selector

    pygame.quit()


if __name__ == "__main__":
    main()
