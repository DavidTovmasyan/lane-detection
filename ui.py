"""3-panel Pygame UI: Map, FPV Camera, and Dashboard."""

import math
from collections import deque
import numpy as np
import pygame
import cv2
import config as cfg


class UI:
    def __init__(self):
        self.fonts = {}
        self._init_fonts()
        self.offset_history = deque(maxlen=200)
        self._display_speed = 0.0
        self._display_steering = 0.0

    def _init_fonts(self):
        mono = cfg.FONT_MONO
        sans = cfg.FONT_SANS
        self.fonts = {
            "title": pygame.font.SysFont(sans, 16, bold=True),
            "value": pygame.font.SysFont(mono, 28, bold=True),
            "label": pygame.font.SysFont(sans, 13),
            "unit": pygame.font.SysFont(sans, 11),
            "pid": pygame.font.SysFont(mono, 14),
            "fps": pygame.font.SysFont(mono, 13),
            "hud": pygame.font.SysFont(mono, 12, bold=True),
        }

    # ── Main draw ───────────────────────────────────────────────────

    def draw(self, screen, track, vehicle, trail, annotated_img, metrics):
        screen.fill(cfg.BG_DARK)
        self._draw_map(screen, track, vehicle, trail)
        self._draw_fpv(screen, annotated_img, metrics)
        self._draw_dashboard(screen, metrics)
        # Separators
        pygame.draw.line(screen, cfg.SEPARATOR, (cfg.FPV_X, 0), (cfg.FPV_X, cfg.WINDOW_HEIGHT), 2)
        pygame.draw.line(screen, cfg.SEPARATOR, (0, cfg.DASH_Y), (cfg.MAP_W, cfg.DASH_Y), 2)

    # ── Map Panel ───────────────────────────────────────────────────

    def _draw_map(self, screen, track, vehicle, trail):
        rect = pygame.Rect(cfg.MAP_X, cfg.MAP_Y, cfg.MAP_W, cfg.MAP_H)
        pygame.draw.rect(screen, cfg.BG_PANEL, rect)

        # Title
        title_surf = self.fonts["title"].render("MAP VIEW", True, cfg.TEXT_ACCENT)
        screen.blit(title_surf, (rect.x + 12, rect.y + 8))
        pygame.draw.line(screen, cfg.SEPARATOR,
                         (rect.x + 8, rect.y + 28),
                         (rect.x + rect.w - 8, rect.y + 28), 1)

        # Drawable area
        draw_rect = pygame.Rect(
            rect.x + cfg.MAP_PADDING,
            rect.y + cfg.MAP_TITLE_HEIGHT + 5,
            rect.w - 2 * cfg.MAP_PADDING,
            rect.h - cfg.MAP_TITLE_HEIGHT - cfg.MAP_PADDING - 5,
        )

        bx0, by0, bx1, by1 = track.bounds
        bw = bx1 - bx0
        bh = by1 - by0
        if bw < 1 or bh < 1:
            return
        margin = 15
        sx = (draw_rect.w - 2 * margin) / bw
        sy = (draw_rect.h - 2 * margin) / bh
        scale = min(sx, sy)
        ox = draw_rect.x + margin + (draw_rect.w - 2 * margin - bw * scale) / 2
        oy = draw_rect.y + margin + (draw_rect.h - 2 * margin - bh * scale) / 2

        def w2s(wx, wy):
            return (int(ox + (wx - bx0) * scale),
                    int(oy + (wy - by0) * scale))

        # Road surface - draw thick lines between centerline points
        cl = track.centerline
        road_w = max(2, int(cfg.ROAD_WIDTH * scale))
        for i in range(0, len(cl) - 1, 3):
            p1 = w2s(cl[i, 0], cl[i, 1])
            p2 = w2s(cl[min(i + 3, len(cl) - 1), 0], cl[min(i + 3, len(cl) - 1), 1])
            pygame.draw.line(screen, cfg.ROAD_SURFACE, p1, p2, road_w)

        # Lane boundaries
        lb = track.left_boundary
        rb = track.right_boundary
        left_screen = [w2s(lb[i, 0], lb[i, 1]) for i in range(0, len(lb), 4)]
        right_screen = [w2s(rb[i, 0], rb[i, 1]) for i in range(0, len(rb), 4)]
        if len(left_screen) > 1:
            pygame.draw.lines(screen, cfg.LANE_WHITE, True, left_screen, 1)
        if len(right_screen) > 1:
            pygame.draw.lines(screen, cfg.LANE_WHITE, True, right_screen, 1)

        # Center dashes
        for s_idx, e_idx in track.center_dashes:
            if s_idx < len(cl) and e_idx < len(cl):
                p1 = w2s(cl[s_idx, 0], cl[s_idx, 1])
                p2 = w2s(cl[e_idx, 0], cl[e_idx, 1])
                pygame.draw.line(screen, cfg.LANE_DASH_COLOR, p1, p2, 1)

        # Trail
        if len(trail) > 1:
            trail_surf = pygame.Surface((draw_rect.w, draw_rect.h), pygame.SRCALPHA)
            trail_list = list(trail)
            for i in range(1, len(trail_list)):
                alpha = int(180 * i / len(trail_list))
                color = (*cfg.TRAIL_COLOR, alpha)
                p1_w = trail_list[i - 1]
                p2_w = trail_list[i]
                p1 = (int((p1_w[0] - bx0) * scale + margin + (draw_rect.w - 2 * margin - bw * scale) / 2),
                       int((p1_w[1] - by0) * scale + margin + (draw_rect.h - 2 * margin - bh * scale) / 2))
                p2 = (int((p2_w[0] - bx0) * scale + margin + (draw_rect.w - 2 * margin - bw * scale) / 2),
                       int((p2_w[1] - by0) * scale + margin + (draw_rect.h - 2 * margin - bh * scale) / 2))
                pygame.draw.line(trail_surf, color, p1, p2, 2)
            screen.blit(trail_surf, (draw_rect.x, draw_rect.y))

        # Car marker (triangle)
        cx, cy = w2s(vehicle.x, vehicle.y)
        h = vehicle.heading
        size = max(6, int(14 * scale))
        tip = (cx + int(size * math.cos(h)), cy + int(size * math.sin(h)))
        rear_l = (cx + int(size * 0.6 * math.cos(h + 2.5)),
                  cy + int(size * 0.6 * math.sin(h + 2.5)))
        rear_r = (cx + int(size * 0.6 * math.cos(h - 2.5)),
                  cy + int(size * 0.6 * math.sin(h - 2.5)))
        pygame.draw.polygon(screen, cfg.CAR_BODY, [tip, rear_l, rear_r])
        pygame.draw.polygon(screen, cfg.CAR_OUTLINE, [tip, rear_l, rear_r], 1)

    # ── FPV Panel ───────────────────────────────────────────────────

    def _draw_fpv(self, screen, annotated_img, metrics):
        rect = pygame.Rect(cfg.FPV_X, cfg.FPV_Y, cfg.FPV_W, cfg.FPV_H)
        pygame.draw.rect(screen, cfg.BG_PANEL, rect)

        if annotated_img is not None:
            # Convert RGB numpy array to Pygame surface
            # annotated_img is (H, W, 3) RGB
            surf = pygame.surfarray.make_surface(
                np.transpose(annotated_img, (1, 0, 2))
            )
            scaled = pygame.transform.scale(surf, (rect.w, rect.h))
            screen.blit(scaled, (rect.x, rect.y))

        # HUD overlay
        hud_surf = pygame.Surface((rect.w, 40), pygame.SRCALPHA)
        hud_surf.fill((0, 0, 0, 100))
        screen.blit(hud_surf, (rect.x, rect.y))

        label = self.fonts["hud"].render("LANE DETECTION ACTIVE", True, cfg.LANE_DETECT_GREEN)
        screen.blit(label, (rect.x + 12, rect.y + 12))

        # Confidence indicator
        conf = metrics.get("confidence", 0)
        conf_text = f"CONF: {conf:.0%}"
        conf_color = cfg.GAUGE_FILL_NORMAL if conf > 0.7 else (
            cfg.GAUGE_FILL_WARN if conf > 0.3 else cfg.GAUGE_FILL_DANGER
        )
        conf_surf = self.fonts["hud"].render(conf_text, True, conf_color)
        screen.blit(conf_surf, (rect.x + rect.w - 120, rect.y + 12))

        # FPV label
        fpv_label = self.fonts["title"].render("FPV CAMERA", True, cfg.TEXT_ACCENT)
        screen.blit(fpv_label, (rect.x + rect.w - 120, rect.y + rect.h - 25))

    # ── Dashboard Panel ─────────────────────────────────────────────

    def _draw_dashboard(self, screen, metrics):
        rect = pygame.Rect(cfg.DASH_X, cfg.DASH_Y, cfg.DASH_W, cfg.DASH_H)
        pygame.draw.rect(screen, cfg.BG_PANEL, rect)

        # Title
        title_surf = self.fonts["title"].render("DASHBOARD", True, cfg.TEXT_ACCENT)
        screen.blit(title_surf, (rect.x + 12, rect.y + 8))
        pygame.draw.line(screen, cfg.SEPARATOR,
                         (rect.x + 8, rect.y + 28),
                         (rect.x + rect.w - 8, rect.y + 28), 1)

        # Track offset history
        offset = metrics.get("offset", 0)
        self.offset_history.append(offset)

        # Smooth display values
        self._display_speed += (metrics.get("speed", 0) - self._display_speed) * 0.15
        self._display_steering += (
            math.degrees(metrics.get("steering", 0)) - self._display_steering
        ) * 0.15

        y_start = rect.y + 36
        card_w = 228
        card_h = 52
        gap = 10
        left_x = rect.x + 12
        right_x = rect.x + 12 + card_w + gap

        # Row 1: Speed | Steering
        self._draw_metric_card(screen, left_x, y_start, card_w, card_h,
                               "SPEED", f"{self._display_speed:.0f}", "px/s",
                               self._display_speed / 200, cfg.GAUGE_FILL_NORMAL)
        self._draw_metric_card(screen, right_x, y_start, card_w, card_h,
                               "STEERING", f"{self._display_steering:.1f}", "deg",
                               abs(self._display_steering) / 30, cfg.TEXT_ACCENT,
                               centered_bar=True,
                               bar_value=self._display_steering / 30)

        # Row 2: Offset | FPS
        y2 = y_start + card_h + gap
        offset_norm = abs(offset) / (cfg.CAMERA_IMG_W / 2) if cfg.CAMERA_IMG_W > 0 else 0
        offset_color = (cfg.GAUGE_FILL_NORMAL if offset_norm < 0.3
                        else cfg.GAUGE_FILL_WARN if offset_norm < 0.6
                        else cfg.GAUGE_FILL_DANGER)
        self._draw_metric_card(screen, left_x, y2, card_w, card_h,
                               "LANE OFFSET", f"{offset:.0f}", "px",
                               offset_norm, offset_color,
                               centered_bar=True,
                               bar_value=offset / (cfg.CAMERA_IMG_W / 2))

        fps = metrics.get("fps", 0)
        fps_color = (cfg.GAUGE_FILL_NORMAL if fps >= 30
                     else cfg.GAUGE_FILL_WARN if fps >= 15
                     else cfg.GAUGE_FILL_DANGER)
        self._draw_metric_card(screen, right_x, y2, card_w, card_h,
                               "FPS", f"{fps:.0f}", "", fps / 60, fps_color)

        # Row 3: PID Components
        y3 = y2 + card_h + gap + 5
        self._draw_pid_section(screen, left_x, y3, card_w * 2 + gap, 95, metrics)

        # Row 4: Steering gauge
        y4 = y3 + 95 + gap
        self._draw_steering_gauge(screen, left_x, y4, card_w * 2 + gap, 100, metrics)

    def _draw_metric_card(self, screen, x, y, w, h, label, value_str, unit,
                          fill_ratio, fill_color, centered_bar=False, bar_value=0):
        card_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, cfg.BG_CARD, card_rect, border_radius=4)

        # Label
        lbl = self.fonts["label"].render(label, True, cfg.TEXT_SECONDARY)
        screen.blit(lbl, (x + 10, y + 5))

        # Value
        val = self.fonts["value"].render(value_str, True, cfg.TEXT_PRIMARY)
        screen.blit(val, (x + 10, y + 20))

        # Unit
        if unit:
            u = self.fonts["unit"].render(unit, True, cfg.TEXT_SECONDARY)
            vw = val.get_width()
            screen.blit(u, (x + 14 + vw, y + 32))

        # Bar gauge
        bar_x = x + w - 90
        bar_y = y + h // 2 - 4
        bar_w = 75
        bar_h = 8
        pygame.draw.rect(screen, cfg.GAUGE_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)

        if centered_bar:
            center = bar_x + bar_w // 2
            fill_w = int(abs(bar_value) * (bar_w // 2))
            fill_w = min(fill_w, bar_w // 2)
            if bar_value >= 0:
                pygame.draw.rect(screen, fill_color,
                                 (center, bar_y, fill_w, bar_h), border_radius=3)
            else:
                pygame.draw.rect(screen, fill_color,
                                 (center - fill_w, bar_y, fill_w, bar_h), border_radius=3)
        else:
            fill_w = int(max(0, min(1, fill_ratio)) * bar_w)
            if fill_w > 0:
                pygame.draw.rect(screen, fill_color,
                                 (bar_x, bar_y, fill_w, bar_h), border_radius=3)

    def _draw_pid_section(self, screen, x, y, w, h, metrics):
        pid_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, cfg.BG_CARD, pid_rect, border_radius=4)

        title = self.fonts["label"].render("PID COMPONENTS", True, cfg.TEXT_SECONDARY)
        screen.blit(title, (x + 10, y + 5))

        components = [
            ("P", metrics.get("pid_p", 0), cfg.PID_P_COLOR),
            ("I", metrics.get("pid_i", 0), cfg.PID_I_COLOR),
            ("D", metrics.get("pid_d", 0), cfg.PID_D_COLOR),
        ]

        max_val = max(abs(c[1]) for c in components) if components else 1.0
        max_val = max(max_val, 0.01)

        bar_x = x + 30
        bar_w = w - 100
        for i, (name, val, color) in enumerate(components):
            by = y + 25 + i * 22
            lbl = self.fonts["pid"].render(name, True, color)
            screen.blit(lbl, (x + 10, by))

            # Bar track
            pygame.draw.rect(screen, cfg.GAUGE_BG, (bar_x, by + 2, bar_w, 10), border_radius=4)

            # Centered fill
            center = bar_x + bar_w // 2
            fill_w = int(abs(val) / max_val * (bar_w // 2))
            fill_w = min(fill_w, bar_w // 2)
            if val >= 0:
                pygame.draw.rect(screen, color, (center, by + 2, fill_w, 10), border_radius=4)
            else:
                pygame.draw.rect(screen, color, (center - fill_w, by + 2, fill_w, 10), border_radius=4)

            # Value text
            vtxt = self.fonts["pid"].render(f"{val:+.3f}", True, cfg.TEXT_PRIMARY)
            screen.blit(vtxt, (bar_x + bar_w + 8, by))

    def _draw_steering_gauge(self, screen, x, y, w, h, metrics):
        # Split into two halves: left = offset sparkline, right = steering arc
        gauge_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, cfg.BG_CARD, gauge_rect, border_radius=4)

        # ── Left half: Offset history sparkline ──
        graph_x = x + 10
        graph_y = y + 20
        graph_w = w // 2 - 20
        graph_h = h - 35

        glbl = self.fonts["label"].render("OFFSET HISTORY", True, cfg.TEXT_SECONDARY)
        screen.blit(glbl, (graph_x, y + 5))

        pygame.draw.rect(screen, cfg.GAUGE_BG,
                         (graph_x, graph_y, graph_w, graph_h), border_radius=3)

        if len(self.offset_history) > 2:
            history = list(self.offset_history)
            max_off = max(abs(v) for v in history) if history else 1.0
            max_off = max(max_off, 10)
            points = []
            for i, v in enumerate(history):
                px = graph_x + int(i * graph_w / max(len(history) - 1, 1))
                py = graph_y + graph_h // 2 - int(v / max_off * (graph_h // 2 - 2))
                py = max(graph_y + 1, min(graph_y + graph_h - 1, py))
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(screen, cfg.TEXT_ACCENT, False, points, 1)
            # Zero line
            pygame.draw.line(screen, cfg.TEXT_SECONDARY,
                             (graph_x, graph_y + graph_h // 2),
                             (graph_x + graph_w, graph_y + graph_h // 2), 1)

        # ── Right half: Steering arc gauge ──
        steer_title = self.fonts["label"].render("STEERING", True, cfg.TEXT_SECONDARY)
        screen.blit(steer_title, (x + w // 2 + 10, y + 5))

        arc_cx = x + w * 3 // 4
        arc_cy = y + h - 18
        arc_r = 35

        # Arc track
        arc_rect = pygame.Rect(arc_cx - arc_r, arc_cy - arc_r, arc_r * 2, arc_r * 2)
        pygame.draw.arc(screen, cfg.GAUGE_BG, arc_rect, 0, math.pi, 4)

        # Tick marks
        for angle_deg in [-30, -15, 0, 15, 30]:
            t = (angle_deg + 30) / 60
            arc_angle = math.pi * (1 - t)
            tx = arc_cx + int((arc_r + 6) * math.cos(arc_angle))
            ty = arc_cy - int((arc_r + 6) * math.sin(arc_angle))
            pygame.draw.circle(screen, cfg.TEXT_SECONDARY, (tx, ty), 2)

        # L/R labels
        l_label = self.fonts["unit"].render("L", True, cfg.TEXT_SECONDARY)
        r_label = self.fonts["unit"].render("R", True, cfg.TEXT_SECONDARY)
        screen.blit(l_label, (arc_cx - arc_r - 14, arc_cy - 5))
        screen.blit(r_label, (arc_cx + arc_r + 6, arc_cy - 5))

        # Needle
        steering_deg = math.degrees(metrics.get("steering", 0))
        needle_t = max(0, min(1, (steering_deg + 30) / 60))
        needle_angle = math.pi * (1 - needle_t)
        nx = arc_cx + int((arc_r - 5) * math.cos(needle_angle))
        ny = arc_cy - int((arc_r - 5) * math.sin(needle_angle))
        pygame.draw.line(screen, cfg.TEXT_ACCENT, (arc_cx, arc_cy), (nx, ny), 3)
        pygame.draw.circle(screen, cfg.TEXT_ACCENT, (arc_cx, arc_cy), 4)

        val_text = self.fonts["pid"].render(f"{steering_deg:.1f} deg", True, cfg.TEXT_PRIMARY)
        vw = val_text.get_width()
        screen.blit(val_text, (arc_cx - vw // 2, arc_cy + 5))
