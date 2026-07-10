"""
Senxe Cerebellum — HUD Overlay System
==================================
v5.0 Cold Cyberpunk HUD — Bloom + EMA + Breathing + Zero Matplotlib

ARCHITECTURE:
  1. All high-intensity elements are drawn onto a separate black `glow_layer`.
  2. The glow_layer is Gaussian-blurred, then additively blended back
     onto the main frame — producing a hardware-accelerated-looking bloom.
  3. The force gauge cursor uses EMA smoothing for fluid "inertia" motion.
  4. Critical UI labels pulse via `np.sin(time * freq)` for a "breathing" feel.
  5. The evolution heatmap is a pure cv2 scrolling sparkline — zero Matplotlib.
"""

from __future__ import annotations

import time as _time
import cv2
import numpy as np

# ── HUD State (encapsulated, resettable) ──

class HUDState:
    """Encapsulated HUD rendering state — supports proper reset between runs."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_counter = 0
        self.particle_pool = []
        self.last_spike_time = np.zeros(64)
        self.force_ema = 0.0
        self.episode_firing_history = []
        self.evolution_cache = [None, 0]

hud = HUDState()
_FORCE_EMA_ALPHA = 0.18                     # EMA coefficient: lower = smoother glide

# ── Cold Cyberpunk 4-color semantic palette (RGB order) ──
# Force=Ice Blue | Torque=Neon Cyan | Position=Magenta | Goal=Muted Amber
_GROUP_COLORS = {
    'force':    {'echo': (255, 255, 255), 'active': (180, 210, 255), 'inactive': (20, 30, 50)},
    'torque':   {'echo': (255, 255, 255), 'active': (0,   255, 240), 'inactive': (5,  40, 38)},
    'position': {'echo': (255, 255, 255), 'active': (220, 80,  220), 'inactive': (38, 12, 38)},
    'goal':     {'echo': (255, 255, 255), 'active': (220, 185, 90),  'inactive': (40, 32, 12)},
}
_CH_GROUP_MAP = ['force'] * 16 + ['torque'] * 16 + ['position'] * 16 + ['goal'] * 16

# Bloom kernel size (must be odd). Larger = softer glow, more GPU-like feel.
_BLOOM_KSIZE = 31
_BLOOM_SIGMA = 12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper: Drop-shadow text (military HUD typography)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _hud_text(target, text, x, y, color=(255, 255, 255),
              scale=0.35, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Render pixel-perfect HUD text with a 2px black drop shadow for depth."""
    # Shadow pass (1px down-right offset, thick black outline for contrast)
    cv2.putText(target, text, (x + 1, y + 1), font, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Foreground pass
    cv2.putText(target, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Particle system — spawn + update + draw onto glow_layer for bloom
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _spawn_particles(cx, cy, count, color_base, speed=1.8, life_range=(10, 22)):
    """Spawn radial particles around (cx, cy). Drawn onto glow_layer for bloom."""
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        spd = np.random.uniform(0.4, speed)
        lf = np.random.randint(life_range[0], life_range[1])
        hud.particle_pool.append(dict(
            x=float(cx) + np.random.uniform(-3, 3),
            y=float(cy) + np.random.uniform(-3, 3),
            vx=np.cos(angle) * spd, vy=np.sin(angle) * spd,
            life=lf, max_life=lf, color_base=color_base
        ))


def _update_and_draw_particles(frame, glow_layer):
    """Advance physics & render particles onto glow_layer for additive bloom."""
    h, w = frame.shape[:2]
    alive = []
    for p in hud.particle_pool:
        p['x'] += p['vx']; p['y'] += p['vy']; p['life'] -= 1
        p['vx'] *= 0.93; p['vy'] *= 0.93          # drag
        if p['life'] <= 0:
            continue
        px, py = int(p['x']), int(p['y'])
        if px < 4 or py < 4 or px >= w - 4 or py >= h - 4:
            continue
        t = p['life'] / p['max_life']               # 1.0→0.0 fade
        cb = p['color_base']
        brightness = t * 0.85
        r = max(2, int(3 * t))
        color = (int(cb[0] * brightness),
                 int(cb[1] * brightness),
                 int(cb[2] * brightness))
        cv2.circle(glow_layer, (px, py), r + 1, color, -1, cv2.LINE_AA)
        alive.append(p)
    hud.particle_pool.clear()
    hud.particle_pool.extend(alive)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feathered darken (unchanged utility, used for subtle panel backdrops)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _feathered_darken(frame, y0, y1, x0, x1, darkness=0.08, feather=12):
    """Darken a rectangular ROI with feathered edges for a frosted-glass effect."""
    h, w = frame.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    rh, rw = y1 - y0, x1 - x0
    if rh <= 0 or rw <= 0:
        return
    alpha = np.ones((rh, rw), dtype=np.float32)
    f = min(feather, rh // 2, rw // 2)
    for i in range(f):
        t = (i + 1) / (f + 1)
        alpha[i, :] = np.minimum(alpha[i, :], t)
        alpha[rh - 1 - i, :] = np.minimum(alpha[rh - 1 - i, :], t)
        alpha[:, i] = np.minimum(alpha[:, i], t)
        alpha[:, rw - 1 - i] = np.minimum(alpha[:, rw - 1 - i], t)
    factor = darkness + (1.0 - darkness) * (1.0 - alpha)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi *= factor[:, :, np.newaxis]
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HUD Text Overlay — Military-grade precision typography
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _overlay_text(frame, ep, reward, pdi, min_health, glow_layer,
                  distance=0.0, force_mag=0.0, torque_mag=0.0,
                  depth=0.0, success_rate=0.0, force_safe_rate=0.0,
                  force_threshold=20.0, is_sim=False):
    """Minimalist military HUD — small scale, pure white, drop shadows.
    WARNING / DANGER text pulses with a sine-wave breathing effect
    and is drawn onto glow_layer for bloom."""
    h, w = frame.shape[:2]
    fc = hud.frame_counter
    curr_time = fc / 30.0

    # ── Safety classification ──
    fn = force_mag / force_threshold if force_threshold > 0 else 0
    if fn < 0.5:
        status, s_color = "NOMINAL", (160, 220, 160)
    elif fn < 1.0:
        status, s_color = "CAUTION", (220, 200, 80)
    else:
        status, s_color = "DANGER", (255, 80, 80)

    # ── Top-left HUD block (3 lines, larger + brighter for 720p legibility) ──
    lx, ly = 12, 22
    line_h = 20  # increased vertical spacing for breathing room

    _hud_text(frame, f"EP {ep:03d}   R {reward:+.1f}", lx, ly, (255, 255, 255), 0.50)
    
    # ── Simulator Status Indicator ──
    sim_y = ly + line_h
    if is_sim:
        _hud_text(frame, "[SIMULATOR MODE]", lx, sim_y, (80, 80, 255), 0.42)
    else:
        _hud_text(frame, "[PURE WETWARE]", lx, sim_y, (80, 255, 80), 0.42)

    _hud_text(frame, f"F {force_mag:5.1f}N  T {torque_mag:4.2f}Nm  D {depth:.3f}m",
              lx, sim_y + line_h, (140, 240, 255), 0.42)
    _hud_text(frame, f"SR {success_rate:3.0f}%  FSR {force_safe_rate:3.0f}%  PDI {pdi:.2f}",
              lx, sim_y + line_h * 2, (100, 210, 230), 0.42)

    # ── Status badge — breathing pulse on WARNING/DANGER ──
    badge_x = lx + 300
    badge_y = ly
    if fn >= 0.5:
        # Breathing: sinusoidal alpha modulation (0.5–1.0 range)
        breath = 0.5 + 0.5 * np.sin(curr_time * 5.0)      # ~0.8 Hz pulse
        pulse_color = tuple(int(c * breath) for c in s_color)
        # Draw on glow_layer for bloom halo around warning text
        _hud_text(glow_layer, status, badge_x, badge_y, pulse_color, 0.40, 1)
    _hud_text(frame, status, badge_x, badge_y, s_color, 0.40, 1)

    # ── Bottom-left: minimal health indicator ──
    _hud_text(frame, f"HEALTH {min_health:.2f}", lx, h - 12, (100, 110, 120), 0.30)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8×8 Neuron Grid — Circles + Cold Cyberpunk + Bloom spikes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _overlay_neuron_grid(frame, firing_rates, min_health, glow_layer,
                         health_arr=None, force_mag=0.0, force_threshold=20.0):
    """8×8 neuron grid rendered as sleek circles with Cold Cyberpunk palette.
    Force=Ice Blue | Torque=Neon Cyan | Position=Magenta | Goal=Muted Amber.
    Spiking neurons flash WHITE with heavy colored bloom, then smoothly decay.
    Right-side force safety gauge with EMA-smoothed cursor."""
    h, w = frame.shape[:2]
    fc = hud.frame_counter
    curr_time = fc / 30.0

    # ── Grid geometry ──
    radius = 6                              # circle radius (px)
    spacing = 16                            # center-to-center distance
    grid_n = 8
    total = grid_n * spacing                # 128px
    gauge_w = 6; gauge_gap = 10
    gx0, gy0 = 14, 86                      # origin (pushed down for breathing room below HUD)

    if gy0 + total + 5 > h or gx0 + total + gauge_gap + gauge_w + 5 > w:
        return

    # ── Spike detection + echo decay ──
    spike_thresh = np.percentile(firing_rates, 75)
    spiking = firing_rates > spike_thresh
    for ch in range(64):
        if spiking[ch]:
            hud.last_spike_time[ch] = curr_time

    echo_decay = np.zeros(64, dtype=np.float32)
    for ch in range(64):
        dt = curr_time - hud.last_spike_time[ch]
        # Smooth exponential decay over 0.45s — longer tail than before
        echo_decay[ch] = max(0.0, 1.0 - dt / 0.70)

    fr_median = np.median(firing_rates)
    fr_range = max(firing_rates.max() - firing_rates.min(), 1.0)

    # ── Pure black background behind the entire grid area ──
    pad = 4  # padding around the grid
    bg_x0 = max(0, gx0 - pad)
    bg_y0 = max(0, gy0 - pad)
    bg_x1 = min(w, gx0 + total + pad)
    bg_y1 = min(h, gy0 + total + pad)
    frame[bg_y0:bg_y1, bg_x0:bg_x1] = 0

    # ── Draw 8×8 circle grid ──
    for row in range(grid_n):
        for col in range(grid_n):
            ch = row * grid_n + col
            cx = gx0 + col * spacing + radius
            cy = gy0 + row * spacing + radius

            if cx + radius >= w or cy + radius >= h:
                continue

            group = _CH_GROUP_MAP[ch]
            colors = _GROUP_COLORS[group]
            ed = echo_decay[ch]
            fr = firing_rates[ch]

            if ed > 0.05:
                # ── STATE A: Spike echo — flash white → decay to active color ──
                ec = colors['echo']   # pure white
                ac = colors['active']
                t = ed                # 1.0 at spike, decays to 0
                cr = int(ec[0] * t + ac[0] * (1 - t))
                cg = int(ec[1] * t + ac[1] * (1 - t))
                cb_c = int(ec[2] * t + ac[2] * (1 - t))
                # Draw filled circle on frame
                cv2.circle(frame, (cx, cy), radius, (cr, cg, cb_c), -1, cv2.LINE_AA)
                # ── BLOOM: Draw high-intensity version onto glow_layer ──
                bloom_brightness = ed * 1.2
                bloom_color = (int(min(255, ac[0] * bloom_brightness)),
                               int(min(255, ac[1] * bloom_brightness)),
                               int(min(255, ac[2] * bloom_brightness)))
                cv2.circle(glow_layer, (cx, cy), radius + 5, bloom_color, -1, cv2.LINE_AA)
                # Spawn particles on recent spikes (ed > 0.65 for wider emission window)
                if ed > 0.65:
                    _spawn_particles(cx, cy, 3, ac, speed=2.0, life_range=(8, 16))

            elif fr > fr_median:
                # ── STATE B: Active — saturated group color, thin stroke ──
                ac = colors['active']
                intensity = np.clip((fr - fr_median) / (fr_range * 0.5 + 1e-6), 0, 1)
                alpha = 0.55 + intensity * 0.45
                cr = int(ac[0] * alpha)
                cg = int(ac[1] * alpha)
                cb_c = int(ac[2] * alpha)
                cv2.circle(frame, (cx, cy), radius, (cr, cg, cb_c), -1, cv2.LINE_AA)
                # Ultra-thin bright stroke for definition
                cv2.circle(frame, (cx, cy), radius, ac, 1, cv2.LINE_AA)
            else:
                # ── STATE C: Inactive — ghost circle, barely visible stroke ──
                ic = colors['inactive']
                cv2.circle(frame, (cx, cy), radius, ic, 1, cv2.LINE_AA)

    # ── Force Safety Gauge (right of grid) — EMA-smoothed cursor ──
    #
    # EMA LOGIC: Instead of directly mapping force_mag to the bar fill,
    # we smoothly interpolate using an exponential moving average.
    # hud.force_ema = alpha * new_value + (1 - alpha) * old_value
    # Lower alpha → smoother/slower response (more "inertia").
    raw_fn = np.clip(force_mag / force_threshold, 0, 1.5)
    hud.force_ema = _FORCE_EMA_ALPHA * raw_fn + (1.0 - _FORCE_EMA_ALPHA) * hud.force_ema
    smoothed_fn = hud.force_ema

    bar_x = gx0 + total + gauge_gap
    bar_y0 = gy0
    bar_h = total
    fill_h = int(bar_h * min(smoothed_fn, 1.0))

    if bar_x + gauge_w <= w and bar_y0 + bar_h <= h:
        # Subtle darkened background track
        _feathered_darken(frame, bar_y0, bar_y0 + bar_h, bar_x - 1, bar_x + gauge_w + 1,
                          darkness=0.12, feather=4)

        # Draw gradient fill from bottom upward
        if fill_h > 0:
            fill_top = bar_y0 + bar_h - fill_h
            for py in range(fill_top, bar_y0 + bar_h):
                t = (bar_y0 + bar_h - py) / bar_h  # 0=bottom, 1=top
                # Cold gradient: Ice-blue(0) → Cyan(0.5) → Magenta-red(1.0)
                if t < 0.5:
                    t2 = t * 2.0
                    cr = int(80 + 100 * t2)
                    cg = int(200 + 55 * (1 - t2))
                    cb_c = int(255 - 30 * t2)
                else:
                    t2 = (t - 0.5) * 2.0
                    cr = int(180 + 75 * t2)
                    cg = int(140 * (1 - t2))
                    cb_c = int(225 * (1 - t2) + 60 * t2)
                for px in range(bar_x, min(bar_x + gauge_w, w)):
                    frame[py, px] = [cr, cg, cb_c]

        # Threshold line at 100% — crisp white dash
        thresh_y = bar_y0
        if 0 <= thresh_y < h:
            cv2.line(frame, (bar_x - 2, thresh_y), (bar_x + gauge_w + 2, thresh_y),
                     (200, 200, 200), 1, cv2.LINE_AA)

        # Floating cursor for current smoothed force (horizontal tick mark)
        cursor_y = int(bar_y0 + bar_h - bar_h * min(smoothed_fn, 1.3))
        cursor_y = max(bar_y0, min(bar_y0 + bar_h - 1, cursor_y))
        cursor_color = (255, 255, 255)
        if smoothed_fn > 1.0:
            # Danger: pulsing red cursor on glow_layer
            breath = 0.6 + 0.4 * np.sin(curr_time * 8.0)
            cursor_color = (int(255 * breath), int(60 * breath), int(60 * breath))
            cv2.line(glow_layer, (bar_x - 4, cursor_y), (bar_x + gauge_w + 4, cursor_y),
                     (255, 80, 80), 2, cv2.LINE_AA)
        cv2.line(frame, (bar_x - 3, cursor_y), (bar_x + gauge_w + 3, cursor_y),
                 cursor_color, 2, cv2.LINE_AA)

        # Tiny force label next to gauge
        _hud_text(frame, f"{force_mag:.0f}N", bar_x - 2, bar_y0 + bar_h + 14,
                  (160, 170, 180), 0.28)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Channel Evolution — Pure cv2 scrolling sparkline (ZERO Matplotlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_EVOLUTION_WINDOW = 50   # Show last N episodes as scrolling sparkline columns

# Sparkline group colors (BGR for cv2) — matches the Cold Cyberpunk palette
_SPARK_COLORS = [
    (255, 210, 180),   # Force  — Ice Blue (BGR)
    (240, 255, 0),     # Torque — Neon Cyan (BGR)
    (220, 80, 220),    # Position — Magenta (BGR)
    (90, 185, 220),    # Goal   — Muted Amber (BGR)
]
_SPARK_LABELS = ['F', 'T', 'P', 'G']


def _overlay_evolution_heatmap(frame, glow_layer):
    """Channel Evolution — Pure cv2 scrolling dot-matrix sparkline.
    4 rows (Force/Torque/Position/Goal), each a horizontal sparkline
    of per-episode average firing rate. Cached per-episode for speed."""
    h, w = frame.shape[:2]
    n_eps = len(hud.episode_firing_history)
    if n_eps < 2:
        return

    # ── Panel geometry (top-right) ──
    panel_w, panel_h = 200, 100
    margin_r, margin_t = 10, 10
    px0 = w - panel_w - margin_r
    py0 = margin_t

    if px0 < w // 3 or py0 + panel_h > h:
        return

    # Only re-render when new episode data arrives (cached)
    need_update = (hud.evolution_cache[0] is None or hud.evolution_cache[1] != n_eps)

    if need_update:
        hud.evolution_cache[1] = n_eps

        window = min(n_eps, _EVOLUTION_WINDOW)
        recent = np.array(hud.episode_firing_history[-window:])  # (window, 64)

        # Group into 4 channel banks: mean firing rate per bank per episode
        # Force(0-15), Torque(16-31), Position(32-47), Goal(48-63)
        grouped = np.zeros((4, window), dtype=np.float32)
        for gi, (lo, hi) in enumerate([(0, 16), (16, 32), (32, 48), (48, 64)]):
            grouped[gi] = recent[:, lo:hi].mean(axis=1)

        # Per-row normalize to [0, 1] for sparkline height
        for gi in range(4):
            mn, mx = grouped[gi].min(), grouped[gi].max()
            rng = mx - mn if (mx - mn) > 0.5 else 0.5
            grouped[gi] = (grouped[gi] - mn) / rng

        # Render onto a small black canvas
        canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        row_h = panel_h // 4  # 25px per sparkline row

        for gi in range(4):
            base_y = gi * row_h
            color = _SPARK_COLORS[gi]
            vals = grouped[gi]

            # Draw sparkline as connected dots
            x_step = max(1.0, (panel_w - 24) / max(window - 1, 1))
            pts = []
            for si in range(window):
                sx = int(20 + si * x_step)
                # Map normalized value to vertical position within row
                sy = int(base_y + row_h - 3 - vals[si] * (row_h - 6))
                pts.append((sx, sy))

                # Dot-matrix trail: small filled circles
                brightness = 0.3 + 0.7 * (si / max(window - 1, 1))  # fade-in
                dot_color = tuple(int(c * brightness) for c in color)
                cv2.circle(canvas, (sx, sy), 2, dot_color, -1, cv2.LINE_AA)

            # Connect with thin line
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    t = (i + 1) / len(pts)
                    line_color = tuple(int(c * t * 0.5) for c in color)
                    cv2.line(canvas, pts[i], pts[i + 1], line_color, 1, cv2.LINE_AA)

            # Latest point gets a bright glow dot
            if pts:
                lx, ly = pts[-1]
                cv2.circle(canvas, (lx, ly), 3, color, -1, cv2.LINE_AA)

            # Row label (left side)
            label_y = base_y + row_h // 2 + 4
            cv2.putText(canvas, _SPARK_LABELS[gi], (3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

            # Subtle separator line
            if gi < 3:
                sep_y = base_y + row_h
                cv2.line(canvas, (18, sep_y), (panel_w - 4, sep_y),
                         (30, 35, 40), 1, cv2.LINE_AA)

        # Episode range label at bottom
        start_ep = max(1, n_eps - window + 1)
        ep_label = f"ep {start_ep}-{n_eps}"
        cv2.putText(canvas, ep_label, (panel_w - 70, panel_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 90, 100), 1, cv2.LINE_AA)

        hud.evolution_cache[0] = canvas

    # ── Alpha blend cached sparkline panel onto frame ──
    cached = hud.evolution_cache[0]
    if cached is not None:
        ch_h, ch_w = cached.shape[:2]
        y_end = min(py0 + ch_h, h)
        x_end = min(px0 + ch_w, w)
        ah = y_end - py0; aw = x_end - px0
        if ah > 0 and aw > 0:
            # Darken backdrop for contrast
            _feathered_darken(frame, py0, y_end, px0, x_end, darkness=0.06, feather=8)
            # Blend sparkline canvas (only non-black pixels to preserve transparency)
            roi = frame[py0:y_end, px0:x_end].astype(np.float32)
            overlay = cached[:ah, :aw].astype(np.float32)
            # Additive-style blend: wherever overlay is bright, add it
            mask = (overlay.max(axis=2, keepdims=True) > 10).astype(np.float32)
            blended = roi * (1.0 - mask * 0.7) + overlay * 0.9
            frame[py0:y_end, px0:x_end] = np.clip(blended, 0, 255).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY: draw_overlay — Global Bloom Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_overlay(frame, ep, reward, pdi, min_health, firing_rates, reward_history,
                 distance=0.0, force_mag=0.0, torque_mag=0.0, depth=0.0,
                 success_rate=0.0, force_safe_rate=0.0,
                 force_vec=None, health_arr=None, force_threshold=20.0, is_sim=False):
    """v5.0 Cold Cyberpunk HUD — Global Bloom + EMA + Breathing + Zero Matplotlib.

    BLOOM PIPELINE:
      1. Create a black glow_layer (same size as frame).
      2. Draw high-intensity elements (spike flashes, warning text, danger cursor)
         onto this glow_layer.
      3. Apply fast cv2.GaussianBlur to glow_layer → produces soft glow halos.
      4. Additively blend glow_layer back onto the main frame using cv2.add().
    This emulates the hardware-accelerated bloom seen in native macOS AppKit UIs.

    EMA SMOOTHING:
      The force gauge cursor uses exponential moving average so the indicator
      glides with physical "inertia" instead of jittering per-frame.
      Formula: ema = alpha * new + (1 - alpha) * old   (alpha = 0.18)
    """
    frame = np.ascontiguousarray(frame)
    hud.frame_counter += 1

    # ── Step 1: Allocate glow layer (black canvas, same dims) ──
    glow_layer = np.zeros_like(frame)

    # ── Step 2: Draw all UI components ──
    # Neuron grid draws spiking neurons + force gauge onto glow_layer
    _overlay_neuron_grid(frame, firing_rates, min_health, glow_layer,
                         health_arr=health_arr, force_mag=force_mag,
                         force_threshold=force_threshold)

    # Particles are drawn onto glow_layer for bloom effect
    _update_and_draw_particles(frame, glow_layer)

    # HUD text (warning/danger text drawn onto glow_layer for bloom)
    _overlay_text(frame, ep, reward, pdi, min_health, glow_layer,
                  distance=distance, force_mag=force_mag, torque_mag=torque_mag,
                  depth=depth, success_rate=success_rate,
                  force_safe_rate=force_safe_rate, force_threshold=force_threshold,
                  is_sim=is_sim)

    # Heatmap
    _overlay_evolution_heatmap(frame, glow_layer)

    # ── Step 3: Blur the glow layer → soft bloom halos ──
    bloom = cv2.GaussianBlur(glow_layer, (_BLOOM_KSIZE, _BLOOM_KSIZE), _BLOOM_SIGMA)

    # ── Step 4: Additive blend → premium glowing edges ──
    # cv2.add() clamps at 255 automatically, which is exactly what we want
    # for additive light blending (like real optical bloom).
    frame = cv2.add(frame, bloom)

    return frame
