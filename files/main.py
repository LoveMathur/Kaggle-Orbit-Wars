"""
Orbit Wars — Competitive Agent v10 "Production Rush" (Fully Wired Edition)
========================================================================
"""

import math
from typing import List, Tuple, Optional

# ── Engine constants (do NOT change) ─────────────────────────────────────────
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_FLEET_SPEED = 6.0
ORBIT_THRESHOLD = 50.0

# ── Tunable Parameters (Updated with Optuna Best Estimates) ──────────────────
MIN_GARRISON          = 6       # Absolute ship floor per planet
SAFETY_MARGIN         = 8       # Extra ships added on top of computed needed (safety_margin_ships)
MIN_LAUNCH_SIZE       = 1       # Minimum size allowed for an outbound offensive fleet
SUN_SAFE_MARGIN       = 1.4641  # Collision buffer around sun radius
PRODUCTION_ROI_WEIGHT = 0.8414  # Scaling factor for economic targeting evaluation
EN_ROUTE_BUFFER       = 4       # Reinforcement padding added to net-needed calculations

MAX_LAUNCHES          = 3       # Max simultaneous launches per planet per turn
REINFORCE_RATIO       = 0.9     # Trigger emergency reinforce if threat/garrison > this
COMET_MAX_DIST        = 40      # Ignore comets beyond this distance

# Sun bypass offsets (radians) — tried in order until one clears
SUN_BYPASSES      = [0.30, -0.30, 0.55, -0.55, 0.85, -0.85, 1.2, -1.2]


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LAYER 1 — PHYSICS
# ╚══════════════════════════════════════════════════════════════════════════════

def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def fleet_speed(n: int) -> float:
    """Official spec: speed = 1 + (maxSpeed-1) * (log(n)/log(1000))^1.5"""
    if n <= 1:
        return 1.0
    n = min(max(int(n), 1), 1000)
    ratio = math.log(n) / math.log(1000)
    return 1.0 + (MAX_FLEET_SPEED - 1.0) * (ratio ** 1.5)


def travel_time(d: float, n: int) -> float:
    return d / max(fleet_speed(n), 1e-9)


def is_orbiting(planet: list) -> bool:
    """True if the planet rotates around the sun."""
    return dist(planet[2], planet[3], CENTER_X, CENTER_Y) + planet[4] < ORBIT_THRESHOLD


def predict_pos(sx: float, sy: float, target: list,
                ang_vel: float, n_ships: int, iters: int = 8) -> Tuple[float, float]:
    """Iterative intercept solver — finds where fleet meets orbiting planet."""
    tx, ty = target[2], target[3]
    if not is_orbiting(target):
        return tx, ty
    dx, dy = tx - CENTER_X, ty - CENTER_Y
    r = math.hypot(dx, dy)
    if r < 1e-9:
        return tx, ty
    theta0 = math.atan2(dy, dx)
    spd = max(fleet_speed(n_ships), 1e-9)
    t = 0.0
    for _ in range(iters):
        theta = theta0 + ang_vel * t
        fx = CENTER_X + r * math.cos(theta)
        fy = CENTER_Y + r * math.sin(theta)
        t = dist(sx, sy, fx, fy) / spd
    theta = theta0 + ang_vel * t
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """True if the line segment (x1,y1)→(x2,y2) clips the sun kill zone."""
    r = SUN_RADIUS + SUN_SAFE_MARGIN
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return dist(x1, y1, CENTER_X, CENTER_Y) <= r
    t = max(0.0, min(1.0, ((CENTER_X - x1)*dx + (CENTER_Y - y1)*dy) / (dx*dx + dy*dy)))
    return dist(x1 + t*dx, y1 + t*dy, CENTER_X, CENTER_Y) <= r


def safe_angle(sx: float, sy: float, tx: float, ty: float) -> Optional[float]:
    """Return a launch angle from (sx,sy) toward (tx,ty) that avoids the sun."""
    direct = math.atan2(ty - sy, tx - sx)
    if not segment_hits_sun(sx, sy, tx, ty):
        return direct

    d_to_tgt = dist(sx, sy, tx, ty)
    for offset in SUN_BYPASSES:
        a = direct + offset
        far_x = sx + (d_to_tgt + 30) * math.cos(a)
        far_y = sy + (d_to_tgt + 30) * math.sin(a)
        if not segment_hits_sun(sx, sy, far_x, far_y):
            return a
    return None


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LAYER 2 — ECONOMY
# ╚══════════════════════════════════════════════════════════════════════════════

def incoming_threat(planet: list, fleets: list, my_id: int) -> float:
    """Enemy ship-count plausibly heading toward this planet."""
    px, py = planet[2], planet[3]
    threat = 0.0
    for f in fleets:
        if f[1] == my_id:
            continue
        fx, fy, fa = f[2], f[3], f[4]
        d = dist(fx, fy, px, py)
        if d > 70:
            continue
        aim = math.atan2(py - fy, px - fx)
        diff = abs((fa - aim + math.pi) % (2 * math.pi) - math.pi)
        if diff < 0.45:
            threat += f[6] * (1.0 if d < 25 else 0.6)
    return threat


def garrison(planet: list, fleets: list, my_id: int) -> int:
    """Ships to keep on this planet for basic and threat defense."""
    thr = incoming_threat(planet, fleets, my_id)
    return max(MIN_GARRISON, int(MIN_GARRISON + thr * 1.25))


def available(planet: list, fleets: list, my_id: int) -> int:
    """Launchable ships after keeping garrison."""
    return max(0, planet[5] - garrison(planet, fleets, my_id))


def ships_required(sx: float, sy: float, target: list,
                   n_ships: int, ang_vel: float) -> int:
    """Ships needed to capture target, accounting for production growth during transit."""
    px, py = predict_pos(sx, sy, target, ang_vel, n_ships)
    tt = travel_time(dist(sx, sy, px, py), n_ships)
    growth = target[6] * tt if target[1] >= 0 else 0.0
    return int(target[5] + growth) + SAFETY_MARGIN


def en_route(src_id: int, fleets: list, my_id: int, tx: float, ty: float) -> int:
    """Ships already in flight from src_id toward (tx,ty)."""
    total = 0
    for f in fleets:
        if f[1] != my_id or f[5] != src_id:
            continue
        aim = math.atan2(ty - f[3], tx - f[2])
        diff = abs((f[4] - aim + math.pi) % (2 * math.pi) - math.pi)
        if diff < 0.35:
            total += f[6]
    return total


def production_roi(sx: float, sy: float, target: list,
                   n_ships: int, ang_vel: float) -> float:
    """Production return-on-investment score (Wired with PRODUCTION_ROI_WEIGHT)."""
    px, py = predict_pos(sx, sy, target, ang_vel, n_ships)
    d = dist(sx, sy, px, py) + 1.0
    prod = target[6]
    cost = ships_required(sx, sy, target, n_ships, ang_vel)
    
    neutral_bonus = 1.4 if target[1] == -1 else 1.0
    orbit_bonus = 1.25 if is_orbiting(target) else 1.0
    
    base_roi = (prod * orbit_bonus * neutral_bonus) / (math.log(d + 1) * (cost + 1) ** 0.4)
    return PRODUCTION_ROI_WEIGHT * base_roi


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LAYER 3 — TACTICS
# ╚══════════════════════════════════════════════════════════════════════════════

def pick_targets(sx: float, sy: float, enemies: list,
                 comet_ids: set, budget: int,
                 ang_vel: float) -> List[Tuple]:
    """Rank all valid enemy/neutral targets by production ROI."""
    scored = []
    for t in enemies:
        if t[0] in comet_ids and dist(sx, sy, t[2], t[3]) > COMET_MAX_DIST:
            continue

        px, py = predict_pos(sx, sy, t, ang_vel, max(1, budget))
        angle = safe_angle(sx, sy, px, py)
        if angle is None:
            continue

        score = production_roi(sx, sy, t, max(1, budget), ang_vel)
        scored.append((score, t, px, py, angle))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LAYER 4 — AGENT
# ╚══════════════════════════════════════════════════════════════════════════════

def agent(obs, conf=None):
    if isinstance(obs, dict):
        my_id     = obs.get('player', 0)
        planets   = obs.get('planets', [])
        fleets    = obs.get('fleets', [])
        ang_vel   = obs.get('angular_velocity', 0.035)
        comet_ids = set(obs.get('comet_planet_ids', []) or [])
    else:
        my_id     = obs.player
        planets   = obs.planets
        fleets    = getattr(obs, 'fleets', [])
        ang_vel   = getattr(obs, 'angular_velocity', 0.035)
        comet_ids = set(getattr(obs, 'comet_planet_ids', None) or [])

    moves: List = []
    my_planets  = [p for p in planets if p[1] == my_id]
    foes        = [p for p in planets if p[1] != my_id]

    if not my_planets or not foes:
        return moves

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Emergency reinforcement
    # ──────────────────────────────────────────────────────────────────────────
    reinforced_from = set()
    for tgt in sorted(my_planets,
                      key=lambda p: incoming_threat(p, fleets, my_id) / max(p[5], 1),
                      reverse=True)[:2]:
        thr = incoming_threat(tgt, fleets, my_id)
        if thr == 0 or thr / max(tgt[5], 1) < REINFORCE_RATIO:
            break
        helpers = [p for p in my_planets
                   if p[0] != tgt[0]
                   and p[0] not in reinforced_from
                   and incoming_threat(p, fleets, my_id) == 0
                   and available(p, fleets, my_id) >= 8]
        if not helpers:
            continue
        helper = max(helpers, key=lambda p: available(p, fleets, my_id))
        send = min(available(helper, fleets, my_id), int(thr * 1.5) + 5)
        if send < 1:
            continue
        angle = safe_angle(helper[2], helper[3], tgt[2], tgt[3])
        if angle is None:
            continue
        moves.append([helper[0], angle, send])
        reinforced_from.add(helper[0])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Offensive launches (Wired with EN_ROUTE_BUFFER & MIN_LAUNCH_SIZE)
    # ──────────────────────────────────────────────────────────────────────────
    for src in my_planets:
        src_id = src[0]
        sx, sy = src[2], src[3]
        budget = available(src, fleets, my_id)

        if budget < MIN_LAUNCH_SIZE:
            continue

        candidates = pick_targets(sx, sy, foes, comet_ids, budget, ang_vel)
        if not candidates:
            continue

        remaining   = budget
        n_launched  = 0

        for (score, target, pred_x, pred_y, angle) in candidates:
            if remaining < MIN_LAUNCH_SIZE or n_launched >= MAX_LAUNCHES:
                break

            needed = ships_required(sx, sy, target, remaining, ang_vel)
            already = en_route(src_id, fleets, my_id, pred_x, pred_y)
            
            # Integration of EN_ROUTE_BUFFER to safely cover arrival variance
            needed_net = max(1, (needed - already) + EN_ROUTE_BUFFER)

            if needed_net <= remaining:
                launch = needed_net
            else:
                if n_launched == 0:
                    launch = remaining  
                else:
                    continue

            launch = min(launch, src[5] - MIN_GARRISON)
            if launch < MIN_LAUNCH_SIZE:
                break

            pred_x2, pred_y2 = predict_pos(sx, sy, target, ang_vel, launch)
            angle2 = safe_angle(sx, sy, pred_x2, pred_y2)
            if angle2 is None:
                if n_launched == 0:
                    continue  
                else:
                    break

            moves.append([src_id, angle2, launch])
            remaining  -= launch
            n_launched += 1

    return moves