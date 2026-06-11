# ─── Baseline (v7 simple attacker) ────────────────────────────────────────────

import math
from typing import List, Tuple, Optional

# ── Board constants ──────────────────────────────────────────────────────────
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_SAFE_MARGIN = 1.2
MAX_FLEET_SPEED = 6.0
ORBIT_THRESHOLD = 50.0

# ── Tuning knobs ─────────────────────────────────────────────────────────────
MIN_GARRISON = 4               
SAFETY_MARGIN_SHIPS = 4        
THREAT_LOOKAHEAD = 25          
COMET_OPPORTUNITY_DIST = 25    
REINFORCE_THRESHOLD = 0.60     
MIN_LAUNCH_SIZE = 8            
SUN_BYPASS_ANGLES = [0.22, -0.22, 0.45, -0.45, 0.6, -0.6]

# ── Physics helpers ──────────────────────────────────────────────────────────

def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

def fleet_speed(num_ships: int) -> float:
    if num_ships <= 1:
        return 1.0
    n = min(max(int(num_ships), 1), 1000)
    ratio = math.log(n) / math.log(1000)
    return 1.0 + (MAX_FLEET_SPEED - 1.0) * (ratio ** 1.5)

def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    r = SUN_RADIUS + SUN_SAFE_MARGIN
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return dist(x1, y1, CENTER_X, CENTER_Y) <= r
    t = max(0.0, min(1.0, ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / (dx * dx + dy * dy)))
    return dist(x1 + t * dx, y1 + t * dy, CENTER_X, CENTER_Y) <= r

def safe_angle(src_x: float, src_y: float, tgt_x: float, tgt_y: float) -> Optional[float]:
    direct = math.atan2(tgt_y - src_y, tgt_x - src_x)
    if not segment_hits_sun(src_x, src_y, tgt_x, tgt_y):
        return direct
    for offset in SUN_BYPASS_ANGLES:
        angle = direct + offset
        far_x = src_x + 200 * math.cos(angle)
        far_y = src_y + 200 * math.sin(angle)
        if not segment_hits_sun(src_x, src_y, far_x, far_y):
            return angle
    return None

def is_orbiting(planet: list) -> bool:
    return dist(planet[2], planet[3], CENTER_X, CENTER_Y) + planet[4] < ORBIT_THRESHOLD

def predict_pos(src_x: float, src_y: float, target: list, angular_velocity: float, launch_ships: int, iterations: int = 6) -> Tuple[float, float]:
    tx, ty = target[2], target[3]
    if not is_orbiting(target):
        return tx, ty
    dx, dy = tx - CENTER_X, ty - CENTER_Y
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return tx, ty
    theta0 = math.atan2(dy, dx)
    t = 0.0
    speed = max(fleet_speed(launch_ships), 1e-6)
    for _ in range(iterations):
        theta = theta0 + angular_velocity * t
        fx = CENTER_X + r * math.cos(theta)
        fy = CENTER_Y + r * math.sin(theta)
        d = dist(src_x, src_y, fx, fy)
        t = d / speed
    theta = theta0 + angular_velocity * t
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)

def eta(src_x: float, src_y: float, tgt_x: float, tgt_y: float, ships: int) -> float:
    return dist(src_x, src_y, tgt_x, tgt_y) / max(fleet_speed(ships), 1e-6)

# ── Strategy helpers ─────────────────────────────────────────────────────────

def incoming_threat(planet: list, fleets: list, player_id: int) -> float:
    px, py = planet[2], planet[3]
    threat = 0.0
    for f in fleets:
        if f[1] == player_id:
            continue
        fx, fy = f[2], f[3]
        d = dist(fx, fy, px, py)
        if d > 80:
            continue
        expected_angle = math.atan2(py - fy, px - fx)
        angle_diff = abs((f[4] - expected_angle + math.pi) % (2 * math.pi) - math.pi)
        if angle_diff < 0.5:
            threat += f[6]
    return threat

def garrison_needed(planet: list, fleets: list, player_id: int) -> int:
    threat = incoming_threat(planet, fleets, player_id)
    if threat == 0:
        return MIN_GARRISON
    return int(MIN_GARRISON + threat * 1.15)

def ships_needed(src_x: float, src_y: float, target: list, launch_ships: int, angular_velocity: float) -> int:
    px, py = predict_pos(src_x, src_y, target, angular_velocity, launch_ships)
    travel_time = eta(src_x, src_y, px, py, launch_ships)
    current_ships = target[5]
    growth = target[6] * travel_time if target[1] != -1 else 0.0
    return int(current_ships + growth + SAFETY_MARGIN_SHIPS)

def target_score(src_x: float, src_y: float, target: list, comet_ids: set, total_my_production: float) -> float:
    d = dist(src_x, src_y, target[2], target[3])
    log_dist = math.log(d + 2.0)
    prod = target[6]
    ships = target[5]

    if target[0] in comet_ids:
        if d > COMET_OPPORTUNITY_DIST:
            return -999.0
        return (prod + 15.0) / log_dist

    if target[1] == -1:
        neutral_multiplier = 4.0 if total_my_production < 15.0 else 2.0
        return (prod * neutral_multiplier) / log_dist - (0.01 * ships)
    else:
        enemy_multiplier = 0.5 if total_my_production < 12.0 else 1.5
        return (prod * enemy_multiplier) / log_dist - (0.02 * ships)

def needs_reinforcement(planet: list, fleets: list, player_id: int) -> float:
    if planet[5] <= 0:
        return 0.0
    return incoming_threat(planet, fleets, player_id) / planet[5]

# ─── Core Agent Engine ────────────────────────────────────────────────────────

def agent_v3(obs, conf=None):
    if isinstance(obs, dict):
        player_id = obs.get('player', 0)
        planets   = obs.get('planets', [])
        fleets    = obs.get('fleets', [])
        ang_vel   = obs.get('angular_velocity', 0.035)
        comet_ids = set(obs.get('comet_planet_ids', []) or [])
    else:
        player_id = obs.player
        planets   = obs.planets
        fleets    = getattr(obs, 'fleets', [])
        ang_vel   = getattr(obs, 'angular_velocity', 0.035)
        comet_ids = set(getattr(obs, 'comet_planet_ids', None) or [])

    moves = []
    my_planets    = [p for p in planets if p[1] == player_id]
    enemy_planets = [p for p in planets if p[1] != player_id]

    if not my_planets or not enemy_planets:
        return moves

    total_my_production = sum(p[6] for p in my_planets)

    # ── PHASE 1: Emergency Reinforcements (Fixed Ping-Pong Loop) ─────────────────
    threatened = sorted(
        [p for p in my_planets if needs_reinforcement(p, fleets, player_id) > REINFORCE_THRESHOLD],
        key=lambda p: needs_reinforcement(p, fleets, player_id),
        reverse=True
    )
    for tgt in threatened[:1]:
        tgt_x, tgt_y = tgt[2], tgt[3]
        # FIX: A helper planet cannot be under threat itself to prevent trading loops
        helpers = [p for p in my_planets if p[0] != tgt[0] and p[5] > MIN_GARRISON + 25 and incoming_threat(p, fleets, player_id) == 0]
        if not helpers:
            continue
        best_helper = max(helpers, key=lambda p: p[5])
        bx, by = best_helper[2], best_helper[3]
        send = int((best_helper[5] - MIN_GARRISON) * 0.60)
        if send < MIN_LAUNCH_SIZE:
            continue
        angle = safe_angle(bx, by, tgt_x, tgt_y)
        if angle is not None:
            moves.append([best_helper[0], angle, send])

    # ── PHASE 2: Offensive Launches (Fixed Intercept Disconnect) ────────────────
    for src in my_planets:
        src_id = src[0]
        sx, sy = src[2], src[3]
        
        reserve = garrison_needed(src, fleets, player_id)
        max_available = src[5] - reserve
        if max_available < MIN_LAUNCH_SIZE:
            continue

        scored_targets = sorted(
            enemy_planets,
            key=lambda t: target_score(sx, sy, t, comet_ids, total_my_production),
            reverse=True,
        )

        for target in scored_targets:
            if max_available < MIN_LAUNCH_SIZE:
                break

            tx, ty = target[2], target[3]
            
            # Gauge baseline requirements
            needed = ships_needed(sx, sy, target, max_available, ang_vel)

            # Determine exact launch allocation
            if max_available >= needed:
                launch = needed
            else:
                continue  # Hold ships to consolidate a single crushing wave

            # CRITICAL FIX: Clamp the fleet size BEFORE calculating the trajectory angle
            launch = min(launch, src[5] - MIN_GARRISON)
            if launch < MIN_LAUNCH_SIZE:
                continue

            # Now calculate intercept coordinates with the exact finalized launch count
            pred_x, pred_y = predict_pos(sx, sy, target, ang_vel, launch)

            angle = safe_angle(sx, sy, pred_x, pred_y)
            if angle is None:
                continue

            moves.append([src_id, angle, launch])
            max_available -= launch
            break  # Consolidate into one high-velocity wave per turn

    return moves

def agent(obs, conf=None):
    return agent_v3(obs, conf)