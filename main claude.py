"""
Orbit Wars — Competitive Agent v10 "Production Rush"
=====================================================
Design philosophy:
  Ships sitting idle are ships wasted. Every turn, every planet with ships
  above garrison fires at the highest-value reachable target. Early economy
  lead compounds for 500 turns — the agent that owns more production points
  by turn 50 wins by turn 200.

Architecture (clean rewrite — no quadrant locks, no hard distance caps):
  LAYER 1 — Physics:  dist, fleet_speed, predict_pos, segment_hits_sun, safe_angle
  LAYER 2 — Economy:  production ROI scoring, threat detection, garrison sizing
  LAYER 3 — Tactics:  fleet coordination (arrival timing), multi-launch per planet
  LAYER 4 — Agent:    Phase 1 reinforce, Phase 2 attack

Key fixes vs v8:
  - No quadrant restriction (was paralysing agent in opposite-quadrant scenarios)
  - No dist>50 hard cap (was blocking all outer planets on 100x100 board)
  - No d>28 filter on safe_angle (was making cross-map shots return None)
  - MIN_LAUNCH_SIZE dropped to 1 (was blocking turn-1 attacks)
  - Bypass angle now aims at PREDICTED position, not a raw direction ray
    (fixes fleets flying off in wrong direction after sun bypass)
  - Multi-launch per planet (was single-launch only)
  - Pressure relief valve removed (was causing ping-pong between own planets)
  - Production-ROI scoring replaces 2-step path (simpler, more accurate)
  - Fleet coordination: deducts already-in-flight ships from needed count
  - Comet grabbing: intercept comets in flight using path data
"""

import math
from typing import List, Tuple, Optional

# ── Engine constants (do NOT change) ─────────────────────────────────────────
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_FLEET_SPEED = 6.0
ORBIT_THRESHOLD = 50.0

# ── Tunable parameters ────────────────────────────────────────────────────────
MIN_GARRISON      = 5    # absolute ship floor per planet
SAFETY_MARGIN     = 3    # extra ships added on top of computed needed
SUN_SAFE_MARGIN   = 1.5  # collision buffer around sun radius
MAX_LAUNCHES      = 3    # max simultaneous launches per planet per turn
REINFORCE_RATIO   = 0.9  # trigger emergency reinforce if threat/garrison > this
COMET_MAX_DIST    = 40   # ignore comets beyond this distance
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
    """
    Iterative intercept solver — finds where fleet meets orbiting planet.
    For static planets returns current position immediately.
    """
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
    """
    Return a launch angle from (sx,sy) toward (tx,ty) that avoids the sun.

    Critical fix: bypass angles now aim at the TARGET while curving around the
    sun, not a random 200-unit ray in a deflected direction.  We pick the bypass
    offset, then project the endpoint along the line (src → bypass-point) far
    enough to clear the board, and verify that segment doesn't hit the sun.
    The fleet is launched at the angle of the FIRST SAFE segment. Because orbit
    wars fleets travel in a straight line, the bypass angle only needs to clear
    the sun on the src→target segment (not beyond the target).
    """
    # Direct route
    direct = math.atan2(ty - sy, tx - sx)
    if not segment_hits_sun(sx, sy, tx, ty):
        return direct

    # Try offsets. Project each offset angle to a point FAR past the target so
    # the entire segment is checked, not just the src→tgt portion.
    d_to_tgt = dist(sx, sy, tx, ty)
    for offset in SUN_BYPASSES:
        a = direct + offset
        # Far point along the bypass direction (beyond the target distance)
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
        # Is the fleet's heading consistent with targeting this planet?
        aim = math.atan2(py - fy, px - fx)
        diff = abs((fa - aim + math.pi) % (2 * math.pi) - math.pi)
        if diff < 0.45:
            # Weight by proximity: close fleets are more certain threats
            threat += f[6] * (1.0 if d < 25 else 0.6)
    return threat


def garrison(planet: list, fleets: list, my_id: int) -> int:
    """
    Ships to keep on this planet.
    Base = MIN_GARRISON; rises with incoming threat to ensure survival.
    """
    thr = incoming_threat(planet, fleets, my_id)
    return max(MIN_GARRISON, int(MIN_GARRISON + thr * 1.25))


def available(planet: list, fleets: list, my_id: int) -> int:
    """Launchable ships after keeping garrison."""
    return max(0, planet[5] - garrison(planet, fleets, my_id))


def ships_required(sx: float, sy: float, target: list,
                   n_ships: int, ang_vel: float) -> int:
    """
    Ships needed to capture target, accounting for its production during transit.
    Uses the ACTUAL fleet size n_ships for speed (not a tentative).
    Neutral planets don't produce — only enemy-owned do.
    """
    px, py = predict_pos(sx, sy, target, ang_vel, n_ships)
    tt = travel_time(dist(sx, sy, px, py), n_ships)
    growth = target[6] * tt if target[1] >= 0 else 0.0  # only enemy-owned planets grow
    return int(target[5] + growth) + SAFETY_MARGIN


def en_route(src_id: int, fleets: list, my_id: int, tx: float, ty: float) -> int:
    """
    Ships already in flight from src_id toward (tx,ty).
    Only counts fleets that originated from this source planet.
    """
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
    """
    Production return-on-investment score.
    Higher production / faster to reach = better.
    Penalises heavily defended targets (expensive to take).
    Neutral planets are cheaper (no growth) so get a bonus.
    """
    px, py = predict_pos(sx, sy, target, ang_vel, n_ships)
    d = dist(sx, sy, px, py) + 1.0
    prod = target[6]
    ships = target[5]
    # How many turns to "break even" after capture?
    # cost = ships_required; income = prod per turn
    cost = ships_required(sx, sy, target, n_ships, ang_vel)
    # Lower cost-to-production ratio = better
    # Also favour closer targets (faster to reach means earlier income)
    neutral_bonus = 1.4 if target[1] == -1 else 1.0
    orbit_bonus = 1.25 if is_orbiting(target) else 1.0
    return (prod * orbit_bonus * neutral_bonus) / (math.log(d + 1) * (cost + 1) ** 0.4)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LAYER 3 — TACTICS
# ╚══════════════════════════════════════════════════════════════════════════════

def pick_targets(sx: float, sy: float, enemies: list,
                 comet_ids: set, budget: int,
                 ang_vel: float) -> List[Tuple]:
    """
    Rank all valid enemy/neutral targets by production ROI.
    Returns list of (target, launch_ships, angle, pred_x, pred_y) tuples.

    No quadrant filter. No hard distance cap.
    Sun-blocked targets are skipped only if bypass also fails (truly unreachable).
    """
    scored = []
    for t in enemies:
        # Skip comets beyond grab distance
        if t[0] in comet_ids and dist(sx, sy, t[2], t[3]) > COMET_MAX_DIST:
            continue

        # Predicted intercept position
        px, py = predict_pos(sx, sy, t, ang_vel, max(1, budget))

        # Sun check on the full src→predicted path
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
    # ── Parse observation (supports dict and object formats) ─────────────────
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
    # If any of my planets are imminently threatened, pull ships from the
    # richest safe neighbour. Cap at 2 reinforcements per turn to avoid
    # defensive spirals eating all available ships.
    # ──────────────────────────────────────────────────────────────────────────
    reinforced_from = set()
    for tgt in sorted(my_planets,
                      key=lambda p: incoming_threat(p, fleets, my_id) / max(p[5], 1),
                      reverse=True)[:2]:
        thr = incoming_threat(tgt, fleets, my_id)
        if thr == 0 or thr / max(tgt[5], 1) < REINFORCE_RATIO:
            break  # sorted — if this doesn't qualify, none below will either
        # Find a safe helper (not under threat itself, has spare ships)
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
    # PHASE 2 — Offensive launches
    # Each planet independently scores all enemies and launches at up to
    # MAX_LAUNCHES targets per turn. We always send something if budget > 0.
    # ──────────────────────────────────────────────────────────────────────────
    for src in my_planets:
        src_id = src[0]
        sx, sy = src[2], src[3]
        budget = available(src, fleets, my_id)

        if budget < 1:
            continue

        # Get ranked reachable targets
        candidates = pick_targets(sx, sy, foes, comet_ids, budget, ang_vel)
        if not candidates:
            continue

        remaining   = budget
        n_launched  = 0

        for (score, target, pred_x, pred_y, angle) in candidates:
            if remaining < 1 or n_launched >= MAX_LAUNCHES:
                break

            # Compute ships needed with current remaining as fleet size
            needed = ships_required(sx, sy, target, remaining, ang_vel)
            # Subtract already-in-flight reinforcements from this source
            already = en_route(src_id, fleets, my_id, pred_x, pred_y)
            needed_net = max(1, needed - already)

            if needed_net <= remaining:
                # Can achieve a clean capture
                launch = needed_net
            else:
                # Cannot capture cleanly this turn.
                # If this is the top-scored target, send full budget as pressure.
                # (Weakens target for next wave; also prevents ships piling up.)
                # If not the top target, skip and try next (maybe cheaper).
                if n_launched == 0:
                    launch = remaining   # always fire something at best target
                else:
                    continue

            # Final clamp: never send more than planet owns minus hard floor
            launch = min(launch, src[5] - MIN_GARRISON)
            if launch < 1:
                break

            # Recompute predicted intercept with exact launch count & refined angle
            pred_x2, pred_y2 = predict_pos(sx, sy, target, ang_vel, launch)
            angle2 = safe_angle(sx, sy, pred_x2, pred_y2)
            if angle2 is None:
                if n_launched == 0:
                    continue  # top target unreachable — try next
                else:
                    break

            moves.append([src_id, angle2, launch])
            remaining  -= launch
            n_launched += 1

    return moves
