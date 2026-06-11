import math
from typing import List, Tuple, Optional

# ── Board constants ──────────────────────────────────────────────────────────
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_SAFE_MARGIN = 1.2
MAX_FLEET_SPEED = 6.0
ORBIT_THRESHOLD = 50.0

# ── Tuning Knobs ─────────────────────────────────────────────────────────────
MIN_GARRISON = 3               
SAFETY_MARGIN_SHIPS = 5        
MIN_LAUNCH_SIZE = 10            
SUN_BYPASS_ANGLES = [0.22, -0.22, 0.45, -0.45, 0.6, -0.6]

# ── Physics Layer ────────────────────────────────────────────────────────────

def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

def fleet_speed(num_ships: int) -> float:
    if num_ships <= 1:
        return 1.0
    n = min(max(int(num_ships), 1), 1000)
    ratio = math.log(n) / math.log(1000)
    return 1.0 + (MAX_FLEET_SPEED - 1.0) * (ratio ** 1.5)

def is_orbiting(planet: list) -> bool:
    return dist(planet[2], planet[3], CENTER_X, CENTER_Y) + planet[4] < ORBIT_THRESHOLD

def predict_pos_at_time(target: list, angular_velocity: float, t: float) -> Tuple[float, float]:
    tx, ty = target[2], target[3]
    if not is_orbiting(target):
        return tx, ty
    dx, dy = tx - CENTER_X, ty - CENTER_Y
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return tx, ty
    theta0 = math.atan2(dy, dx)
    theta = theta0 + angular_velocity * t
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)

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
        t = dist(src_x, src_y, fx, fy) / speed
    return CENTER_X + r * math.cos(theta0 + angular_velocity * t), CENTER_Y + r * math.sin(theta0 + angular_velocity * t)

def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    r = SUN_RADIUS + SUN_SAFE_MARGIN
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return dist(x1, y1, CENTER_X, CENTER_Y) <= r
    t = max(0.0, min(1.0, ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / (dx * dx + dy * dy)))
    return dist(x1 + t * dx, y1 + t * dy, CENTER_X, CENTER_Y) <= r

def safe_angle(src_x: float, src_y: float, tgt_x: float, tgt_y: float) -> Optional[float]:
    """Direct angle to target; uses infinite ray to prevent sun overshoots and rejects distant detours."""
    direct = math.atan2(tgt_y - src_y, tgt_x - src_x)
    d = dist(src_x, src_y, tgt_x, tgt_y)
    
    # Project a 200-unit ray to ensure the line of sight behind the target doesn't intersect the sun
    far_direct_x = src_x + 200.0 * math.cos(direct)
    far_direct_y = src_y + 200.0 * math.sin(direct)
    
    if not segment_hits_sun(src_x, src_y, far_direct_x, far_direct_y):
        return direct
        
    # Deflected bypass angles drift too wide over long distances.
    # Only attempt a sun bypass if the target is close enough to guarantee interception.
    if d > 28.0:
        return None
        
    for offset in SUN_BYPASS_ANGLES:
        angle = direct + offset
        far_x = src_x + 200.0 * math.cos(angle)
        far_y = src_y + 200.0 * math.sin(angle)
        if not segment_hits_sun(src_x, src_y, far_x, far_y):
            return angle
    return None

def get_quadrant(x: float, y: float) -> int:
    return 1 if x >= CENTER_X and y >= CENTER_Y else 2 if x < CENTER_X and y >= CENTER_Y else 3 if x < CENTER_X and y < CENTER_Y else 4

# ── Strategic Sequence Layer ──────────────────────────────────────────────────

def evaluate_2_step_path(src: list, p_B: list, p_C: list, ang_vel: float, player_id: int) -> float:
    """
    Evaluates the strategic fitness of the path: Source -> Planet B -> Planet C
    Locks calculation inside spatial constraints and applies static prioritization weights.
    """
    dist_AB = dist(src[2], src[3], p_B[2], p_B[3])
    
    # HARD HORIZON: Completely ban cross-map snipes to prevent drift and wasted assets
    if dist_AB > 50.0:
        return -99999.0
        
    # Step 1 Check: Transit from Source to Planet B
    eta_B = dist_AB / max(fleet_speed(src[5]), 1e-6)
    
    # Calculate ships needed to cleanly seize B upon arrival
    growth_B = p_B[6] * eta_B if p_B[1] != -1 else 0.0
    needed_B = int(p_B[5] + growth_B + SAFETY_MARGIN_SHIPS)
    
    if (src[5] - MIN_GARRISON) < needed_B or needed_B < MIN_LAUNCH_SIZE:
        return -99999.0 # Absolute execution failure: Unaffordable

    # Step 2 Check: Virtual Projection from Planet B to Planet C
    c_future_x, c_future_y = predict_pos_at_time(p_C, ang_vel, eta_B)
    dist_BC = dist(p_B[2], p_B[3], c_future_x, c_future_y)
    
    # Calculate path value metrics
    total_yield = p_B[6] + p_C[6]
    total_time = eta_B + (dist_BC / MAX_FLEET_SPEED)
    
    # Apply Priority Knob Choice 1: Penalize revolving planets to favor stable anchors
    orbit_penalty = 1.0
    if is_orbiting(p_B): orbit_penalty += 1.5
    if is_orbiting(p_C): orbit_penalty += 1.5

    # Composite Sequence Score
    score = total_yield / (math.log(total_time + 2.0) * orbit_penalty)
    
    # Extra strategic adjustment: Prefer empty neutral spaces over fortified strongholds early on
    if p_B[1] == -1: score *= 1.5 
    
    return score

# ── Main Agent Loop ───────────────────────────────────────────────────────────

def agent_v8(obs, conf=None):
    if isinstance(obs, dict):
        player_id = obs.get('player', 0)
        planets   = obs.get('planets', [])
        ang_vel   = obs.get('angular_velocity', 0.035)
    else:
        player_id = obs.player
        planets   = obs.planets
        ang_vel   = getattr(obs, 'angular_velocity', 0.035)

    moves = []
    my_planets = [p for p in planets if p[1] == player_id]
    if not my_planets:
        return moves

    for src in my_planets:
        src_id = src[0]
        sx, sy = src[2], src[3]
        
        available = src[5] - MIN_GARRISON
        if available < MIN_LAUNCH_SIZE:
            continue

        src_quad = get_quadrant(sx, sy)
        
        # Bounded Search Tree: Prune any candidates outside our target quadrant
        quadrant_candidates = [p for p in planets if p[1] != player_id and get_quadrant(p[2], p[3]) == src_quad]
        
        # DYNAMIC QUADRANT RELEASE - Drop boundary restriction if our hemisphere is clear
        if not quadrant_candidates:
            quadrant_candidates = [p for p in planets if p[1] != player_id]

        best_immediate_target = None
        best_path_score = -99999.0

        # Execute the 2-step permutations lookahead within the bounded set
        for p_B in quadrant_candidates:
            for p_C in quadrant_candidates:
                if p_B[0] == p_C[0]:
                    if len(quadrant_candidates) > 1:
                        continue
                
                path_score = evaluate_2_step_path(src, p_B, p_C, ang_vel, player_id)
                if path_score > best_path_score:
                    best_path_score = path_score
                    best_immediate_target = p_B

        # Action Phase: Execute Step 1 of the optimal sequence discovered
        if best_immediate_target:
            dist_to_tgt = dist(sx, sy, best_immediate_target[2], best_immediate_target[3])
            eta_est = dist_to_tgt / max(fleet_speed(available), 1e-6)
            growth = best_immediate_target[6] * eta_est if best_immediate_target[1] != -1 else 0.0
            
            launch_ships = int(best_immediate_target[5] + growth + SAFETY_MARGIN_SHIPS)
            launch_ships = min(launch_ships, available)
            
            # ABSOLUTE FLOOR FILTER - Drop immediately if calculation zeroes out
            if launch_ships < MIN_LAUNCH_SIZE or launch_ships <= 0:
                continue

            # Lock physics intercept angle with the exact launch count
            pred_x, pred_y = predict_pos(sx, sy, best_immediate_target, ang_vel, launch_ships)
            angle = safe_angle(sx, sy, pred_x, pred_y)
            
            if angle is not None:
                moves.append([src_id, angle, launch_ships])
        
        # PRESSURE RELIEF VALVE - If no targets are viable, reinforce our local core anchor
        else:
            local_anchors = [p for p in my_planets if p[0] != src_id and get_quadrant(p[2], p[3]) == src_quad and not is_orbiting(p)]
            if not local_anchors:
                local_anchors = [p for p in my_planets if p[0] != src_id and not is_orbiting(p)]
            
            if local_anchors:
                # Target our single highest production core base to hold the front line
                best_anchor = max(local_anchors, key=lambda p: p[6])
                d_anchor = dist(sx, sy, best_anchor[2], best_anchor[3])
                
                # Prevent defensive fleets from executing cross-map snipes into the void
                if d_anchor > 45.0:
                    continue
                    
                angle = safe_angle(sx, sy, best_anchor[2], best_anchor[3])
                
                # Push out 50% of spare pool to balance defensive lines safely
                pool_send = int(available * 0.5)
                if angle is not None and pool_send >= MIN_LAUNCH_SIZE and pool_send > 0:
                    moves.append([src_id, angle, pool_send])

    return moves

def agent(obs, conf=None):
    return agent_v8(obs, conf)