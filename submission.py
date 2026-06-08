import math
from statistics import mean

# --- CONSTANTS ---
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_FLEET_SPEED = 6.0
MIN_GARRISON = 10
THREAT_LOOKAHEAD = 10.0
SUN_SAFE_MARGIN = 0.5

# --- UTILITIES & PHYSICS FUNCTIONS ---
def distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def fleet_speed_for_ships(num_ships, max_speed=MAX_FLEET_SPEED):
    if num_ships <= 1:
        return 1.0
    ships = min(max(num_ships, 1), 1000)
    ratio = math.log(ships) / math.log(1000)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)

def is_orbiting_planet(planet):
    return distance(planet[2], planet[3], CENTER_X, CENTER_Y) + planet[4] < 50.0

def segment_intersects_circle(x1, y1, x2, y2, cx, cy, radius):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return distance(x1, y1, cx, cy) <= radius
    projection = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))
    closest_x = x1 + projection * dx
    closest_y = y1 + projection * dy
    return distance(closest_x, closest_y, cx, cy) <= radius

def path_crosses_sun(source_x, source_y, target_x, target_y):
    return segment_intersects_circle(
        source_x, source_y, target_x, target_y,
        CENTER_X, CENTER_Y, SUN_RADIUS + SUN_SAFE_MARGIN,
    )

def get_fleets(obs):
    if isinstance(obs, dict):
        return obs.get('fleets', []), set(obs.get('comet_planet_ids', []))
    return getattr(obs, 'fleets', []), set(getattr(obs, 'comet_planet_ids', []) or [])

# --- THREAT & PRIORITY MODELING ---
def opponent_aggression_score(planets, fleets, player_id):
    enemy_fleets = [f for f in fleets if f[1] != player_id]
    our_planets = [p for p in planets if p[1] == player_id]
    if not enemy_fleets or not our_planets:
        return 0.0
    pressured = 0.0
    for planet in our_planets:
        planet_pressure = 0.0
        for fleet in enemy_fleets:
            dist = distance(planet[2], planet[3], fleet[2], fleet[3])
            if dist <= planet[4] + THREAT_LOOKAHEAD * 2:
                planet_pressure += fleet[6] * 1.0
            elif dist <= planet[4] + THREAT_LOOKAHEAD * 4:
                planet_pressure += fleet[6] * 0.4
        pressured += planet_pressure
    return pressured / max(1, len(our_planets))

def estimate_incoming_threat(planets, fleets, player_id, planet):
    planet_x, planet_y = planet[2], planet[3]
    planet_radius = planet[4]
    threat = 0.0
    for fleet in fleets:
        if fleet[1] == player_id:
            continue
        dist = distance(fleet[2], fleet[3], planet_x, planet_y)
        if dist <= planet_radius + THREAT_LOOKAHEAD:
            threat += fleet[6] * 0.75
        elif dist <= planet_radius + 2.0 * THREAT_LOOKAHEAD:
            threat += fleet[6] * 0.35
    return threat

def launch_budget(source_planet, threat=0.0, aggression=0.0):
    ships = source_planet[5]
    production = source_planet[6]
    reserve = MIN_GARRISON + int(threat) + int(aggression * 0.25) + max(2, production)
    return max(0, ships - reserve)

def predict_planet_position(source_x, source_y, target, angular_velocity, launch_ships, iterations=4):
    target_x, target_y = target[2], target[3]
    if not is_orbiting_planet(target):
        return target_x, target_y
    orbit_dx = target_x - CENTER_X
    orbit_dy = target_y - CENTER_Y
    orbit_radius = math.hypot(orbit_dx, orbit_dy)
    if orbit_radius == 0:
        return target_x, target_y
    current_theta = math.atan2(orbit_dy, orbit_dx)
    travel_time = 0.0
    for _ in range(iterations):
        future_theta = current_theta + angular_velocity * travel_time
        future_x = CENTER_X + orbit_radius * math.cos(future_theta)
        future_y = CENTER_Y + orbit_radius * math.sin(future_theta)
        speed = fleet_speed_for_ships(launch_ships)
        travel_time = distance(source_x, source_y, future_x, future_y) / max(speed, 1e-6)
    final_theta = current_theta + angular_velocity * travel_time
    return CENTER_X + orbit_radius * math.cos(final_theta), CENTER_Y + orbit_radius * math.sin(final_theta)

def target_priority(source_x, source_y, target, comet_ids=None):
    comet_ids = comet_ids or set()
    if target[0] in comet_ids:
        return -10.0
    dist = distance(source_x, source_y, target[2], target[3])
    production = target[6]
    ships = target[5]
    owner_bonus = 0.7 if target[1] == -1 else 0.45
    orbit_bonus = 1.15 if is_orbiting_planet(target) else 1.0
    return orbit_bonus * (production + owner_bonus) / (dist + 1.0) - 0.015 * ships

# --- STRATEGIC ENGINE TARGET SELECTION ---
def candidate_launches_for_source(source, planets, fleets, player_id, angular_velocity, comet_ids):
    source_id, _, source_x, source_y, _, source_ships, _ = source
    source_threat = estimate_incoming_threat(planets, fleets, player_id, source)
    aggression = opponent_aggression_score(planets, fleets, player_id)
    max_launch = launch_budget(source, threat=source_threat, aggression=aggression)
    if max_launch <= 0:
        return []

    enemy_targets = [planet for planet in planets if planet[1] != player_id]
    ranked_targets = sorted(
        enemy_targets,
        key=lambda planet: target_priority(source_x, source_y, planet, comet_ids=comet_ids),
        reverse=True,
    )

    candidates = []
    for target in ranked_targets[:4]:
        for launch_ships in sorted({max(1, max_launch // 3), max(1, max_launch // 2), max_launch}):
            predicted_x, predicted_y = predict_planet_position(
                source_x, source_y, target, angular_velocity, launch_ships,
            )
            if path_crosses_sun(source_x, source_y, predicted_x, predicted_y):
                continue

            travel_speed = fleet_speed_for_ships(launch_ships)
            travel_time = distance(source_x, source_y, predicted_x, predicted_y) / max(travel_speed, 1e-6)
            target_threat_buffer = estimate_incoming_threat(planets, fleets, player_id, target)
            ships_needed = int(target[5] + target[6] * travel_time + target_threat_buffer + 1)

            if launch_ships >= ships_needed:
                launch_angle = math.atan2(predicted_y - source_y, predicted_x - source_x)
                candidates.append({
                    'source_id': source_id,
                    'target_id': target[0],
                    'launch_angle': launch_angle,
                    'launch_ships': ships_needed,
                    'predicted_x': predicted_x,
                    'predicted_y': predicted_y,
                    'target_priority': target_priority(source_x, source_y, target, comet_ids=comet_ids),
                })
    return candidates

def score_candidate(source, candidate, planets, fleets, player_id):
    _, _, source_x, source_y, _, source_ships, source_production = source
    ships_after_launch = source_ships - candidate['launch_ships']
    survival_score = ships_after_launch * 0.75 + source_production * 6.0

    target = next((planet for planet in planets if planet[0] == candidate['target_id']), None)
    if target is None:
        return -1e9

    target_value = target[6] * 10.0 + (22.0 - target[5]) * 0.7
    distance_penalty = distance(source_x, source_y, target[2], target[3]) * 0.03
    orbit_bonus = 4.0 if is_orbiting_planet(target) else 0.0
    threat_penalty = estimate_incoming_threat(planets, fleets, player_id, source) * 0.5
    aggression_bonus = opponent_aggression_score(planets, fleets, player_id) * 0.03

    return survival_score + target_value + orbit_bonus + candidate['target_priority'] * 12.0 - distance_penalty - threat_penalty + aggression_bonus

def choose_moves_with_lookahead(obs):
    if isinstance(obs, dict):
        player_id = obs.get('player', 0)
        planets = obs.get('planets', [])
        angular_velocity = obs.get('angular_velocity', 0.05)
    else:
        player_id = obs.player
        planets = obs.planets
        angular_velocity = getattr(obs, 'angular_velocity', 0.05)

    fleets, comet_ids = get_fleets(obs)
    moves = []
    owned_planets = [p for p in planets if p[1] == player_id]
    if not owned_planets:
        return moves

    for source in owned_planets:
        candidates = candidate_launches_for_source(source, planets, fleets, player_id, angular_velocity, comet_ids)
        if not candidates:
            continue
        best_candidate = max(candidates, key=lambda c: score_candidate(source, candidate=c, planets=planets, fleets=fleets, player_id=player_id))
        if best_candidate['launch_ships'] > 0:
            moves.append([best_candidate['source_id'], best_candidate['launch_angle'], best_candidate['launch_ships']])
    return moves

def simple_attacker(obs, conf=None):
    if isinstance(obs, dict):
        player_id = obs.get('player', 0)
        planets = obs.get('planets', [])
        angular_velocity = obs.get('angular_velocity', 0.05)
    else:
        player_id = obs.player
        planets = obs.planets
        angular_velocity = getattr(obs, 'angular_velocity', 0.05)

    moves = []
    owned_planets = [p for p in planets if p[1] == player_id]
    enemy_targets = [p for p in planets if p[1] != player_id]
    if not owned_planets or not enemy_targets:
        return moves

    for source in owned_planets:
        source_id, _, source_x, source_y, _, source_ships, _ = source
        max_launch = max(0, source_ships - MIN_GARRISON)
        if max_launch <= 0:
            continue

        ranked_targets = sorted(enemy_targets, key=lambda p: target_priority(source_x, source_y, p), reverse=True)
        for target in ranked_targets:
            tentative_launch = max(1, max_launch // 2)
            p_x, p_y = predict_planet_position(source_x, source_y, target, angular_velocity, tentative_launch)
            speed = fleet_speed_for_ships(tentative_launch)
            t_time = distance(source_x, source_y, p_x, p_y) / max(speed, 1e-6)
            ships_needed = int(target[5] + target[6] * t_time + 1)

            if ships_needed <= max_launch:
                p_x, p_y = predict_planet_position(source_x, source_y, target, angular_velocity, ships_needed)
                launch_angle = math.atan2(p_y - source_y, p_x - source_x)
                moves.append([source_id, launch_angle, ships_needed])
                break
    return moves

# --- MAIN SUBMISSION ENTRYPOINT ---
def agent(obs, conf=None):
    """
    Kaggle Submission Controller: Runs high-value predictive lookahead lookups, 
    falling back seamlessly to a solid secondary tactical bot to safeguard runTimeout windows.
    """
    try:
        primary_moves = choose_moves_with_lookahead(obs)
        if primary_moves:
            return primary_moves
    except Exception:
        pass  # Graceful fallback to avoid disqualifications
    return simple_attacker(obs, conf)