import math

# Game constants (board center and sun radius per spec)
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_FLEET_SPEED = 6.0

def distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def fleet_speed_for_ships(num_ships, max_speed=MAX_FLEET_SPEED):
    """Compute fleet speed according to the competition spec.

    speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ** 1.5
    """
    if num_ships <= 1:
        return 1.0
    try:
        frac = math.log(min(max(num_ships, 1), 1000), 10) / math.log(1000, 10)
        # frac is log(ships)/log(1000)
        return 1.0 + (max_speed - 1.0) * (frac ** 1.5)
    except Exception:
        return 1.0

def predict_planet_position(p_x, p_y, target, angular_velocity, launch_ships, iterations=4):
    """Predict where a rotating planet will be when a fleet launched from (p_x,p_y)
    would intercept it.

    Uses the board center (50,50) as the solar center for rotation.
    Iteratively solves for travel time because fleet speed depends on the fleet size.
    """
    t_x, t_y = target[2], target[3]

    # Orbit radius around the center
    dx = t_x - CENTER_X
    dy = t_y - CENTER_Y
    r = math.hypot(dx, dy)

    # If the planet is essentially at the center (rare) or non-orbiting, return current
    if r == 0:
        return t_x, t_y

    current_theta = math.atan2(dy, dx)

    # Iteratively solve for time-to-impact t (seconds/turns)
    t = 0.0
    for _ in range(iterations):
        future_theta = current_theta + angular_velocity * t
        future_x = CENTER_X + r * math.cos(future_theta)
        future_y = CENTER_Y + r * math.sin(future_theta)

        # Estimate speed for the fleet we plan to launch
        speed = fleet_speed_for_ships(launch_ships)

        # Distance from source to predicted future position
        dist = distance(p_x, p_y, future_x, future_y)
        t = dist / max(speed, 1e-6)

    future_theta = current_theta + angular_velocity * t
    return CENTER_X + r * math.cos(future_theta), CENTER_Y + r * math.sin(future_theta)


def simple_attacker(obs, conf=None):
    moves = []

    # Handle environment object structures safely
    if isinstance(obs, dict):
        my_id = obs['player']
        planets = obs['planets']
        angular_velocity = obs.get('angular_velocity', 0.05)
    else:
        my_id = obs.player
        planets = obs.planets
        angular_velocity = getattr(obs, 'angular_velocity', 0.05)

    my_planets = [p for p in planets if p[1] == my_id]

    for planet in my_planets:
        planet_id = planet[0]
        p_x, p_y = planet[2], planet[3]
        p_ships = planet[5]

        # Keep a small garrison for defense
        min_garrison = 10

        if p_ships <= min_garrison + 1:
            continue

        # Candidate targets: neutral or enemy
        targets = [p for p in planets if p[1] != my_id]
        if not targets:
            continue

        # Choose target by a simple value heuristic (production / distance)
        def value(t):
            prod = t[6]
            d = distance(p_x, p_y, t[2], t[3])
            return - (prod / (d + 1e-6))

        target = min(targets, key=value)

        # Estimate required ships to capture at intercept time
        # Start by assuming we'll send at most available - min_garrison
        max_launch = max(0, p_ships - min_garrison)
        if max_launch <= 0:
            continue

        # We'll try sending only what's necessary
        # First estimate travel time using a placeholder launch size (e.g., half our available)
        tentative_launch = max(1, max_launch // 2)
        pred_x, pred_y = predict_planet_position(p_x, p_y, target, angular_velocity, tentative_launch)
        speed = fleet_speed_for_ships(tentative_launch)
        dist = distance(p_x, p_y, pred_x, pred_y)
        transit_time = dist / max(speed, 1e-6)

        # Compute ships on target when we arrive
        target_ships = target[5]
        target_prod = target[6]
        ships_needed = int(target_ships + target_prod * transit_time + 1)

        # If needed ships is larger than we can safely send, skip or send all-in depending on margin
        if ships_needed <= max_launch:
            launch_ships = ships_needed
        else:
            # If we can't send enough, be conservative and skip this target
            continue

        # Recompute predicted intercept using chosen launch_ships (more accurate speed)
        pred_x, pred_y = predict_planet_position(p_x, p_y, target, angular_velocity, launch_ships)
        angle = math.atan2(pred_y - p_y, pred_x - p_x)

        moves.append([planet_id, angle, launch_ships])

    return moves


def agent(obs, conf=None):
    """Kaggle submission entrypoint."""
    return simple_attacker(obs, conf)