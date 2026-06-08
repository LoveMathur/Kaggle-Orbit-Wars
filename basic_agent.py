import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def simple_attacker(obs, conf):
    moves = []
    
    # Handle environment structures safely
    if isinstance(obs, dict):
        my_id = obs['player']
        planets = obs['planets']
    else:
        my_id = obs.player
        planets = obs.planets
    
    # Filter for planets I own (where owner == my_id)
    # Based on schema: p[1] is the owner
    my_planets = [p for p in planets if p[1] == my_id]
    
    for planet in my_planets:
        planet_id = planet[0]
        p_x, p_y = planet[2], planet[3]  # x is index 2, y is index 3
        p_ships = planet[5]              # ships is index 5
        
        # Only attack if this planet has a decent fallback defense
        if p_ships > 15:
            # Find targets (neutral or enemy planets where owner != my_id)
            targets = [p for p in planets if p[1] != my_id]
            if not targets:
                continue
            
            # Find the closest target planet
            closest_target = min(
                targets, 
                key=lambda t: distance(p_x, p_y, t[2], t[3])
            )
            
            t_x, t_y = closest_target[2], closest_target[3]
            
            # Calculate linear angle toward the target
            angle = math.atan2(t_y - p_y, t_x - p_x)
            
            # Action schema format for Orbit Wars: [source_id, launch_angle, ship_count]
            moves.append([planet_id, angle, p_ships // 2])
            
    return moves