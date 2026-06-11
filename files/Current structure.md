# Orbit Wars: Agent v9 & Tournament Framework Documentation

## 1. Overview

This documentation covers the architecture of **Competitive Agent v9 (The Localized Anchor Edition)** and its associated **Jupyter Notebook Tournament Harness**. 

The v9 Agent transitions away from blind, greedy map-wide snipes into a highly localized, dynamically bounded 2-step lookahead system. The testing harness procedurally generates diverse maps to accurately benchmark the agent against previous baselines without falling into cache or namespace traps.

---

## 2. `main.py` (The Agent Codebase)

### 2.1. Global Parameters & Tuning Knobs
These variables sit at the top of the file and dictate the core behavior of the physics engine and the strategic thresholds.

#### Unchangeable Engine Constants
* **`CENTER_X` / `CENTER_Y` (50.0):** Absolute center coordinates of the map.
* **`SUN_RADIUS` (10.0):** The hard collision boundary of the sun.
* **`MAX_FLEET_SPEED` (6.0):** The engine's absolute speed cap for fleets.
* **`ORBIT_THRESHOLD` (50.0):** Radius bound used to differentiate moving (outer) planets from static (inner) strongholds.

#### Changeable Tuning Knobs (Strategic Variables)
* **`MIN_GARRISON` (4):** The absolute floor of ships left behind on a source planet to defend it after a launch.
* **`SAFETY_MARGIN_SHIPS` (5):** Extra buffer ships added to any launch sequence to account for enemy production ticks during flight.
* **`MIN_LAUNCH_SIZE` (8):** The threshold preventing small "trickle" fleets. Smaller fleets travel slowly and act as free food for the enemy.
* **`SUN_SAFE_MARGIN` (1.2):** Invisible padding added around the sun to prevent rounding errors.
* **`SUN_BYPASS_ANGLES`:** Radian deflections to bend trajectories around the sun if the direct path is blocked.

---

### 2.2. The Physics Layer
These helpers simulate the *Orbit Wars* engine to predict game states before taking action.

* **`dist(x1, y1, x2, y2)`:** Calculates Euclidean distance.
* **`fleet_speed(num_ships)`:** Simulates non-linear speed scaling.
  * **Formula:**
    $$V = 1.0 + 5.0 \times \left( \frac{\log_{10}(n)}{\log_{10}(1000)} \right)^{1.5}$$
* **`predict_pos(...)`:** The Intercept Solver. Iteratively calculates where a moving fleet and a moving planet will collide. It uses 6 iterations to converge on a perfect intercept vector.
* **`safe_angle(...)`:** Determines the exact radian heading to launch a fleet. It casts an infinite 200-unit ray to ensure trajectory safety.

---

### 2.3. The Strategic Sequence Layer
* **`evaluate_2_step_path(...)`:** The core brain. Simulates a chain reaction of captures ($A \to B \to C$) rather than isolated greed.
  * **The Horizon Filter:** Rejects any $A \to B$ distance $> 50.0$, permanently preventing cross-map drift.
  * **The Temporal Shift:** Projects Planet $C$'s coordinates forward in time by the $ETA$ it will take to reach Planet $B$.
  * **Score Formula:**
    $$\text{Score} = \frac{\text{Prod}_{B} + \text{Prod}_{C}}{\ln(\text{TotalTime} + 2.0) \times \text{OrbitPenalty}}$$

---

## 3. `test_simulation.ipynb` (Tournament Harness)

This notebook mimics the Kaggle Server environment locally.

### 3.1. Key Components
1. **Module Bridging:** Uses `importlib.reload(main)` to prevent Jupyter's namespace caching, ensuring your tests run against the *latest* version of your agent.
2. **`run_benchmark(num_games)`:** * Creates a fresh `orbit_wars` environment for every game.
    * Alternates player seats (Player 1/2) to ensure unbiased evaluation.
    * Outputs win rates and tactical diagnostics.
3. **Replay Exporter:** Saves failures as `Seed [ID].html`, allowing for visual browser-based playback of specific problematic scenarios.

---

## 4. Troubleshooting & Best Practices

* **Replay Analysis:** When an agent fails, use the `TARGET_SEED` variable in the notebook to isolate that game instance.
* **Drift:** If fleets are consistently missing, check if `predict_pos` iterations are sufficient or if your `MAX_FLEET_SPEED` constants have been altered.
* **Static vs Dynamic:** Always favor static core strongholds for defensive pooling to ensure your economy remains stable while the edges of the map are contested.