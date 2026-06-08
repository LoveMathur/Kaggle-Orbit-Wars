```markdown
# Kaggle Orbit Wars - AI Strategy Sandbox

A simulation-based AI development environment for the Kaggle **Orbit Wars** challenge. The objective of this project is to build an automated, real-time strategy agent capable of managing planetary ship production, predicting orbital trajectories, and conquering competing solar systems.

---

## 🚀 Project Overview & Objectives

In **Orbit Wars**, players control a set of starter planets in a rotating solar system. Planets naturally generate ships over time based on their production rate. The core challenge requires agents to launch fleets across open space to conquer neutral and enemy planets while accounting for complex physics constraints:

* **Orbital Mechanics:** Planets continuously revolve around a central star at a constant `angular_velocity`.
* **Transit Delays:** Fleets take time to travel between coordinates. Firing directly at a planet's *current* position will result in a miss because the planet will have moved by the time the fleet arrives.
* **Resource Management:** Keeping too few ships on a planet leaves it vulnerable to interception; sending too many leaves your home base exposed.

---

## 🛠️ System Architecture & Environment Setup

This project is built using a isolated local Python environment on Fedora Linux, communicating directly with the Kaggle Environments API.

### Current Directory Tree
```text
kaggle-orbit-wars/
│
├── venv/                 # Isolated Python 3.14 virtual environment
├── .kaggle/              # Local authentication credentials (kaggle.json)
├── main.py               # Active production bot (Predictive Aiming Dev)
├── basic_agent.py        # Static baseline bot (Linear Aiming / Benchmark)
├── test_simulation.ipynb # Jupyter execution notebook for local matchmaking
└── README.md             # Project documentation

```

---

## 📈 Current Progress (What We Have Done)

We have successfully engineered the core simulation sandbox and established a local validation pipeline:

1. **Authentication & API Connectivity:** Successfully integrated Kaggle API tokens and verified connection parameters using `kaggle competitions list`.
2. **Environment Calibration:** Isolated an `AttributeError` parsing discrepancy. Configured the project to map internal Kaggle engine lists using raw index matrix mapping instead of standard object dot-notation.
3. **Data Schema Mapping:** Decoded the official environment state observation values:
* **Planets (`p`):** `[0] id`, `[1] owner`, `[2] x`, `[3] y`, `[4] radius`, `[5] ships`, `[6] production`
* **Fleets (`f`):** `[0] id`, `[1] owner`, `[2] x`, `[3] y`, `[4] angle`, `[5] from_planet_id`, `[6] ships`


4. **Baseline Implementation:** Developed `basic_agent.py`, a rule-based script using standard Euclidean distance calculations to target the nearest available enemy using standard linear firing vectors.
5. **Local Visualizer Engine:** Built a notebook validation script that executes an isolated match (`main.py` vs `basic_agent.py`) and compiles the simulation output into an interactive standalone HTML replay file (`match_replay.html`).

---

## 🧬 Core Logic: The Linear Baseline

Currently, both `main.py` and `basic_agent.py` utilize a **Linear Attack** heuristic. The core loop functions as follows:

```python
# 1. Identify owned systems
my_planets = [p for p in planets if p[1] == my_id]

# 2. Check defensive allocation thresholds 
if p_ships > 15:
    # 3. Scan for closest proximity targets
    closest_target = min(targets, key=lambda t: distance(p_x, p_y, t[2], t[3]))
    
    # 4. Calculate instantaneous angle
    angle = math.atan2(t_y - p_y, t_x - p_x)
    
    # 5. Commit 50% payload to trajectory
    moves.append([planet_id, angle, p_ships // 2])

```

---

## 🎯 Next Steps & Engineering Goals (What To Do Next)

To climb the leaderboard, `main.py` must be upgraded to outsmart the static baseline file. The immediate development roadmap consists of:

### 1. Lead-Target Trajectory Prediction (Predictive Aiming)

Instead of aiming at where a target planet *is*, compute where it *will be* upon fleet arrival.

* Calculate transit time: $\text{Time} = \frac{\text{Distance}}{\text{Fleet Speed}}$.
* Extract `angular_velocity` from the environment configuration state.
* Apply rotational matrices to project the planet's future coordinates ($X_{\text{future}}, Y_{\text{future}}$) before deploying ships.

### 2. Strategic Fleet Sizing

* Optimize the defensive threshold (currently static at `15`).
* Implement dynamic launching: check the targeted planet's ship count and production rate to send *only* the exact number of ships needed to conquer it, saving resources for defense.

### 3. Competitor Benchmarking

* Iterate on code modifications inside `main.py`.
* Execute `test_simulation.ipynb` to run local matches against `basic_agent.py`.
* Analyze `match_replay.html` to visually debug trajectory errors and optimize win rates.

```
```