# Plan: Orbit Wars Winning Strategy

**TL;DR** — Implement physics-aware targeting (lead prediction) and precise fleet-sizing, then iterate with large-scale self-play and model-based search to reach top leaderboard positions. Start with quick, high-impact fixes (prediction + sizing), validate locally, then invest in opponent modeling, planning (MCTS/beam search), and RL/self-play ensembles.

## Steps

1. **Implement Lead Targeting (High impact, blocks step 2)**
   - Add `predict_planet_position(obs, target, launch_ships)` using `angular_velocity` and the fleet speed formula from the spec.
   - Aim using predicted coordinates when creating moves.

2. **Fleet Sizing & Threat Modeling (High/Medium, depends on 1)**
   - Compute `needed_ships = target.ships + target.production * transit_time + safety_margin`.
   - Track incoming enemy fleets and adapt attack/defense thresholds per planet.

3. **Local Evaluation Harness (Parallelizable)**
   - Create automated round-robin tournaments using `test_simulation.ipynb` or a new script to measure win rate vs baseline and variants.
   - Log episode replays and result statistics for analysis.

4. **Tactical Enhancements (Medium)**
   - Prioritize planets by production-weighted value and control bottlenecks (sector control).
   - Avoid sun-crossing flight paths; prefer multi-stage hops when direct path risks sun collision.

5. **Planning & Search (Medium/High)**
   - Implement short-horizon simulation-based search (beam search or MCTS) that simulates candidate launches for 3–10 turns.
   - Use heuristics (ship advantage, production gain) to score leaf states.

6. **Opponent Modeling & Self-play (High)**
   - Collect match logs, cluster opponent behaviors, and maintain simple strategy classifiers.
   - Run self-play to train/refine policies (policy gradients or PPO) or to produce data for imitation learning.

7. **Ensembling & Submission Strategy (High)**
   - Maintain several agents (prediction+heuristic, search-based, learned policy) and ensemble via a meta-controller.
   - Use frequent submissions early (up to 5/day) to get rapid skill estimate updates; lock the best performing agent for the final period.

8. **Polishing & Hardening (Low)**
   - Optimize runtime to fit `actTimeout` (<1s). Add graceful failure logging for submission validation.

## Relevant files

- `kaggle-orbit-wars/main.py` — modify agent entrypoint to add prediction + sizing
- `kaggle-orbit-wars/basic_agent.py` — baseline for comparisons
- `kaggle-orbit-wars/test_simulation.ipynb` — harness for local evaluation and replay
- `kaggle-orbit-wars/readme.md` — documentation and roadmap reference

## Verification

1. Unit test `predict_planet_position()` with deterministic planet states and known transit times.
2. Run 1000 head-to-head simulated episodes between baseline and new agent; expect +40–60% win-rate improvement from lead targeting alone.
3. After adding sizing/threat logic, re-run tournaments; expect further +10–20% improvements.
4. Submit incremental agents on Kaggle (sandboxed): 3–5/day to collect episode feedback, log failures, and download replays for analysis.

## Decisions & Assumptions

- Assume `angular_velocity` and `initial_planets` are available in each observation (per spec).
- Focus on 1v1 ladder performance first; FFA later if time permits.
- Do not attempt full-scale RL from scratch until prediction and local search are validated.

## Further Considerations

1. Compute: short-horizon search and self-play require CPU/GPU resources; start with CPU-based simulation and scale later.
2. Submission cadence: early frequent submissions speed up skill estimate convergence — use this to iterate quickly.
3. If you want, I can draft the `predict_planet_position()` implementation and a small test harness next.
