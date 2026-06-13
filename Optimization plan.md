# Optimization Project Plan: Orbit Wars Agent Tuning

This document outlines the systematic hyperparameter optimization (HPO) process to tune our V8 and V10 agents. By utilizing a fixed V7 baseline and consistent seed sets, we will ensure our benchmarking is statistically significant and reproducible.

---

## 1. Environment & Infrastructure (Colab)
To maximize computational resources and ensure persistence, we will configure the Google Colab environment:
* **Database:** A shared `sqlite:///optimization.db` file stored on Google Drive to record trial history, allowing for experiment resumption if the connection drops.
* **Logging:** A `results_log.csv` file to track `(parameters, fitness, avg_score, turns)` per trial for post-hoc convergence plotting.
* **Parallelization:** Implement `joblib` within the fitness function to parallelize game simulations across available CPU cores, significantly reducing total runtime.

---

## 2. Global Benchmarking Constraints
* **The Anchor:** **V7 (Static)** will remain the immutable baseline for all fitness evaluations.
* **The Seed Sets:**
    * **Training Set:** 80 fixed seeds (used by the optimizer to calculate fitness).
    * **Test Set:** 30 random seeds (held out to evaluate final agent generalization).
    * **Final Showdown:** 150 random seeds (used only to pit optimized V8 vs. optimized V10).
* **Fitness Function Formula:** $$\text{Fitness} = (\text{WinRate} \times 1.0) + (\text{AvgScore} \times 0.001) - (\text{AvgTurns} \times 0.0001)$$
    *(Note: The `AvgTurns` penalty discourages "turtling" or inefficient play loops.)*

---

## 3. Optimization Workflow

### Phase A: Calibration
1.  Run V7 against the 80 training seeds.
2.  Establish the "Zero Line" (average score/win rate) to assess agent improvement.

### Phase B: V8 Tuning (Bounded Agent)
* **Objective:** Optimize parameters to maximize performance against V7.
* **Search Space:**
    * `MIN_GARRISON`: Range `[2, 10]` (Integers)
    * `SAFETY_MARGIN_SHIPS`: Range `[2, 10]` (Integers)
    * `MIN_LAUNCH_SIZE`: Range `[5, 15]` (Integers)
* **Process:** Execute 100 trials using Optuna’s `TPESampler`. Use **Pruning** to kill trials early if the agent falls below a 20% win-rate threshold by turn 100.

### Phase C: V10 Tuning (Aggressive Agent)
* **Objective:** Maximize production scaling without compromising defense.
* **Search Space (Constraints applied):**
    * `MIN_GARRISON`: Range `[2, 6]`
    * `SAFETY_MARGIN_SHIPS`: Range `[2, 8]`
    * `MIN_LAUNCH_SIZE`: Range `[1, 5]`
    * `SUN_SAFE_MARGIN`: Range `[1.0, 1.5]` (Floats)
    * `PRODUCTION_ROI_WEIGHT`: Range `[0.5, 2.0]` (Floats)
    * `EN_ROUTE_BUFFER`: Range `[0, 5]` (Integers)
* **Process:** Warm-start the optimization using the best parameters found in the V8 study where applicable.

---

## 4. Final Showdown: V8 vs. V10
Once both models are optimized, we will conduct the final tournament:
1.  **Head-to-Head:** 150 unique, unseen seeds.
2.  **Seat Alternation:** Every seed is played twice (V8 as Player 1, V10 as Player 1) to negate starting position bias.
3.  **Visualization:** Generate a final performance report using `optuna.visualization.plot_optimization_history(study)` and `plot_parallel_coordinate(study)` to understand which parameters truly drive success.

---

## 5. Critical Technical Interventions
* **Timeout Guard:** Any agent game exceeding 600 turns will be automatically terminated and marked as a loss to prevent infinite simulation loops.
* **Warm-Start:** If a study terminates, Optuna will resume from the SQLite database rather than restarting, preserving your Google Colab compute hours.
* **Constraint Checking:** For every trial, the model will validate parameters against the physics engine constraints (e.g., ensuring `SAFETY_MARGIN_SHIPS` > 0) to avoid `ValueError` crashes.

---