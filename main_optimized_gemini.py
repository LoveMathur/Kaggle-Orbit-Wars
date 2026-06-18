from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.4              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- ETA-aware reinforcement risk (capture sizing) -------------------
    # Inflate the capture floor by ``reinforce_size_beta * rho(eta) * C_k`` where
    # C_k is enemy supply reachable to the target during the fleet's flight, so the
    # agent *declines* captures the enemy will reinforce mid-flight instead of
    # sinking its whole garrison into a doomed attack (the flow scorer projects
    # opponents do-nothing, so it can't see reactive reinforcement). ``beta = 0``
    # disables it (bare floor).
    reinforce_size_beta: float = 1.4
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    # --- light v5-style 4P FFA leader pressure ---------------------------
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0
    # Do not chase the leader when we are far behind; avoids 4P self-suicide.
    ffa_min_strength_ratio: float = 0.70
    # --- conservative exp50-style candidate sizing -----------------------
    # Keep (1.0,) to recover the original single-size planner exactly.
    size_multipliers: tuple[float, ...] = (1.0,)
    # --- safe terminal phase ---------------------------------------------
    # Not exp50's fixed all-in.  Late-game config changes are conditional:
    #   leading: keep the base policy; close/behind: slightly lower ROI.
    # ETA filters below also suppress attacks that arrive too late to matter.
    terminal_phase_turns: int = 60
    terminal_close_ratio: float = 0.75
    terminal_lead_ratio: float = 0.95
    terminal_close_roi_drop: float = 0.10
    terminal_behind_roi_drop: float = 0.20
    terminal_close_roi_floor: float = 1.35
    terminal_behind_roi_floor: float = 1.30
    terminal_max_waves_per_turn: int = 7
    terminal_neutral_eta_margin: float = 8.0
    terminal_enemy_eta_margin: float = 4.0
    terminal_neutral_decay_scale: float = 80.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


    trail_roi_modifier: float = 0.85
    terminal_phase_turns: int = 60
    terminal_close_ratio: float = 0.75
    terminal_behind_roi_floor: float = 1.30
    ffa_min_strength_ratio: float = 0.70


TOTAL_STEPS = 500


def _owner_strength(obs, prod: Tensor, *, player_count: int, dtype, device) -> Tensor:
    """Current owner strength proxy: production + a small ship stock term."""
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    if int(player_count) <= 0 or int(obs.P) <= 0:
        return strength
    owner = obs.owner_abs.to(device=device)
    alive = obs.alive.to(device=device)
    ships = obs.ships.to(device=device, dtype=dtype)
    prod_v = prod.to(device=device, dtype=dtype)
    for owner_id in range(int(player_count)):
        mask = alive & (owner == int(owner_id))
        if bool(mask.any()):
            strength[owner_id] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength


def _apply_safe_terminal_config(config, step, my_strength, leader_strength, player_count):
    """Applies continuous ROI scaling based on relative game position throughout the match."""
    # Calculate relative position safely
    ratio = my_strength / max(leader_strength, 1e-6)

    # Continuous signal scaling factor (Fix 1)
    if ratio < 0.35:        # Severely behind
        scale = 0.75
    elif ratio < 0.55:      # Behind
        scale = 0.85
    elif ratio < 0.75:      # Slightly behind / even
        scale = 0.95
    else:                   # Leading
        scale = 1.0         # Do not lower standards if winning

    # Apply the continuous scale factor to the base threshold
    new_roi = max(1.15, config.roi_threshold * scale)

    # Backward compatibility: Keep your original terminal phase override if in the final 60 steps
    if TOTAL_STEPS - step < config.terminal_phase_turns:
        if ratio < config.terminal_close_ratio:
            new_roi = min(new_roi, config.terminal_behind_roi_floor)

    return dataclasses.replace(config, roi_threshold=new_roi)

def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources



def _apply_ffa_leader_bonus(
    *,
    score: Tensor,
    obs,
    prod: Tensor,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    cand_is_def: Tensor,
    config: ProducerLiteConfig,
    player_id: int,
    player_count: int,
) -> Tensor:
    """Light v5-style 4P anti-leader bonus.

    Only biases offensive candidates against occupied enemy targets whose owner is
    currently stronger than us. Neutral targets and defensive candidates are
    untouched. Kept deliberately small because raw v5-style bonus did not
    reliably separate from v2 after leaderboard fit.
    """
    if int(player_count) < 4:
        return score
    leader_bonus = float(config.ffa_leader_attack_bonus)
    prod_bonus = float(config.ffa_target_prod_bonus)
    if leader_bonus == 0.0 and prod_bonus == 0.0:
        return score

    P = int(obs.P)
    if P <= 0 or target_idx.numel() == 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(player_id)

    owner = obs.owner_abs.to(device=device)
    ships = obs.ships.to(device=device, dtype=dtype)
    prod_v = prod.to(device=device, dtype=dtype)
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    for owner_id in range(int(player_count)):
        mask = obs.alive.to(device=device) & (owner == int(owner_id))
        if bool(mask.any()):
            strength[owner_id] = prod_v[mask].sum() + 0.025 * ships[mask].sum()

    if pid < 0 or pid >= int(player_count):
        return score
    my_strength = strength[pid]

    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = owner[tgt_abs].long()
    target_prod = prod_v[tgt_abs]
    occupied_enemy = (tgt_owner >= 0) & (tgt_owner != pid) & (~cand_is_def)
    owner_strength = torch.zeros_like(score)
    valid_owner = occupied_enemy & (tgt_owner < int(player_count))
    if bool(valid_owner.any()):
        owner_strength[valid_owner] = strength[tgt_owner[valid_owner]]
    leader_delta = (owner_strength - my_strength).clamp(min=0.0)
    # Anti-suicide gate: if we are much weaker than the target owner, do not
    # create a bonus that makes us trade into the leader for free.
    strong_enough = my_strength >= float(config.ffa_min_strength_ratio) * owner_strength.clamp(min=1e-6)
    bonus = leader_bonus * leader_delta + prod_bonus * target_prod
    apply_bonus = valid_owner & (leader_delta > 0.0) & strong_enough
    return score + torch.where(apply_bonus, bonus, torch.zeros_like(score))


def _apply_safe_terminal_candidate_adjustments(
    *,
    score: Tensor,
    obs,
    prod: Tensor,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    cand_is_def: Tensor,
    cand_eta: Tensor,
    config: ProducerLiteConfig,
    step: int,
    player_id: int,
) -> Tensor:
    """Suppress late attacks that cannot arrive in time and devalue late neutrals."""
    remaining = TOTAL_STEPS - int(step)
    if remaining > int(config.terminal_phase_turns):
        return score
    P = int(obs.P)
    if P <= 0 or score.numel() == 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(player_id)
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs.to(device=device)[tgt_abs].long()
    eta = cand_eta.reshape(score.shape).to(device=device, dtype=dtype)

    is_neutral = tgt_owner < 0
    is_enemy = (tgt_owner >= 0) & (tgt_owner != pid) & (~cand_is_def)

    # Defense/reinforcement candidates are never filtered here.
    too_late_neutral = is_neutral & (eta > max(1.0, float(remaining) - float(config.terminal_neutral_eta_margin)))
    too_late_enemy = is_enemy & (eta > max(1.0, float(remaining) - float(config.terminal_enemy_eta_margin)))
    invalid_late = too_late_neutral | too_late_enemy

    # Late neutral captures are less valuable because they have fewer turns to
    # repay their launch cost. Keep a floor so close, cheap neutrals can remain.
    neutral_factor = ((float(remaining) - eta) / max(1.0, float(config.terminal_neutral_decay_scale))).clamp(min=0.20, max=1.0)
    score = torch.where(is_neutral, score * neutral_factor, score)
    return torch.where(invalid_late, torch.full_like(score, float("-inf")), score)


def _tier_candidates(
    *,
    movement: PlanetMovement,
    obs,
    source_idx: Tensor,
    source_exists: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    drain: Tensor,
    floor: Tensor,
    eta_cap: Tensor,
    size_mult: float,
    S: int,
    T: int,
    pid: int,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
    device,
    dtype,
):
    """Build and score one conservative exp50-style size tier.

    Each tier computes reachability and intercept using its own fleet size. This
    avoids the v6 bug where small fleets were filtered by a max-drain precheck.
    """
    drain_floor = drain.view(S, 1).floor().clamp(min=0.0)
    raw_sizes = (drain.view(S, 1) * float(size_mult)).floor()
    sizes = torch.minimum(raw_sizes, drain_floor).clamp(min=1.0).expand(S, T)
    K = int(floor.shape[-1])

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= float(config.min_ships_to_launch)) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    score = _apply_ffa_leader_bonus(
        score=score, obs=obs, prod=prod, target_idx=target_idx,
        cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def,
        config=config, player_id=pid, player_count=int(player_count),
    )
    score = _apply_safe_terminal_candidate_adjustments(
        score=score, obs=obs, prod=prod, target_idx=target_idx,
        cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def, cand_eta=cand_eta,
        config=config, step=int(step), player_id=pid,
    )
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    return cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, score


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    step = int(obs_tensors["step"].reshape(-1)[0].item())

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    # Reachable-enemy-mass proxy ([P]) — computed ONCE and reused for BOTH the
    # reinforcement-risk floor margin (below) and the regroup gradient (further
    # down). Its decay distance-scale is the attack reach K_eta.
    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    # ETA-aware reinforcement risk: inflate the capture floor by ``beta * rho(k) *
    # reachable-enemy-mass(target)``. The per-arrival-turn growth comes from the
    # rho(k) timing ramp. Gated by beta > 0.
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]                     # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )                                                                        # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)       # [T, K_eta]
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- conservative exp50-style multi-size candidates ------------------------
    # Evaluate a small set of commit fractions per (source, target). This keeps the
    # latest-v2 reinforcement-risk floor intact while avoiding exp50's overly broad
    # (0.5, 0.75, 1.0) default. Each tier is independently screened for
    # reachability/intercept viability.
    multipliers = tuple(float(m) for m in config.size_multipliers)
    if len(multipliers) == 0:
        multipliers = (1.0,)
    tier_parts = [
        _tier_candidates(
            movement=movement,
            obs=obs,
            source_idx=source_idx,
            source_exists=source_exists,
            target_idx=target_idx,
            target_exists=target_exists,
            target_is_mine=target_is_mine,
            drain=drain,
            floor=floor,
            eta_cap=eta_cap,
            size_mult=float(mult),
            S=S,
            T=T,
            pid=pid,
            garrison_status=garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            config=config,
            player_count=int(player_count),
            step=int(step),
            device=device,
            dtype=dtype,
        )
        for mult in multipliers
    ]

    cand_src = torch.cat([p[0] for p in tier_parts], dim=0)
    cand_send = torch.cat([p[1] for p in tier_parts], dim=0)
    cand_angle = torch.cat([p[2] for p in tier_parts], dim=0)
    cand_eta = torch.cat([p[3] for p in tier_parts], dim=0)
    cand_active = torch.cat([p[4] for p in tier_parts], dim=0)
    cand_tgt_slot = torch.cat([p[5] for p in tier_parts], dim=0)
    cand_tgt_short = torch.cat([p[6] for p in tier_parts], dim=0)
    cand_is_def = torch.cat([p[7] for p in tier_parts], dim=0)
    score = torch.cat([p[8] for p in tier_parts], dim=0)

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries

    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → apply safe terminal config → plan waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    config = _apply_safe_terminal_config(
        config, obs=obs, prod=movement.planet_prod, step=step, player_count=int(player_count)
    )
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    size_multipliers=(1.0,),
    ffa_leader_attack_bonus=0.020,
    ffa_target_prod_bonus=0.045,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point with Error Logging Insulation
# ---------------------------------------------------------------------------

def agent(obs, configuration=None):
    """Single-observation entry point for local play and Kaggle."""
    try:
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        
        # Convert observation to PyTorch tensors
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        
        with torch.no_grad():
            sparse_row = _RUNTIME.tensor_action(obs_tensors)
            
        # Convert tensor selections to executable moves list
        return sparse_action_row_to_moves(sparse_row, obs=obs, player_id=player_id)
        
    except Exception as e:
        # FORCE error visibility in your test_simulation.ipynb notebook!
        import traceback
        print(f"!!! AGENT CRASH AT STEP {obs.get('step', 'unknown')} !!!")
        traceback.print_exc()
        
        # Safe fallback layout so the game engine doesn't choke
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Warmup / Tensor Shape Allocation (Fix 4)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    dummy_obs = {
        "initial_planets": [
            [0, 0, 25.0, 50.0, 3.0, 50.0, 5.0],
            [1, -1, 75.0, 50.0, 3.0, 30.0, 3.0],
            [2, 1, 50.0, 25.0, 3.0, 40.0, 4.0],
        ],
        # Pre-populate dummy fleets to force compile allocation layouts early
        "fleets": [[i, 0, 50.0, 50.0, 0.0, 0, 10] for i in range(20)],
        "step": 0,
        "player": 0,
        "angular_velocity": 0.03,
        "episode_steps": 500,
        "remainingOverageTime": 2.0,
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    with torch.no_grad():
        # Run safely through our error-insulated wrapper
        _ = agent(dummy_obs)
    _RUNTIME.reset()
except Exception as e:
    print(f"Warmup warning skipped: {e}")