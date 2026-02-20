"""FMAT-style 8-arm radial-maze behavioral analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_RADIAL_COLUMNS = [
    "rat",
    "day",
    "trial",
    "configuration",
    "group",
    "ref_errors",
    "work_errors",
    "rewards_found",
    "visits",
    "performance",
    "penalty_errors",
    "adjusted_errors",
]


def _validate_radial_data(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("data must be an Nx6 matrix [rat, day, trial, configuration, group, arm].")
    out = np.asarray(np.round(arr[:, :6]), dtype=int)
    if np.any(out[:, 5] < 1) or np.any(out[:, 5] > 8):
        raise ValueError("arm values must be integers in [1, 8].")
    return out


def _baited_lookup(baited_arms: Any) -> dict[tuple[int, int], np.ndarray]:
    if isinstance(baited_arms, dict):
        out: dict[tuple[int, int], np.ndarray] = {}
        for key, value in baited_arms.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("baited_arms dict keys must be (rat, configuration) tuples.")
            arms = np.asarray(value, dtype=int).reshape(-1)
            if arms.shape[0] != 3:
                raise ValueError("Each baited arm entry must contain exactly 3 arms.")
            out[(int(key[0]), int(key[1]))] = arms
        return out

    arr = np.asarray(baited_arms)
    if arr.ndim == 3 and arr.shape[2] == 3:
        out = {}
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                out[(r + 1, c + 1)] = np.asarray(arr[r, c, :], dtype=int).reshape(3)
        return out
    if arr.ndim == 2 and arr.shape[1] >= 5:
        out = {}
        for row in np.asarray(np.round(arr), dtype=int):
            out[(int(row[0]), int(row[1]))] = np.asarray(row[2:5], dtype=int)
        return out
    raise ValueError(
        "baited_arms must be dict[(rat, config) -> 3 arms], a (n_rats,n_configs,3) array, or an Nx5 table [rat, config, arm1, arm2, arm3]."
    )


def _count_index(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(x, dtype=float) + 1.0)


def _proportion_index(x: np.ndarray) -> np.ndarray:
    p = np.asarray(x, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    return np.arcsin(np.sqrt(p)) / (np.pi / 2.0)


def _simulate_one_trial(n_wme: float, n_visits_day_one: int, rng: np.random.Generator) -> float:
    if np.isinf(n_wme):
        visits = int(max(1, n_visits_day_one))
        drawn = rng.integers(1, 9, size=visits)
        found = int(np.any(drawn == 1) + np.any(drawn == 2) + np.any(drawn == 3))
    else:
        # FMAT-inspired sequential discovery model for 3 rewards among 8 arms.
        r = rng.permutation(8) + 1
        n1 = int(np.flatnonzero(np.isin(r, [1, 2, 3]))[0] + 1)
        r = rng.permutation(8 - n1) + 1
        n2 = int(np.flatnonzero(np.isin(r, [1, 2]))[0] + 1)
        r = rng.permutation(8 - n1 - n2) + 1
        n3 = int(np.flatnonzero(r == 1)[0] + 1)
        found = 3
        visits = n1 + n2 + n3 + int(n_wme)
    return float(_proportion_index(np.array([found / visits], dtype=float))[0])


def estimate_performance_chance_levels(
    data: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    random_seed: int | None = None,
) -> dict[str, float]:
    """
    Estimate lower/upper chance levels for radial-maze performance index.
    """
    d = _validate_radial_data(data)
    visits_day_one = np.sum(d[:, 1] == np.min(d[:, 1]))
    n_rats = np.unique(d[:, 0]).shape[0]
    n_trials = np.unique(d[:, 2]).shape[0]
    n_visits_day_one = int(max(1, round(visits_day_one / max(1, n_rats * n_trials))))
    rng = np.random.default_rng(random_seed)
    vals_0 = np.array([_simulate_one_trial(0, n_visits_day_one, rng) for _ in range(int(n_bootstrap))], dtype=float)
    vals_inf = np.array([_simulate_one_trial(np.inf, n_visits_day_one, rng) for _ in range(int(n_bootstrap))], dtype=float)
    return {"p0": float(np.mean(vals_0)), "p_inf": float(np.mean(vals_inf))}


def radial_maze(
    data: np.ndarray,
    baited_arms: Any,
    *,
    estimate_chance_levels: bool = False,
    chance_bootstrap: int = 10000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Compute FMAT-style radial-maze count measures and transformed indices.

    Parameters
    ----------
    data
        Nx6 matrix `[rat, day, trial, configuration, group, arm]`.
    baited_arms
        Baited-arm definitions (3 rewarded arms per rat/config).
    """
    d = _validate_radial_data(data)
    baited = _baited_lookup(baited_arms)

    # Stable sort by task keys while preserving within-trial visit order from input.
    order = np.lexsort((np.arange(d.shape[0]), d[:, 2], d[:, 1], d[:, 3], d[:, 0]))
    d = d[order]

    rows: list[list[float]] = []
    for rat in np.unique(d[:, 0]):
        rat_rows = d[d[:, 0] == rat]
        group = int(rat_rows[0, 4])
        for conf in np.unique(rat_rows[:, 3]):
            key = (int(rat), int(conf))
            if key not in baited:
                continue
            b = np.asarray(baited[key], dtype=int).reshape(-1)
            unbaited = np.setdiff1d(np.arange(1, 9, dtype=int), b)
            conf_rows = rat_rows[rat_rows[:, 3] == conf]
            for day in np.unique(conf_rows[:, 1]):
                day_rows = conf_rows[conf_rows[:, 1] == day]
                for trial in np.unique(day_rows[:, 2]):
                    block = day_rows[day_rows[:, 2] == trial]
                    if block.shape[0] == 0:
                        continue
                    arms = block[:, 5].astype(int)
                    n_visits = int(arms.shape[0])
                    visited_baited = {int(a): False for a in b}
                    visited_unbaited = {int(a): False for a in unbaited}
                    n_ref = 0
                    n_work = 0
                    n_rewards = 0
                    for arm in arms:
                        if arm in visited_baited:
                            if visited_baited[arm]:
                                n_work += 1
                            else:
                                visited_baited[arm] = True
                                n_rewards += 1
                        elif arm in visited_unbaited:
                            if visited_unbaited[arm]:
                                n_work += 1
                            else:
                                visited_unbaited[arm] = True
                                n_ref += 1
                    perf = n_rewards / max(1, n_visits)
                    adjusted = n_ref + (3 - n_rewards)
                    penalty = 5 if n_rewards != 3 else n_ref
                    rows.append(
                        [
                            float(rat),
                            float(day),
                            float(trial),
                            float(conf),
                            float(group),
                            float(n_ref),
                            float(n_work),
                            float(n_rewards),
                            float(n_visits),
                            float(perf),
                            float(penalty),
                            float(adjusted),
                        ]
                    )

    if not rows:
        counts = np.empty((0, len(_RADIAL_COLUMNS)), dtype=float)
    else:
        counts = np.asarray(rows, dtype=float)
    indices = counts.copy()
    if indices.shape[0]:
        count_cols = [5, 6, 7, 8, 10, 11]
        indices[:, count_cols] = _count_index(indices[:, count_cols])
        indices[:, 9] = _proportion_index(indices[:, 9])

    counts_df = pd.DataFrame(counts, columns=_RADIAL_COLUMNS)
    indices_df = pd.DataFrame(indices, columns=_RADIAL_COLUMNS)
    out: dict[str, Any] = {
        "counts": counts,
        "indices": indices,
        "counts_df": counts_df,
        "indices_df": indices_df,
        "columns": _RADIAL_COLUMNS.copy(),
    }
    if estimate_chance_levels:
        out["chance_levels"] = estimate_performance_chance_levels(
            d,
            n_bootstrap=chance_bootstrap,
            random_seed=random_seed,
        )
    return out


def radial_maze_turns(
    data: np.ndarray,
    mode: str = "preferred",
    *,
    max_visits_good: int = 5,
    n_blocks: int = 3,
) -> dict[str, Any]:
    """
    Compute successive-turn-angle distributions in radial-maze trials.

    Parameters
    ----------
    data
        Nx6 matrix `[rat, day, trial, configuration, group, arm]`.
    mode
        One of:
        - `"good"`: keep trials with at most `max_visits_good` visits
        - `"preferred"`: keep trials starting in preferred arm
        - `"non-preferred"`: keep trials starting outside preferred arm
    """
    d = _validate_radial_data(data)
    mode_use = str(mode).lower()
    if mode_use not in ("good", "preferred", "non-preferred"):
        raise ValueError("mode must be 'good', 'preferred', or 'non-preferred'.")
    if max_visits_good < 2:
        raise ValueError("max_visits_good must be >= 2.")
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1.")

    order = np.lexsort((np.arange(d.shape[0]), d[:, 2], d[:, 1], d[:, 3], d[:, 0]))
    d = d[order]

    days = d[:, 1].astype(int)
    day_min = int(np.min(days))
    day_max = int(np.max(days))
    if day_max == day_min:
        blocks = np.ones(days.shape[0], dtype=int)
    else:
        blocks = np.floor((days - day_min) * n_blocks / (day_max - day_min + 1e-12)).astype(int) + 1
        blocks = np.clip(blocks, 1, n_blocks)

    entries: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    turn_angles_deg = np.arange(-3, 5, dtype=int) * 45

    for group in np.unique(d[:, 4]):
        g_rows = d[d[:, 4] == group]
        for rat in np.unique(g_rows[:, 0]):
            r_rows = g_rows[g_rows[:, 0] == rat]
            for conf in np.unique(r_rows[:, 3]):
                c_rows = r_rows[r_rows[:, 3] == conf]
                # Preferred arm = most visited first arm across this rat/config.
                trial_keys = np.column_stack((c_rows[:, 1], c_rows[:, 2]))
                _, first_idx = np.unique(trial_keys, axis=0, return_index=True)
                first_arms = c_rows[np.sort(first_idx), 5].astype(int)
                hist = np.bincount(first_arms, minlength=9)[1:9]
                preferred_arm = int(np.argmax(hist) + 1)

                for block in range(1, n_blocks + 1):
                    c_idx = np.flatnonzero((d[:, 4] == group) & (d[:, 0] == rat) & (d[:, 3] == conf) & (blocks == block))
                    if c_idx.size == 0:
                        continue
                    block_rows = d[c_idx]
                    # Collect per-trial arms, preserving order.
                    trial_pairs = np.column_stack((block_rows[:, 1], block_rows[:, 2]))
                    uniq_pairs = np.unique(trial_pairs, axis=0)
                    trial_arms: list[np.ndarray] = []
                    for day, trial in uniq_pairs:
                        seq = block_rows[(block_rows[:, 1] == day) & (block_rows[:, 2] == trial), 5].astype(int)
                        if seq.size < 2:
                            continue
                        if mode_use == "good" and seq.size > int(max_visits_good):
                            continue
                        if mode_use == "preferred" and seq[0] != preferred_arm:
                            continue
                        if mode_use == "non-preferred" and seq[0] == preferred_arm:
                            continue
                        trial_arms.append(seq)
                    if not trial_arms:
                        continue

                    # turn bin in [1..8]
                    turn_per_trial = []
                    for seq in trial_arms:
                        turns = np.mod(4 - np.diff(seq), 8)
                        turns[turns == 0] = 8
                        turn_per_trial.append(turns.astype(int))
                    max_turn = max(t.shape[0] for t in turn_per_trial)
                    dist = np.zeros((8, max_turn), dtype=float)
                    for turns in turn_per_trial:
                        for j, tbin in enumerate(turns, start=0):
                            dist[int(tbin) - 1, j] += 1.0
                    pooled.append(dist)
                    entries.append(
                        {
                            "group": int(group),
                            "rat": int(rat),
                            "configuration": int(conf),
                            "block": int(block),
                            "mode": mode_use,
                            "preferred_arm": preferred_arm,
                            "n_trials": int(len(trial_arms)),
                            "distribution": dist,
                        }
                    )

    if pooled:
        max_turn_global = max(p.shape[1] for p in pooled)
        pooled_dist = np.zeros((8, max_turn_global), dtype=float)
        for p in pooled:
            pooled_dist[:, : p.shape[1]] += p
    else:
        pooled_dist = np.zeros((8, 0), dtype=float)

    long_rows: list[dict[str, float]] = []
    for entry in entries:
        dist = entry["distribution"]
        for turn_idx in range(dist.shape[1]):
            total = float(np.sum(dist[:, turn_idx]))
            for bin_idx in range(8):
                count = float(dist[bin_idx, turn_idx])
                long_rows.append(
                    {
                        "group": float(entry["group"]),
                        "rat": float(entry["rat"]),
                        "configuration": float(entry["configuration"]),
                        "block": float(entry["block"]),
                        "turn_index": float(turn_idx + 1),
                        "angle_bin": float(bin_idx + 1),
                        "angle_deg": float(turn_angles_deg[bin_idx]),
                        "count": count,
                        "probability": count / total if total > 0 else np.nan,
                    }
                )
    table = pd.DataFrame(long_rows)
    return {
        "entries": entries,
        "table": table,
        "pooled_distribution": pooled_dist,
        "turn_angles_deg": turn_angles_deg.astype(float),
        "mode": mode_use,
    }


def RadialMaze(data: np.ndarray, baitedArms: Any) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias returning `(counts, indices)` arrays."""
    out = radial_maze(data, baitedArms)
    return out["counts"], out["indices"]


def RadialMazeTurns(data: np.ndarray, mode: str = "preferred") -> np.ndarray:
    """MATLAB-compatibility alias returning pooled turn distribution."""
    out = radial_maze_turns(data, mode=mode)
    return out["pooled_distribution"]

