# Pynacollada SWR + FMAToolbox-Functionality Plan

## 1. Objective

Implement buzcode-equivalent **functionality** (not MATLAB internals) in `pynacollada`, starting with:

1. John Long’s sharp wave ripple (SWR) detector behavior.
2. Ripple-related analysis utilities that do **not** duplicate detection.

All new functionality should:

1. Use `pynapple` objects and primitives (`Tsd`, `TsdFrame`, `TsGroup`, `IntervalSet`) as first-class interfaces.
2. Favor correctness first, then numerical efficiency, then pythonic API design.
3. Avoid C extensions in `pynacollada`.

## 2. Confirmed Constraints

1. Edit only `pynacollada` on `dev`.
2. Inputs for SWR detection should be `pynapple` objects only.
3. Include John’s optional training mode in the first pass.
4. Ripple-related functionality is next priority, excluding duplicate detector logic.
5. Aim for parity first; if strict parity repeatedly fails, document divergence and move forward.

## 3. Skeptical Reality Check (Important)

John’s detector code mixes:

1. Signal processing,
2. heuristic candidate generation,
3. clustering-based discrimination,
4. local statistical validation,
5. optional supervised/training diagnostics.

A literal one-shot parity implementation is high risk for regressions and hard to debug.  
Recommended approach: **deliver parity in stages with explicit test gates** rather than chasing full surface-area parity in one jump.

## 4. Pynapple-First API Policy

This project is now explicitly **pynapple-first** in API design:

1. Canonical (snake_case) APIs must return `pynapple` objects by default (`IntervalSet`, `Tsd`, `TsdFrame`, `Ts`, `TsGroup`).
2. FMAT-compatible wrappers (CamelCase) are allowed for parity and migration, but they are compatibility shells rather than the design center.
3. Manual index/mask workflows should not be the primary public interface when interval/time-series objects are available.
4. Mask/index outputs are permitted only as explicit opt-in secondary outputs (for parity testing/debugging).
5. New tests must validate object-model behavior (types and interval semantics), not only numeric equality.
6. When parity conflicts with robust `pynapple` object semantics, prefer `pynapple` semantics and document divergence.

## 5. Proposed Scope for First Development Slice

### 5.1 SWR detection module (new production code)

Create a new production module (not archive code), tentatively:

1. `pynacollada/swr.py`

Core public APIs:

1. `detect_swr_jlong(lfp: nap.TsdFrame, epochs: nap.IntervalSet, ..., training_labels: nap.IntervalSet | None = None) -> dict`
2. `detect_swr(...)` alias for migration convenience.

Expected detector outputs:

1. `events` as `nap.IntervalSet` (`start`, `end`, metadata).
2. `peaks` as `nap.Tsd` (ripple peak/trough timestamps with per-event values).
3. diagnostic arrays (`SwMax`, `RipMax`, thresholds, clustering stats).
4. `detectorinfo` with full parameter provenance.
5. optional `training` diagnostics block.

### 5.2 Optional training mode (first pass)

Training mode will accept labeled ripple intervals as `IntervalSet` and compute:

1. classifier/cluster quality summary,
2. precision/recall/F1 surfaces over threshold grids,
3. best threshold candidates and score summaries.

This is functionally aligned with John’s training intent while remaining `pynapple`-native.

### 5.3 Ripple post-detection utilities

Add utilities that consume existing ripple/SWR events without re-detecting:

1. `compute_ripple_event_stats(...)` for event-wise duration/amplitude/power/frequency summaries.
2. optional spike coupling summaries per ripple (`TsGroup` input).

These are intentionally downstream of detection and do not duplicate detector logic.

## 6. Reuse vs. New Code

### 6.1 Reuse directly from `pynapple`

1. Interval handling (`merge_close_intervals`, `drop_short_intervals`, etc.).
2. Time restriction and alignment (`restrict`, `in_interval`).
3. Core time series structures and metadata handling.

### 6.2 New in `pynacollada`

1. John-style dual-feature SWR detection pipeline.
2. Training diagnostics specific to SWR candidate features.
3. Ripple event summary functions not present in `pynapple` as SWR-specific workflows.

## 7. Performance Plan

### 7.1 Vectorization and memory

1. Use NumPy vectorized operations for filtering features and thresholding.
2. Avoid repeated copies of large arrays.
3. Work in-place where safe.

### 7.2 JIT acceleration

Use `numba` (optional runtime acceleration, pure Python fallback) for:

1. windowed feature candidate extraction loops,
2. local event-screening loops if profiling shows bottlenecks.

### 7.3 Filtering approach

Implement functionally equivalent filtering behavior for the detector pipeline, but with clean Python implementations and clear parameterization.

## 8. Testing and Validation Plan

### 8.1 Unit tests (deterministic synthetic fixtures)

1. Known injected SWR-like events detected at correct times.
2. Edge-case behavior for short/long durations, boundary intervals, close events.
3. Training mode outputs (surface shapes, metric validity, threshold selection sanity).

### 8.2 Parity-oriented tests (buzcode comparison)

1. Compare event-level outputs on selected reference recordings.
2. Compare at multiple levels:
   1. event count ranges,
   2. peak alignment tolerance,
   3. overlap-based metrics (Jaccard/F1 over intervals),
   4. key summary statistics.

### 8.3 Rigorous-but-pragmatic exit criteria

A detector slice is accepted if:

1. synthetic tests pass,
2. parity tests meet predefined tolerances on reference sessions,
3. known divergences are explicitly documented.

If strict parity fails after repeated attempts:

1. log failure mode + suspected root cause,
2. add regression test documenting current behavior,
3. continue to next milestone.

## 9. Milestones

### Milestone A: Infrastructure + API skeleton

1. Add module scaffold and typed API.
2. Add parameter object/schema and detector output schema.
3. Add initial tests scaffolding.

### Milestone B: Core SWR detector

1. Implement dual-feature extraction.
2. Implement clustering + candidate gating.
3. Implement local screening and event localization.
4. Implement peak alignment and result packaging.

### Milestone C: Training mode

1. Add labeled interval interface.
2. Add precision/recall/F1 diagnostic surfaces.
3. Add training summary outputs and tests.

### Milestone D: Ripple post-detection analytics

1. Add ripple stats utility module.
2. Add tests for event summaries and spike-coupling summaries.

### Milestone E: Validation + documentation

1. Add parity comparisons on selected datasets.
2. Document known divergences and rationale.
3. Add usage docs/examples.

## 10. Risks and Mitigations

1. **Risk:** strict parity impossible due primitive differences.  
   **Mitigation:** define measurable parity tolerances and divergence logs.

2. **Risk:** performance regressions on long sessions.  
   **Mitigation:** profile early, move hot loops to numba, avoid redundant passes.

3. **Risk:** unstable test fixtures.  
   **Mitigation:** deterministic seeds, fixed synthetic fixtures, tolerance-aware checks.

## 11. Deliverables for First Implementation Pass

1. Production SWR detector API in `pynacollada`.
2. Optional training mode integrated into the detector API.
3. Ripple post-detection stats API.
4. Unit + parity-oriented test suite with documented tolerances.
5. Divergence log template for unresolved parity mismatches.

---

This document is intentionally implementation-focused and aligned to the agreed priorities:
correctness > efficiency > pythonic design, with rigorous but pragmatic parity goals.
