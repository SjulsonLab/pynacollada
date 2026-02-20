# FMAToolbox Gap Analysis for `pynapple` + `pynacollada`

## Scope

This document focuses on **analysis functionality** from FMAToolbox that is not already available in:

1. `pynapple` core/process APIs, and
2. `pynacollada` production code.

It is intentionally not a line-by-line porting checklist. The goal is functional coverage with `pynapple`-native objects and workflows.

Archive policy for this project:

1. `pynacollada/archive/` code is treated as valid baseline implementation.
2. When archive functionality overlaps with new development, new production APIs should wrap/reuse archive code instead of re-implementing it.

## Current Coverage Snapshot

## Already covered in `pynapple` (or close equivalents)

1. Interval operations:
   1. set operations (`intersect`, `union`, `set_diff`)
   2. interval filtering (`drop_short_intervals`, `drop_long_intervals`, `merge_close_intervals`)
   3. restricting/alignment (`restrict`, `in_interval`)
2. Core filtering and spectral tools:
   1. bandpass/highpass/lowpass/bandstop
   2. PSD and FFT
   3. Morlet wavelet transform
3. Peri-event and event-triggered analyses:
   1. perievent timestamp realignment
   2. event-triggered averages
4. Spike train correlations:
   1. auto/cross/event correlograms
   2. ISI distributions
5. Tuning-curve workflows:
   1. 1D/2D/nD tuning curves
   2. mutual information

## Already covered in `pynacollada` (new production implementation)

1. John Long-style SWR detection:
   1. dual-feature pipeline (sharp-wave difference + ripple power)
   2. candidate clustering and local screening
   3. optional training diagnostics (precision/recall surfaces)
2. FMAT/buzcode-style NSS ripple detector mode:
   1. `FindRipples`-like low/high NSS threshold workflow
   2. merge + duration gating with buzcode-style output schema
   3. optional ripple-band noise-channel and EMG-based exclusion
3. Ripple post-detection event statistics:
   1. per-event duration/amplitude/rms/dominant-frequency summaries
   2. optional pooled spike-count summaries
4. FMAT/buzcode-style ripple descriptive maps/stats:
   1. instantaneous ripple/frequency/phase/amplitude maps aligned to ripple peaks
   2. peak amplitude/frequency and duration outputs
   3. ripple peak autocorrelogram and pairwise summary correlations
5. Ripple spike-coupling summaries:
   1. per-unit ripple participation probability
   2. in-ripple vs out-of-ripple firing-rate modulation
   3. peri-ripple firing-rate profiles
6. Ripple quality-control metrics:
   1. ripple/sharp-wave and ripple/broadband energy ratios
   2. waveform asymmetry and cycle-count descriptors
   3. spectral-entropy and broadband artifact z-score summaries
7. FMAT-style circular/statistical tests:
   1. `CircularANOVA`:
      1. one-way `ww`, `l2`, `lr`
      2. two-way `lr` (2x2 balanced design)
   2. `WatsonU2Test`
   3. `BartlettTest`
   4. `FisherTest`
   5. `MultinomialConfidenceIntervals`
8. FMAT-style transform helpers:
   1. `CircularShift`
   2. `AdaptiveSmooth`
   3. `DistanceTransform`
9. Spike-cycle process helpers:
   1. `SelectSpikes` burst/single discrimination
   2. `CountSpikesPerCycle`
10. Spike-process fit helper:
   1. `FitCCG`-style damped-sine fit to CCGs
11. Place/remapping analyses:
   1. `FieldShift` (linear/circular mode)
   2. `TestRemapping` bootstrap against random-remapping null
   3. `TestSkewness` distribution-change test
12. Phase-precession analysis:
   1. `PhasePrecession` wrapper with FMAT-like output schema
13. Standardized ripple-event schema:
   1. canonical `IntervalSet` + table + NWB-friendly array adapter
14. Brain-state helper:
   1. `refine_sleep_from_accel` (`refineSleepFromAccel` alias) from archive workflow
15. Radial-maze behavior analyses:
   1. `RadialMaze` core measures and transformed indices
   2. `RadialMazeTurns` turn-angle distributions (`good`/`preferred`/`non-preferred`)
16. FMAT-style binary IO helpers:
   1. `LoadBinary` and `LoadBinaryChunk` APIs for multiplexed binary files
17. FMAT-style event IO helpers:
   1. `LoadEvents` and `SaveEvents`
   2. `GetEvents` / `GetEventTypes` regex-query wrappers
18. FMAT-style parameter IO helpers:
   1. `LoadParameters` and `LoadPar` (XML-based session parameters)
19. FMAT-style LFP data helper:
   1. `GetLFP`/`get_lfp` using `pynapple` `Tsd`/`TsdFrame` outputs
20. FMAT-style session context helpers:
   1. `SetCurrentSession`/`GetCurrentSession` compatibility wrappers
   2. `GetChannels` / `GetUnits` wrappers
21. FMAT-style spike-time helpers:
   1. `LoadSpikeTimes` and `GetSpikeTimes` from `.res/.clu` files
   2. `GetSpikes` first-pass wrapper:
      1. `.spikes.cellinfo.mat` loading with per-unit metadata preservation
      2. `.res/.clu` fallback path
      3. optional mean-waveform extraction from `.dat` for `.res/.clu` sources
      4. unit-level filtering (`units`, `UID`, `region`)

## FMAT functionality still missing (high-value gaps)

## A. Ripple ecosystem gaps (priority now)

FMAT references:

1. `FindRipples`
2. `RippleStats`
3. related ripple QC/plotting helpers

What remains missing after current SWR detector:

1. (no major gaps currently in this subset)

## B. FMAT `General` analytics not fully represented

High-value missing functions/classes of functionality:

1. circular-statistics suite:
   1. (no major gaps currently in this subset)
2. distribution/statistical tests:
   1. (no major gaps currently in this subset)
3. advanced sample transforms still absent as polished APIs:
   1. (no major gaps currently in this transform subset)

## C. FMAT `Analyses` gaps (beyond what `pynapple` provides)

1. Place/phase derived analyses not fully matched by dedicated APIs:
   1. (no major gaps currently in this subset)
2. Spike-process modeling helpers:
   1. (no major gaps currently in this subset)
3. Behavior-specific utility analyses:
   1. (no major gaps currently in this subset)

## D. FMAT data/session abstractions not directly mapped

FMAT’s `Data` and `IO` layers include session-global state and specific loader conventions (`SetCurrentSession`, `GetSpikes`, `LoadBinaryChunk`, etc.).  
`pynapple` already has modern loaders; `pynacollada` now provides FMAT-compatible binary/event readers, XML parameter loading, LFP/session helpers, and spike loaders with `GetSpikes` cellinfo + `.res/.clu` support plus first-pass `.dat` waveform extraction. Remaining gaps are advanced waveform-parity pathways and additional legacy session wrappers.

## E. Plot/database ecosystems (likely non-goals)

1. FMAT `Plot` library (many figure utilities) and
2. FMAT `Database` batch DB tooling

These are likely lower-value for the current project direction and should remain out-of-scope unless explicitly requested.

## Recommended Implementation Plan

## Phase 1 (immediate): expand ripple stack without detector duplication

1. Keep John detector as the primary SWR detector.
2. Add post-detection ripple analytics:
   1. event-level QC metrics (SNR proxies, waveform asymmetry, band-ratio metrics),
   2. ripple-triggered spike summaries (per-unit and pooled),
   3. robust artifact flags (e.g., high broadband/noise-channel proxies).
3. Add standardized serialization helpers for ripple outputs (`IntervalSet` metadata schema + optional NWB-friendly adapters).

## Phase 2: FMAT `General` utility layer (only true gaps)

1. Implement circular stats APIs missing from `pynapple` with vectorized NumPy + optional numba acceleration.
2. Implement selected robust statistics tests needed by downstream modules.
3. Keep APIs pythonic, with optional compatibility aliases only when needed.

## Phase 3: FMAT `Analyses` extensions

1. Build phase-precession and field-shift/remapping utilities on top of `pynapple` tuning/perievent primitives.
2. Add spike-cycle and burst/single discriminators.
3. Add fit routines (`FitCCG`-style) only if they unblock real downstream workflows.

## Phase 4: selective data/session compatibility wrappers

1. Add minimal wrappers for high-demand session conveniences where `pynapple` alone is verbose.
2. Avoid recreating FMAT global-state patterns.

## Validation/Parity Strategy

1. Use strict parity where practical on representative fixtures.
2. If strict parity repeatedly fails due primitive-level differences:
   1. identify source of divergence,
   2. codify tolerances,
   3. document and proceed.
3. Gate each function/module by:
   1. synthetic correctness tests,
   2. parity-oriented benchmark tests,
   3. performance sanity checks.

## Prioritized Next Actions

1. Extend ripple stats to include FMAT-like ripple quality metrics and spike-coupling summaries.
2. Add a compact circular-statistics module (`CircularRegression`/`CircularConfidenceIntervals` first).
3. Add phase-precession and remapping utilities using existing `pynapple` primitives.
