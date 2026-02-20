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
2. Ripple post-detection event statistics:
   1. per-event duration/amplitude/rms/dominant-frequency summaries
   2. optional pooled spike-count summaries
3. FMAT/buzcode-style ripple descriptive maps/stats:
   1. instantaneous ripple/frequency/phase/amplitude maps aligned to ripple peaks
   2. peak amplitude/frequency and duration outputs
   3. ripple peak autocorrelogram and pairwise summary correlations
4. Ripple spike-coupling summaries:
   1. per-unit ripple participation probability
   2. in-ripple vs out-of-ripple firing-rate modulation
   3. peri-ripple firing-rate profiles

## FMAT functionality still missing (high-value gaps)

## A. Ripple ecosystem gaps (priority now)

FMAT references:

1. `FindRipples`
2. `RippleStats`
3. related ripple QC/plotting helpers

What remains missing after current SWR detector:

1. canonical ripple detector mode that mirrors FMAT-style NSS threshold workflow (separate from John detector),
2. richer ripple quality metrics (band-specific energy ratios, asymmetry, cycle-level metrics),
3. artifact/noise rejection utilities beyond current baseline summaries,
4. standardized ripple-centric outputs suitable for downstream population analyses.

## B. FMAT `General` analytics not fully represented

High-value missing functions/classes of functionality:

1. circular-statistics suite:
   1. `CircularANOVA`
   2. `CircularRegression`
   3. `CircularConfidenceIntervals`
   4. concentration tests
2. distribution/statistical tests:
   1. `BartlettTest`, `FisherTest`, `WatsonU2Test`
   2. multinomial confidence interval tools
3. advanced sample transforms still absent as polished APIs:
   1. `AdaptiveSmooth`
   2. `DistanceTransform`
   3. `CircularShift`

## C. FMAT `Analyses` gaps (beyond what `pynapple` provides)

1. Place/phase derived analyses not fully matched by dedicated APIs:
   1. `PhasePrecession` wrappers with FMAT-like summary outputs
   2. `FieldShift`, `TestRemapping`, `TestSkewness`
2. Spike-process modeling helpers:
   1. `SelectSpikes` (burst/single discrimination)
   2. `CountSpikesPerCycle`
   3. `FitCCG`-style damped sinusoid fits
3. Behavior-specific utility analyses:
   1. `RadialMaze`, `RadialMazeTurns`
   2. `BrainStates`-adjacent summaries if not delegated elsewhere

## D. FMAT data/session abstractions not directly mapped

FMAT’s `Data` and `IO` layers include session-global state and specific loader conventions (`SetCurrentSession`, `GetSpikes`, `LoadBinaryChunk`, etc.).  
`pynapple` already has modern loaders, but FMAT-equivalent convenience wrappers are mostly absent in `pynacollada`.

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
