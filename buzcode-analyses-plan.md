# Buzcode Analyses Port Plan (Pynapple-First)

## 1. Objective
Port and unify analysis functionality from `buzcode/analysis`, buzcode's `externalPackages/FMAToolbox`, `pynacollada/archive`, and `pynacollada/FMA_toolbox` into `pynacollada` without duplicating capabilities that already exist in `pynapple`.

This plan is based on reading:
- `buzcode/analysis/*`
- `buzcode/externalPackages/FMAToolbox/*`
- `pynacollada/pynacollada/*`
- `pynacollada/pynacollada/archive/*`
- `pynacollada/pynacollada/FMA_toolbox/*`

## 2. Hard Constraints
1. `pynapple` is the computational core.
2. Do not re-implement functionality already available in `pynapple` (`restrict`, interval operations, filtering, correlograms, tuning curves, decoding, perievent, wavelets, spectra).
3. `buzcode/analysis`, buzcode's `FMAToolbox`, `archive`, and the existing Python `FMA_toolbox` ports are equal reference sources for analysis behavior and outputs.
4. `pynacollada` should add:
   - workflow composition,
   - buzcode-compatible schemas/aliases,
   - missing algorithms not in `pynapple`.
5. Prefer `nap.Tsd`, `nap.TsdFrame`, `nap.TsGroup`, `nap.IntervalSet` interfaces; legacy dict/mat structs are compatibility adapters only.

## 3. Audit Snapshot
- `buzcode/analysis` subdirs: 13
- MATLAB files in `analysis/`: 93 (including private/helpers)
- Top-level analysis functions to consider for porting: 79

### Subdirectory counts
| Subdir | Functions |
|---|---:|
| CrossFrequencyCoupling | 4 |
| RankOrder | 1 |
| SharpWaveRipples | 7 |
| SpectralAnalyses | 5 |
| assemblies | 1 |
| cellTypeClassification | 6 |
| lfp_general | 9 |
| monosynapticPairs | 10 |
| placeFields | 7 |
| positionDecoding | 6 |
| spikeLFPcoupling | 5 |
| spikes | 1 |
| spikes_general | 17 |

### Current coverage signal
- Name-matched in current `pynacollada`: very limited (mainly ripple stack aliases).
- Existing `pynacollada` strength is the FMAT/SWR layer (`FMA_toolbox/swr.py` + tests).
- Existing `pynacollada/FMA_toolbox` contains 14 Python modules from prior porting work and is now a fourth integration source to validate against the other three.
- `buzcode-python` already contains parity-tested ports for several dependencies (`bz_WaveSpec`, `bz_firingMap1D`, `bz_PowerSpectrumSlope`, `CCG`, `bz_SpktToSpkmat`, interval helpers), and should be used as an auxiliary validation source, not copied directly.

### Dependency hotspots in buzcode analyses
High-frequency shared dependencies from FMAT/buzcode helpers include:
- `InIntervals`, `Restrict`, `FindInInterval`
- `Sync`, `SyncMap`
- `CircularDistribution`
- `bz_WaveSpec`, `bz_Filter`, `bz_SpktToSpkmat`, `CCG`

Interpretation: porting should be dependency-layered, not folder-by-folder.

### Archive audit (`pynacollada/pynacollada/archive`)
- Python files discovered: 15
- Files that compile cleanly: 13
- Files with syntax/indentation errors: 2
  - `archive/neural_decoding/neural_decoding.py` (syntax error)
  - `archive/neural_tuning/neural_tuning.py` (indentation error)
- Production imports from archive today: only `FMA_toolbox/swr.py` importing `archive/eeg_processing/eeg_processing.py`.

| Archive module | Current state | Planned role in port |
|---|---|---|
| `brain_state_scoring/brain_state_scoring.py` | Small, usable, already superseded by `FMA_toolbox/brain_states.py` | Keep as historical reference only; no direct reuse needed. |
| `eeg_processing/eeg_processing.py` | Usable core helper code; currently imported by SWR module | Extract stable helpers into non-archive analysis module, then retire archive import. |
| `neural_ensemble/neural_ensemble.py` | Partially implemented with TODOs and brittle paths | Salvage algorithm ideas only; rewrite on pynapple-native interfaces. |
| `neural_decoding/neural_decoding.py` | Broken syntax + undefined variables | Treat as non-viable code; use buzcode/FMAT specs and pynapple decoders instead. |
| `neural_tuning/neural_tuning.py` | Broken indentation, duplicate/legacy patterns | Treat as non-viable code; use pynapple tuning stack and rebuild missing logic cleanly. |
| `graphics/graphics.py` | Stub placeholders | Do not port; keep plotting parity out of core scope. |
| `neural_crosscorr/neural_crosscorr.py` | Minimal/empty scaffold | Do not port directly. |
| `position_tracking/position_tracking.py` | Minimal scaffold | Do not port directly. |
| Tutorial scripts/notebooks in `archive/*` | Demonstration-oriented workflows | Use as behavior examples for tests/docs, not production modules. |

Archive policy:
1. `archive/` is an equal reference source for behavior/specs, but not production runtime code.
2. No new analysis module should import from `pynacollada.archive.*`.
3. Any useful archive logic must be rewritten into typed pynapple-first modules with tests before adoption.

## 4. Architecture: Three Layers
## Layer A: Native pynapple composition (preferred)
Implement analysis workflows by composing `pynapple.process` and core objects.

## Layer B: pynacollada-added algorithms
Only for missing functionality:
- ripple-specific QC maps/coupling beyond base `pynapple`,
- PAC/comodulograms,
- rank-order replay stats,
- monosynaptic significance/convolution GLM stack,
- niche cell-type/ISI modeling utilities.

## Layer C: Compatibility adapters
Thin wrappers for buzcode-style names/schemas (`bz_*`) that call A/B internals.
No separate numerical implementation in compatibility wrappers.

## 4.5 Four-Source Unification Method
For each target function/workflow:
1. Build a behavior matrix from all four reference sources (`buzcode/analysis`, buzcode `FMAToolbox`, `archive`, and Python `FMA_toolbox`) covering inputs, outputs, units, defaults, and edge cases.
2. Implement one canonical pynapple-first version in `pynacollada`.
3. Run differential tests against all available source implementations/fixtures.
4. Resolve disagreements with a documented rubric:
   - Prefer majority behavior when at least three sources agree and tests are stable.
   - If sources split `2-2` or all four differ, choose the most internally consistent, numerically stable, and scientifically interpretable behavior.
   - If two behaviors are both actively used and scientifically meaningful, expose an explicit `mode=` switch instead of silent branching.
   - When Python `FMA_toolbox` disagrees with MATLAB `FMAToolbox`, treat both as hypotheses and require test evidence before choosing either.
5. Record provenance in docs/tests for every non-trivial divergence.

## 5. Proposed Package Organization
Create a dedicated namespace in `pynacollada`:

- `pynacollada/analysis/buzcode/compat.py`
  - MATLAB-style aliases (`bz_*`) with argument normalization.
- `pynacollada/analysis/buzcode/adapters.py`
  - dict/mat <-> pynapple object conversion.
- `pynacollada/analysis/buzcode/spectral.py`
- `pynacollada/analysis/buzcode/placefields.py`
- `pynacollada/analysis/buzcode/decoding.py`
- `pynacollada/analysis/buzcode/spike_lfp.py`
- `pynacollada/analysis/buzcode/monosynaptic.py`
- `pynacollada/analysis/buzcode/isi.py`
- `pynacollada/analysis/buzcode/celltypes.py`
- `pynacollada/analysis/buzcode/rank_order.py`
- `pynacollada/analysis/buzcode/cfc.py`

Keep ripple/SWR implementations in `pynacollada/FMA_toolbox/swr.py`; expose aliases from compat layer.

## 6. Subdirectory Port Strategy (No-Duplication Policy Applied)
| Buzcode subdir | Strategy |
|---|---|
| `SharpWaveRipples` | Reuse existing `pynacollada` SWR/ripple stack. Add only missing wrappers (`bz_DetectSWR`, `bz_getRipSpikes`) and schema compatibility. |
| `SpectralAnalyses` | Use `pynapple` wavelet/spectrum/filter primitives. Add missing pieces only (`MTCoherogram` equivalent, whitening helper, event-triggered spectrogram wrapper). |
| `placeFields` | Build on `pynapple` 1D tuning curves + mutual info. Add place-field boundary/template logic not present in `pynapple`. |
| `positionDecoding` | Use `pynapple` decoders (`decode_1d`, `decode_template`, `decode_bayes`). Add buzcode-specific evaluation/cross-validation/reporting wrappers only. |
| `spikes` (`bz_PETH_Spikes`) | Use `pynapple` perievent functions; adapter layer only. |
| `spikes_general` | Reuse `pynapple` ISI/correlogram/randomization where possible. Implement only missing advanced models (`ConditionalISI`, shared-gamma fitting, replay comparison). |
| `spikeLFPcoupling` | Use `pynapple` filters/wavelets/perievent + existing circular stats. Implement coupling-specific statistics/maps not in `pynapple`. |
| `CrossFrequencyCoupling` | New algorithms (PAC/comodulogram/phase-amplitude distributions) built on `pynapple` filtered/wavelet outputs. |
| `monosynapticPairs` | Use `pynapple` crosscorrelograms as base. Implement Stark/Abeles convolution significance, manual/auto pair classification, plasticity GLM only. |
| `lfp_general` | Avoid duplicate filtering/downsampling internals. Implement CSD/eventCSD + PSS workflow as analysis-level composition. |
| `cellTypeClassification` | Implement waveform/ACG metrics and non-interactive classifiers; skip GUI-first workflows in initial phase. |
| `RankOrder` | New rank-sequence statistics module (event rank correlations, shuffle tests, cluster assignment). |
| `assemblies` | Defer to later phase; implement only if required by active workflows. |

## 7. Phased Implementation Plan
## Phase 0: Guardrails and Inventory Lock
1. Add a function registry mapping each of 79 functions to one of: `WRAP_PYNAPPLE`, `NEW_ALGORITHM`, `DEFER`.
2. Add per-function source matrix columns for `buzcode/analysis`, `FMAToolbox`, `archive`, and `FMA_toolbox` behavior coverage.
3. Add a CI check that blocks new ports tagged `WRAP_PYNAPPLE` from containing custom numerical kernels (enforce wrapper-only rule).
4. Add adapter utilities and type validators for canonical pynapple objects.
5. Add a CI lint rule preventing new `pynacollada.archive` imports (temporary allowlist for current SWR dependency only).

## Phase 1: Shared compatibility floor (thin wrappers)
1. Implement wrappers for frequently used semantics (`InIntervals`/`Restrict`/`Sync`/`SyncMap` equivalents) backed by `pynapple` APIs.
2. Normalize interval/time units and output schemas.
3. Add tests that compare wrapper outputs to direct `pynapple` calls.

## Phase 2: Highest-value workflows
1. Ripples/SWR compatibility completion (`SharpWaveRipples`).
2. Spectral + place fields + decoding pipelines (all `pynapple`-composed first).
3. Spike-LFP phase/power coupling.

## Phase 3: Connectivity and spike statistics
1. `monosynapticPairs` core (`CCG`-based significance, pair outputs).
2. `spikes_general` advanced ISI and burst/population modules.
3. Cell classification metrics (non-interactive first).

## Phase 4: Long-tail analyses
1. CFC modules.
2. Rank-order and assembly modules.
3. Optional plotting parity where scientifically necessary.

## 8. Testing and Validation
1. `pynapple-consistency tests`: wrappers must match direct `pynapple` operations.
2. `four-source differential tests`: compare outputs against available implementations/fixtures from `buzcode/analysis`, `FMAToolbox`, `archive`, and Python `FMA_toolbox` references.
3. `MATLAB parity tests`: required for `NEW_ALGORITHM` functions.
4. Reuse/adapt parity fixtures from `buzcode-python/tests/test_matlab_*_parity.py` where applicable.
5. Add workflow integration tests:
   - ripple -> spike coupling,
   - place fields -> decoding,
   - LFP spectral -> state/coupling summaries.

Acceptance gate per function:
- typed API on pynapple objects,
- tests for edge cases,
- documented tolerance or divergence reason.

## 9. Skeptical Reality Check
A literal one-function-at-a-time port of all 79 analysis functions is not the best engineering target.

Better target: port end-to-end scientific workflows, then attach compatibility aliases. This avoids freezing MATLAB-specific design constraints into `pynacollada` and keeps `pynapple` truly central.

### Recommended alternatives
1. Workflow-first milestone set (ripples, place/decoding, monosynaptic, spike-LFP) before long-tail utilities.
2. Use all four source families as reference behaviors and `buzcode-python` as supplemental test evidence; do not fork any source architecture directly into `pynacollada`.
3. Explicitly defer low-value plotting/UI parity and MATLAB-interactive tooling unless required by active analyses.

## 10. Immediate Next Actions
1. Create the 79-function registry with `WRAP_PYNAPPLE` vs `NEW_ALGORITHM` labels plus four-source coverage columns.
2. Implement a decision log template for source conflicts (consensus, chosen behavior, rationale, test evidence).
3. Implement Phase 1 compatibility floor as thin wrappers over `pynapple`.
4. Start Phase 2 with `SpectralAnalyses` + `placeFields` + `positionDecoding` as `pynapple` compositions.
5. Move SWR dependency off `archive/eeg_processing.py` into a maintained analysis module and keep output parity tests.
