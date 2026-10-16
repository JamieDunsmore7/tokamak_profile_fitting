# Profiling notes: `examples/cmod_shot.py`

Diagnostic run only. **No repo code was changed.** Run on 2026-10-13.

## How it was run

- **Shot:** 1100305023, `t_min=900`, `t_max=1400` ms, `mode='per_slice'`,
  `enforce_mtanh=True`, `shift_to_2pt_model=True`, `plot=True`
- **Result:** 30 Te slices and 30 ne slices fitted, saved to `saved_fits/1100305023.pkl`
- **Environment:** `~/process_shotlists/.venv/bin/python` (Python 3.12.3, numpy 2.5.1, scipy 1.18.0)
  - The repo's own `.venv` is **empty** (no numpy, eqtools, ...).
  - `~/.venv` **fails**: it has the original eqtools, which calls `scipy.mean`
    (`AttributeError: Module 'scipy' has no attribute 'mean'`). Only
    `~/process_shotlists/.venv` has the patched eqtools fork from `requirements.txt`.
- **Method:** an external driver script ran the unmodified example under
  `cProfile`, with wall-clock wrappers around the MDSplus, eqtools and fitting calls.
  It used the matplotlib `Agg` backend so `plt.show()` did not block.
- **Your in-code profiler (`fit.py` lines 115–118 and 265–268) never writes
  `fitting_only.prof`.** `master_fit` returns at line 259/261 before it reaches
  `dump_stats`, so that code is unreachable. It also stays enabled for the rest
  of the process, and on Python 3.12 an outer profiler then fails with "another
  profiling tool is already active". The driver turned it off for this run.

## Where the time goes

Process wall time was **43.8 s**: 3.5 s of imports plus **40.5 s** in `master_fit` and the save.
CPU utilisation was only 40%, so the run is mostly **waiting on I/O** (MDSplus over the network).

| Component | Wall time | % of 40.5 s | Calls | Notes |
|---|---:|---:|---:|---|
| `cmod.get_Te_sep` (2-pt model) | **30.7 s** | **76%** | 30 | ~1.0 s per time slice |
|  ↳ `CModEFITTree(...)` re-created inside it | ~13.5 s | | 30 | ~0.45 s each, rebuilds the whole equilibrium every slice |
|  ↳ `MDSplus.Tree('cmod', ...)` opens | ~7.5 s | | 60 | 2 opens per slice (`_two_point_model` + `_get_P_ohmic`), ~0.13–0.34 s each |
|  ↳ `_get_P_ohmic` | 5.1 s | | 30 | Same whole-shot signals re-read and re-smoothed every slice |
| `_plot_fit` (`plot=True`) | **7.1 s** | 18% | 60 | Mostly text/tick layout. Figures are **never closed**, which triggers matplotlib's ">20 figures open" warning, so memory grows with every slice |
| `cmod.get_thomson_data` | 1.3 s | 3% | 1 | Includes 1 EFIT load and 60 `rho2rho` calls in a per-time Python loop |
| `_fit_one_profile` / `curve_fit` | 1.3 s | 3% | 60 | ~20 ms per fit. The first guess always converged (60 `curve_fit` calls for 60 fits) |
| Import time (numpy/scipy/MDSplus/eqtools/matplotlib) | 3.5 s | n/a | 1 | Paid once per Python process, not per shot |

Totals across the run: 92 MDSplus tree opens (11.6 s) and 31 EFIT tree loads (14.0 s).

### Micro-timings (separate script)

| Operation | Time |
|---|---:|
| `MDSplus.Tree('cmod', shot)` open | 0.32–0.34 s each, **no caching between opens** |
| `CModEFITTree(shot, tree='EFIT20')` | 0.4–0.6 s each |
| `eq.rho2rho(...)` on an already-loaded equilibrium | 2–18 ms |
| Reading each 2-pt model signal (`data()` + `dim_of()`) | 7–70 ms each, ~0.15 s for all six |
| One full `get_Te_sep` call | 0.74–0.89 s |

**Conclusion:** the actual physics and fitting take about 3% of the runtime.
Most of the rest is **opening the same MDSplus trees and rebuilding the same
EFIT equilibrium again for every time slice**. All the 2-point model signals
are whole-shot time traces, so they could be read once per shot and
interpolated per slice.

## Follow-up: MDSplus access in the 2-pt model shift

The 2-pt shift has to stay **per slice**. What can change is how often the
data behind it is loaded. Every tree open and node read during the example
run was traced (`trace_opens.py` in the scratchpad).

### Tree opens for one shot (30 slices)

| Opens | Time | Tree | Opened at |
|---:|---:|---|---|
| 30 | 7.4 s | `cmod` | `devices/cmod.py:136` `_two_point_model` |
| 30 | 3.8 s | `cmod` | `devices/cmod.py:185` `_get_P_ohmic`. This is a **second, nested** open of the same tree while the first is still open |
| 30 | 0.13 s | `efit20` | `eqtools/EFIT.py:128`, a new `CModEFITTree` built in `_two_point_model` |
| 1 | 0.31 s | `cmod` | `get_thomson_data` |
| 1 | 0.21 s | `efit20` | `get_thomson_data` (EFIT object built here, then thrown away) |

**92 opens for one shot; the minimum is 2** (one `cmod` and one `efit20`).

### Node reads inside `get_Te_sep`

There are 21 distinct nodes, each read **30 times** (630 reads in total). Every one is a
**whole-shot time trace or the whole-shot EFIT equilibrium**. None of the
reads depends on the slice time, which is only used afterwards in
`np.interp` / `rho2rho`.

- 9 `cmod` signals: bolometer `main_power`, RF `pwr_net_tot`, magnetics `btor`,
  and EFIT `ssibry`, `cpasma`, `ali`, `wplasm`, `betat`, `qpsib`.
- 12 `efit20` nodes, read by eqtools every time a new `CModEFITTree` is built:
  `time`, **`psirz` (the full 2-D flux grid for every EFIT time, the largest read)**,
  `sibdry`, `simagx`, `vout`, `qpsi`, `rpres`, `volp`, `rmagx`, `zmagx`, `pasmat`, `fpol`.
  eqtools caches these on the object, but a new object per slice discards
  that cache. This is why EFIT costs about 0.45 s per slice even though opening the tree takes only about 4 ms.
- The per-slice whole-trace work (`np.gradient` of W, `smooth` + `gradient`
  for P_ohmic) also does not depend on the slice time.

### Prototype check (scratch only, `proto_2pt.py`)

The prototype is the same 2-pt formula, called once per slice, but it opens `cmod` once,
reads the 9 traces once, builds one EFIT20 object, and reuses them for every slice:

| | Time for 30 slices | Per slice |
|---|---:|---:|
| Current `get_Te_sep` | 22.2 s | 741 ms |
| Load once (0.58 s) + evaluate 30 slices (0.08 s) | **0.66 s** | **2.7 ms** |

- **About 34× faster** on the 2-pt step.
- **Te_sep agrees to within 1.4e-6 eV** (values range from 83 to 104 eV). The only difference is
  float32 rounding from dividing P_rad by 1e6 before interpolating instead of after.
  Keeping the original operation order would make the results bitwise identical.
- The EFIT20 object built in `get_thomson_data` could be passed in and reused,
  which saves another ~0.4 s per shot.

### Implemented (branch `bugfix/ggalleh/resources`, 2026-10-13)

The 2-pt data is now loaded once per shot. `cmod.load_Te_sep_inputs` reuses the EFIT object from
`get_thomson_data`, and `get_Te_sep(..., inputs=...)` is still evaluated for every slice.

- **Tree opens:** 92 → 3 per shot.
- **Example run:** 43.8 s → 10.4 s process wall time, still with `plot=True` (~7 s of it is plotting).
- **`time_window` mode:** 32.3 s → 2.6 s.
- **Results:** every output array in both modes, the aligned raw-data ψ, and
  standalone `get_Te_sep(shot, t)` calls are bitwise identical to the
  original code on shot 1100305023.

### Smaller observations

- Opening the top-level `cmod` tree takes **0.14–0.34 s**. Opening the subtrees directly
  is much cheaper: `mhd` 24 ms, `spectroscopy` 11 ms, `rf` 9 ms, `analysis` 4 ms.
  With a single open per shot this hardly matters any more.
- **Physics consistency (not speed):** the 2-pt model takes W, β_t, q95, Ip and
  li from `\top.MHD.ANALYSIS.EFIT...` (the ANALYSIS EFIT), but takes R_LCFS, B_t and
  B_p from **EFIT20** through eqtools. This may be intentional, but the two EFITs
  can differ.
- `psinorm_to_rmid` (`devices/cmod.py:223`) has the same pattern of building a new EFIT object
  on every call. It isn't used in this example.

## Speed-up opportunities (ranked; not implemented)

1. **Cache the per-shot data used by the 2-pt model (largest win, about 30 s per shot here).**
   Read P_rad, P_RF, P_ohmic, W/dW/dt, B_T, β_t and q95 traces **once per shot**,
   build the EFIT tree once, then evaluate each time slice with
   `np.interp` + `rho2rho`/`rz2BT`/`rz2BZ` (milliseconds each). Expected
   `get_Te_sep` cost: about 0.5–1 s per shot instead of ~1 s per slice.
2. **Reuse one EFIT object per shot across all device functions.**
   `get_thomson_data`, `_two_point_model`, `psinorm_to_rmid` and
   `_map_ts_to_tci` each create their own `CModEFITTree`.
3. **Use `plot=False` for batch runs** (about 7 s per shot here). If plots are
   needed, save to file and call `plt.close(fig)` to stop memory growing across many shots.
4. **Fetch once per shot, not once per window.** `master_fit` refetches Thomson
   and EFIT on every call. `~/process_shotlists/dataset.h5` has 1543 windows from
   496 shots (about 3 per shot), so calling `master_fit` once per row repeats every fetch about 3×.
   Options: fetch the whole shot's Thomson data once and slice it by window, or
   call `master_fit` once per shot over the union window in `per_slice` mode
   and assign the slices to windows afterwards.
5. **Parallelise across shots with processes** (`multiprocessing`/`concurrent.futures.ProcessPoolExecutor`,
   not threads, because the MDSplus Python bindings are not thread-safe). The machine has
   8 cores and the run is I/O-bound (40% CPU), so several workers should overlap
   the network waits. How far this scales depends on the MDSplus server load,
   so start with 4–8 workers. One long-lived worker per core also pays the
   3.5 s import cost only once.
6. Minor: vectorise the per-time `rho2rho` loop in `get_thomson_data`
   (`each_t`/array `t`). This saves under 1 s per shot. The fits themselves (~20 ms each) are not worth optimising yet.

### Rough projection (estimates, not measured)

| Scenario | Per shot (this shot's 30 slices) | 496 shots |
|---|---:|---:|
| As-is (`plot=True`) | ~40 s | ~5.5 h |
| As-is, `plot=False` | ~33 s | ~4.5 h |
| Items 1–3 (cache 2-pt + EFIT, no plots) | ~3–4 s | ~30 min serial |
| Plus 8 worker processes | n/a | probably ~5–10 min, limited by the MDSplus server |

## Inputs needed for a many-shot run

### Per shot (required)

| Input | Units / type | Notes |
|---|---|---|
| `shot` | int (10-digit C-Mod) | `> 1020000000` reads core TS from `YAG_NEW`; older shots use `YAG`. If shots come from a float array (as in `dataset.h5`), cast with `np.rint(...).astype(int)` |
| `t_min`, `t_max` | **milliseconds** | **`dataset.h5` stores `tmin`/`tmax` in seconds** (0.45–1.7), so multiply by 1000. Both ends are **exclusive** (`>`/`<`), so a TS point exactly on a boundary is dropped. Thomson runs at ~60 Hz (30 points in 500 ms), so the median 50 ms window holds only about 2–3 slices |

### Per run (global options, same for all shots)

`device='cmod'`, `mode` (`'per_slice'` or `'time_window'`), `coordinate`,
`enforce_mtanh`, `shift_to_2pt_model`, `scale_to_tci`, `plot` (use `False`),
`return_raw_data`/`return_errorbars` (`return_errorbars=True` runs **100 extra
`curve_fit`s per profile per slice**, so it will dominate runtime if turned on).

### Data that must exist in MDSplus for each shot

- **Thomson:** `\ELECTRONS::YAG_EDGETS.RESULTS:{TE,NE,RMID}` (+ errors), and core
  `YAG_NEW.RESULTS.PROFILES:{TE_RZ,NE_RZ,R_MID_T}` (+ errors). Core and edge time
  bases must match inside the window, otherwise a `ValueError` is raised.
- **EFIT:** `EFIT20`, with a **silent fallback** to `ANALYSIS`. The batch output should record
  which tree was used.
- **2-pt model (if `shift_to_2pt_model=True`):** bolometer `main_power` (fallback
  `twopi_diode`), `RF pwr_net_tot`, EFIT `WPLASM`, `betat`, `qpsib`, `SSIBRY`,
  `CPASMA`, `ali`, and magnetics `btor`. If P_sol < 0 the model **prints a
  message and returns 60 eV**. In batch mode that message gets lost, so this fallback should be
  recorded in the result.
- **Slices that are skipped automatically:** fewer than 3 core points (ψ < 0.8),
  max Te < 100 eV, or max ne < 1e18 m⁻³. With `enforce_mtanh=True`, any slice where all mtanh guesses
  fail is also dropped. Expect fewer output slices than TS times.

### Run environment

- A PSFC workstation with MDSplus access (`CMOD_TREES_PATH` → `alcdata-*` servers).
- A Python env with the **patched eqtools fork**: currently
  `~/process_shotlists/.venv`. The repo `.venv` is empty and `~/.venv` has the broken eqtools.
- A per-shot `try/except` plus a log of failures, so one bad shot doesn't stop
  the batch. Writing one pickle per shot (as the example does) makes reruns resumable.

## Other issues found while reading (not on this run's code path)

These only matter if `scale_to_tci=True`:

- `devices/cmod.py:327`: `MDSplus.Tree('analysis', shoti, ...)`. `shoti` is
  undefined, so this raises `NameError`.
- `devices/cmod.py:303`: `from scipy.integrate import cumtrapz`. This was removed in
  SciPy ≥ 1.14 (installed: 1.18), so it raises `ImportError`. Use `cumulative_trapezoid`.
- `devices/cmod.py` `_parse_yags`: `\n_dark_prior` and `\n_total` are read
  **after** the `with MDSplus.Tree(...)` block has closed the tree.

## Reproducing

The driver and micro-timing scripts are in the session scratchpad
(`profile_driver.py`, `micro.py`). To rerun:

```bash
cd <some empty dir>
~/process_shotlists/.venv/bin/python profile_driver.py   # writes cmod_shot.prof
python -m pstats cmod_shot.prof                           # or: snakeviz cmod_shot.prof
```
