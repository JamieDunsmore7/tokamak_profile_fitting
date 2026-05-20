# tokamak_profile_fitting

A Python package for fitting electron density (ne) and temperature (Te) pedestal profiles to Thomson scattering data from C-Mod and DIII-D.

## What it does

- Fits ne and Te profiles using an Osborne tanh function and/or a cubic polynomial, choosing the best fit based on reduced chi-squared
- Two fitting modes: **per_slice** (one fit per Thomson time point) or **time_window** (pool all data in a time window and fit once)
- Optional post-fit shift of the psi axis to align the separatrix Te with a 2-point model prediction
- Option to fit an exponential decay to the points in the scrape-off-layer, for density/temperature decay length analysis

## Data access requirements

**C-Mod** data is fetched directly from the MDS+ tree using the `MDSplus` Python package. You must be on a **PSFC workstation** for this to work.

**DIII-D** data is fetched using `omfit_classes`, which connects to the DIII-D MDS+ server. You must be on the **DIII-D cluster** for this to work, and `omfit_classes` must be available. Usually, this can be accessed by running `module load omfit/unstable` in the terminal at the start of the session.

## Repository structure

```
tokamak_profile_fitting/
├── fit.py                  # master_fit() — the main function you call
├── utils.py                # statistics and data cleaning helpers
├── devices/
│   ├── base.py             # instructions for adding a new device
│   ├── cmod.py             # C-Mod data fetching and 2-point model
│   └── d3d.py              # DIII-D data fetching and 2-point model
├── profiles/
│   ├── fit_functions.py    # fit functions (Osborne tanh, cubic)
│   └── sol.py              # SOL extension fitting
└── examples/
    ├── cmod_shot.py        # example C-Mod fit
    └── d3d_shot.py         # example DIII-D fit + SOL extension
```

## Running a fit

No installation is required. Clone the repo, then write your script in the same folder that contains the `tokamak_profile_fitting/` directory and run from there. The examples in `examples/` show what a complete script looks like.

```
my_analysis/
├── tokamak_profile_fitting/   <- this repo
└── my_script.py               <- your script, written here
```

```bash
cd my_analysis
python my_script.py
```

In your script:

```python
from tokamak_profile_fitting import master_fit

# Fit every Thomson time slice in a time window
result = master_fit(
    shot   = 1100305023,
    device = 'cmod',       # or 'd3d'
    t_min  = 900,          # ms
    t_max  = 1400,         # ms
    mode   = 'per_slice',  # or 'time_window' if you want to just fit once to all the data in the time window.
    enforce_mtanh      = True,
    shift_to_2pt_model = True,
    plot               = True,
)

# result is a plain dict
psi_grid    = result['psi_grid']       # output psi axis
te_profiles = result['te_profiles']    # shape (N_times, N_psi)
te_times    = result['te_times_ms']    # shape (N_times,)
ne_profiles = result['ne_profiles']
ne_times    = result['ne_times_ms']
```

### Key options

| Parameter | Default | Description |
|---|---|---|
| `mode` | `'per_slice'` | `'per_slice'` or `'time_window'` |
| `core_order` | `3` | polynomial order in the core region (0–3) |
| `sol_order` | `0` | polynomial order in the SOL (0–2) |
| `enforce_mtanh` | `True` | if False, falls back to cubic when mtanh fails |
| `shift_to_2pt_model` | `False` | shift psi so separatrix Te matches 2-point model |
| `scale_to_tci` | `False` | (C-Mod only) scale core Thomson ne to TCI interferometry |
| `plot` | `False` | plot each fit as it is produced |
| `return_raw_data` | `False` | include processed raw data in the result dict |

### SOL extension

```python
import numpy as np
from tokamak_profile_fitting import master_fit
from tokamak_profile_fitting.devices import d3d
from tokamak_profile_fitting.profiles.sol import fit_sol_double_exponential

result = master_fit(shot, 'd3d', t_min, t_max,
                    mode='time_window', return_raw_data=True)

raw = result['raw_data']
all_psi    = np.concatenate([sl['psi_te'] for sl in raw])
all_te     = np.concatenate([sl['te']     for sl in raw])
all_te_err = np.concatenate([sl['te_err'] for sl in raw])

time_ms = int(result['te_times_ms'][0])
R_wall  = d3d.get_wall_R(shot, time_ms)

x_grid, te_extended, *_ = fit_sol_double_exponential(
    shot, time_ms, 'd3d',
    result['psi_grid'], result['te_profiles'][0],
    all_psi, all_te, all_te_err,
    R_wall,
)
```
