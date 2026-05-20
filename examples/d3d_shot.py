"""Example: fit ne and Te pedestal profiles for a DIII-D shot,
then compare three SOL extension methods.

Run this script from the folder that contains the tokamak_profile_fitting/ directory:

    my_analysis/
    ├── tokamak_profile_fitting/   <- the repo
    └── d3d_shot.py                <- your script, written here

    cd my_analysis
    python d3d_shot.py

geqdsk equilibrium files are cached in ./temporary_eqdsk_files/.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokamak_profile_fitting import master_fit
from tokamak_profile_fitting.devices import d3d
from tokamak_profile_fitting.profiles.sol import (
    fit_sol_exponential_decay,
    fit_sol_double_exponential,
    fit_sol_exponential_francesco,
)


shot  = 189627
t_min = 2000   # ms
t_max = 3000   # ms

# ---- pedestal fit ----
result = master_fit(
    shot            = shot,
    device          = 'd3d',
    t_min           = t_min,
    t_max           = t_max,
    mode            = 'time_window',
    sol_order       = 1,
    enforce_mtanh   = True,
    shift_to_2pt_model = True,
    return_raw_data = True,
    plot            = False,
)

psi_grid   = result['psi_grid']
te_profile = result['te_profiles'][0]
ne_profile = result['ne_profiles'][0]

# ---- plot 1: mtanh pedestal fits in psi space ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

raw = result['raw_data']
all_psi_te = np.concatenate([sl['psi_te'] for sl in raw])
all_te     = np.concatenate([sl['te']     for sl in raw])
all_psi_ne = np.concatenate([sl['psi_ne'] for sl in raw])
all_ne     = np.concatenate([sl['ne']     for sl in raw])

ax1.scatter(all_psi_te, all_te, color='red', alpha=0.1, s=10, label='Raw data')
ax1.plot(psi_grid, te_profile, color='black', linewidth=2, label='mtanh fit')
ax1.axvline(1.0, color='grey', linestyle='--')
ax1.set_xlabel(r'$\psi_N$')
ax1.set_ylabel('Te (eV)')
ax1.legend()
ax1.grid(linestyle='--', alpha=0.3)

ax2.scatter(all_psi_ne, all_ne, color='blue', alpha=0.1, s=10, label='Raw data')
ax2.plot(psi_grid, ne_profile, color='black', linewidth=2, label='mtanh fit')
ax2.axvline(1.0, color='grey', linestyle='--')
ax2.set_xlabel(r'$\psi_N$')
ax2.set_ylabel('ne (m$^{-3}$)')
ax2.legend()
ax2.grid(linestyle='--', alpha=0.3)

plt.suptitle(f'DIII-D #{shot}, {t_min}–{t_max} ms')
plt.tight_layout()
plt.show()

# ---- SOL fits ----
time_ms = int(result['te_times_ms'][0])
R_wall  = d3d.get_wall_R(shot, time_ms)
R_lcfs  = float(d3d.psinorm_to_rmid(shot, time_ms, np.array([1.0])))
R_lim   = R_lcfs + 2/3 * (R_wall - R_lcfs)  # Francesco convention

all_te_err = np.concatenate([sl['te_err'] for sl in raw])

x1, te_exp,        *_ = fit_sol_exponential_decay(
    shot, time_ms, 'd3d',
    psi_grid, te_profile,
    all_psi_te, all_te, all_te_err,
    R_wall,
)

x2, te_double_exp, *_ = fit_sol_double_exponential(
    shot, time_ms, 'd3d',
    psi_grid, te_profile,
    all_psi_te, all_te, all_te_err,
    R_wall,
)

x3, te_francesco,  *_ = fit_sol_exponential_francesco(
    shot, time_ms, 'd3d',
    psi_grid, te_profile,
    all_psi_te, all_te, all_te_err,
    R_wall, R_lim, 'Te',
)

# convert pedestal fit to R-Rsep for the plot
Rmid_grid  = d3d.psinorm_to_rmid(shot, time_ms, psi_grid)
R_Rsep_grid = Rmid_grid - R_lcfs

# ---- plot 2: pedestal + three SOL options ----
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(all_psi_te - 1.0, all_te, color='red', alpha=0.1, s=10, label='Raw data')
ax.plot(R_Rsep_grid, te_profile, color='black', linewidth=2, label='mtanh fit')
ax.plot(x1[x1 >= 0], te_exp[x1 >= 0],        color='steelblue',  linewidth=2, label='Single exponential')
ax.plot(x2[x2 >= 0], te_double_exp[x2 >= 0],  color='darkorange', linewidth=2, label='Double exponential')
ax.plot(x3[x3 >= 0], te_francesco[x3 >= 0],   color='forestgreen',linewidth=2, label='Francesco style')

ax.axvline(0, color='grey', linestyle='--', label='Separatrix')
ax.set_xlabel('R - R$_{sep}$ (m)')
ax.set_ylabel('Te (eV)')
ax.set_title(f'DIII-D #{shot} — pedestal + SOL fits')
ax.legend()
ax.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()