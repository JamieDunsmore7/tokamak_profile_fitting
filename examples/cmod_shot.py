"""Example: fit ne and Te pedestal profiles for a C-Mod shot.

Run this script from the folder that contains the tokamak_profile_fitting/ directory:

    my_analysis/
    ├── tokamak_profile_fitting/   <- the repo
    └── cmod_shot.py               <- your script, written here

    cd my_analysis
    python cmod_shot.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokamak_profile_fitting import master_fit


shot  = 1100305023
t_min = 900    # ms
t_max = 1400   # ms

result = master_fit(
    shot               = shot,
    device             = 'cmod',
    t_min              = t_min,
    t_max              = t_max,
    mode               = 'per_slice',
    enforce_mtanh      = True,
    shift_to_2pt_model = True,
    plot               = True,
)

print(f'Fitted {len(result["te_times_ms"])} Te slices, '
      f'{len(result["ne_times_ms"])} ne slices.')

# save
os.makedirs('saved_fits', exist_ok=True)
out_path = f'saved_fits/{shot}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(result, f)
print(f'Saved to {out_path}')
