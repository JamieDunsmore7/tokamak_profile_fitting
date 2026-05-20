# Adding a new device
# ====================
# Create a new file in this folder, e.g. devices/mast.py
# It must define these four functions:
#
#   get_min_errorbars()
#       Returns a dict {'ne': <float>, 'te': <float>} with absolute error floors.
#
#   get_thomson_data(shot, t_min, t_max)
#       Fetches raw Thomson data and maps it to normalised poloidal flux (psi_N).
#       Returns a dict with keys:
#           times_ms  : 1D array of Thomson times (ms)
#           psi       : 2D array (N_spatial, N_times) of psi_N values
#           ne        : 2D array (N_spatial, N_times) of electron density (m^-3)
#           ne_err    : 2D array, uncertainties on ne
#           te        : 2D array (N_spatial, N_times) of electron temperature (eV)
#           te_err    : 2D array, uncertainties on te
#           edge_mask : 1D bool array (N_spatial,), True for edge TS points
#
#   get_Te_sep(shot, time_ms)
#       Returns the 2-point model prediction for separatrix Te (eV).
#
#   psinorm_to_rmid(shot, time_ms, psi)
#       Maps a 1D array of psi_N values to outboard midplane R (m).
#       Used by the SOL fitting functions.
#
# Then pass your module to master_fit:
#
#   from tokamak_profile_fitting.devices import mast
#   result = master_fit(shot, mast, t_min=..., t_max=...)
