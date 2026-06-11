"""DIII-D device module.

Provides the four functions that master_fit expects from any device module:
  get_thomson_data(shot, t_min, t_max)
  get_Te_sep(shot, time_ms)
  psinorm_to_rmid(shot, time_ms, psi)
  get_min_errorbars()

geqdsk equilibrium files are cached locally in _GEQDSK_DIR.
"""

import os
import glob
import shutil

import numpy as np
import eqtools
from eqtools import eqdskreader
from scipy.interpolate import interp1d

import omfit_classes.omfit_mds as omds
from omfit_classes.omfit_eqdsk import OMFITgeqdsk, from_mds_plus


MIN_NE_ERRORBAR = 2e18   # m^-3
MIN_TE_ERRORBAR = 10.0   # eV

_GEQDSK_DIR = 'temporary_eqdsk_files'


def get_min_errorbars():
    return {'ne': MIN_NE_ERRORBAR, 'te': MIN_TE_ERRORBAR}


# ---------------------------------------------------------------------------
# Thomson data
# ---------------------------------------------------------------------------

def get_thomson_data(shot, t_min, t_max):
    """Fetch DIII-D core Thomson data and map to normalised poloidal flux.

    Returns a dict with keys:
      times_ms  (N_t,)
      psi       (N_spatial, N_t)
      ne        (N_spatial, N_t)  [m^-3]
      ne_err    (N_spatial, N_t)
      te        (N_spatial, N_t)  [eV]
      te_err    (N_spatial, N_t)
      edge_mask (N_spatial,) bool  — all False for DIII-D (single TS system)
    """
    te_node     = omds.OMFITmdsValue(server='DIII-D', TDI='TSTE_CORE',   shot=shot)
    te_err_node = omds.OMFITmdsValue(server='DIII-D', TDI='TSTE_E_CORE', shot=shot)
    ne_node     = omds.OMFITmdsValue(server='DIII-D', TDI='TSNE_CORE',   shot=shot)
    ne_err_node = omds.OMFITmdsValue(server='DIII-D', TDI='TSNE_E_CORE', shot=shot)

    te_raw     = te_node.data()
    te_err_raw = te_err_node.data()
    ne_raw     = ne_node.data()
    ne_err_raw = ne_err_node.data()

    z_array   = ne_node.dim_of(1)
    times_raw = ne_node.dim_of(0)   # already in ms for DIII-D

    # sort so smallest z (closest to core) is first
    z_order    = np.argsort(z_array)
    z_array    = z_array[z_order]
    te_raw     = te_raw[z_order, :]
    te_err_raw = te_err_raw[z_order, :]
    ne_raw     = ne_raw[z_order, :]
    ne_err_raw = ne_err_raw[z_order, :]

    times_ms = np.round(times_raw).astype(int)
    mask     = (times_ms > t_min) & (times_ms < t_max)
    times_ms   = times_ms[mask]
    te_raw     = te_raw[:, mask]
    te_err_raw = te_err_raw[:, mask]
    ne_raw     = ne_raw[:, mask]
    ne_err_raw = ne_err_raw[:, mask]

    # download equilibria for the full time window in one go
    _download_geqdsk(shot, times_ms)

    R_ts  = 1.94   # hardcoded R of the DIII-D core TS system (m)
    n_z   = len(z_array)
    n_t   = len(times_ms)
    R_list = [R_ts] * n_z

    psi    = np.zeros((n_z, n_t))
    ne     = np.zeros_like(psi)
    ne_err = np.zeros_like(psi)
    te     = np.zeros_like(psi)
    te_err = np.zeros_like(psi)

    for t_idx, time_ms in enumerate(times_ms):
        try:
            eq      = _load_equilibrium(shot, time_ms)
            raw_psi = eq.rho2rho('RZ', 'psinorm', R_list, z_array, time_ms)[0]
        except Exception as exc:
            print(f'  t={time_ms} ms: equilibrium failed ({exc}), skipping')
            psi[:, t_idx] = np.nan
            continue

        psi[:, t_idx]     = raw_psi
        ne[:, t_idx]      = ne_raw[:, t_idx]
        ne_err[:, t_idx]  = ne_err_raw[:, t_idx]
        te[:, t_idx]      = te_raw[:, t_idx]
        te_err[:, t_idx]  = te_err_raw[:, t_idx]

    edge_mask = np.zeros(n_z, dtype=bool)   # DIII-D has no separate edge system

    return dict(times_ms=times_ms, psi=psi, ne=ne, ne_err=ne_err,
                te=te, te_err=te_err, edge_mask=edge_mask)


# ---------------------------------------------------------------------------
# Separatrix Te (2-point model, Leonard 2017 lambda_q scaling)
# ---------------------------------------------------------------------------

def get_Te_sep(shot, time_ms):
    """Leonard-2017 lambda_q 2-point model for the DIII-D separatrix Te (eV)."""
    time_ms = int(round(time_ms))
    eq      = _load_equilibrium(shot, time_ms)
    t       = time_ms   # EqdskReader ignores this but the API requires it

    R_lcfs = float(eq.rho2rho('psinorm', 'Rmid', 1, t))
    Bt     = abs(float(eq.rz2BT(R_lcfs, 0, t)))
    Bp     = abs(float(eq.rz2BZ(R_lcfs, 0, t)))
    q95    = float(np.nanmean(eq.getQ95()))

    P_tot_node = omds.OMFITmdsValue(server='DIII-D', TDI='PTOT', shot=shot)
    wdot_node  = omds.OMFITmdsValue(server='DIII-D', TDI='WDOT', shot=shot)
    P_tot      = P_tot_node.data()
    P_tot_t    = P_tot_node.dim_of(0)
    wdot       = wdot_node.data()
    wdot_t     = wdot_node.dim_of(0)

    win = 5.0
    wdot_on_ptot = np.array([
        np.mean(wdot[(wdot_t >= tp - win / 2) & (wdot_t <= tp + win / 2)])
        for tp in P_tot_t
    ])

    try:
        P_rad_node = omds.OMFITmdsValue(server='DIII-D', shot=shot, TDI='prad_core')
        P_rad_fn   = interp1d(P_rad_node.dim_of(0), P_rad_node.data(),
                              bounds_error=False, fill_value=0.0)
        P_rad = P_rad_fn(P_tot_t)
    except Exception:
        P_rad = np.zeros_like(P_tot_t)

    P_sol_arr = P_tot - wdot_on_ptot - P_rad
    P_sol_MW  = float(interp1d(P_tot_t, P_sol_arr,
                               bounds_error=False, fill_value=0.0)(time_ms)) / 1e6  # PTOT is in W, convert to MW

    if P_sol_MW < 0:
        print('P_sol < 0 — 2pt model unreliable, returning 80 eV')
        return 80.0

    lam_q_mm = 0.8 / Bp
    L_par    = np.pi * R_lcfs * q95
    q_par    = (0.5 * P_sol_MW / (2.0 * np.pi * R_lcfs * lam_q_mm * 1e-3)
                * np.hypot(Bt, Bp) / Bp)
    return float(((7.0 / 2.0) * q_par * 1e6 * L_par / (2.0 * 2000.0)) ** (2.0 / 7.0))


# ---------------------------------------------------------------------------
# Wall geometry
# ---------------------------------------------------------------------------

def get_wall_R(shot, time_ms):
    '''
    Get the outer midplane radius of the limiter contour (R_wall) from the
    geqdsk file for a given shot and time. This is the value needed by the
    SOL fitting functions.

    The RLIM/ZLIM arrays in the geqdsk define the full limiter contour.
    This function finds where that contour crosses z=0 on the outboard side.
    '''
    time_ms = int(round(time_ms))
    _download_geqdsk(shot, [time_ms])

    # read RLIM/ZLIM directly from the cached file
    gfile, _ = _geqdsk_filenames(shot, time_ms)
    geq = OMFITgeqdsk(os.path.join(_GEQDSK_DIR, gfile))

    return _find_outer_midplane_radius(geq['RLIM'], geq['ZLIM'])


def _find_outer_midplane_radius(R_contour, z_contour):
    '''
    Given R and Z arrays describing a closed contour (e.g. the limiter surface),
    return the maximum R value at the outer midplane (z = 0).
    Finds all crossings of z=0 and returns the outermost one.
    '''
    crossings = []
    for i in range(len(z_contour) - 1):
        if z_contour[i] * z_contour[i + 1] <= 0:  # sign change — crosses z=0
            frac    = -z_contour[i] / (z_contour[i + 1] - z_contour[i])
            R_cross = R_contour[i] + frac * (R_contour[i + 1] - R_contour[i])
            crossings.append(R_cross)

    if crossings:
        return max(crossings)  # outermost crossing is R_wall

    # fallback: return the R of the point closest to z=0
    idx = np.argmin(np.abs(z_contour))
    return float(R_contour[idx])


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def psinorm_to_rmid(shot, time_ms, psi):
    """Map psi_N values to outboard midplane R (m)."""
    eq = _load_equilibrium(shot, int(round(time_ms)))
    return np.array([eq.psinorm2rmid(p) for p in psi])


# ---------------------------------------------------------------------------
# geqdsk file management (internal)
# ---------------------------------------------------------------------------

def _geqdsk_filenames(shot, time_ms):
    fmt = f'{int(time_ms):05d}'
    return f'g{shot}.{fmt}', f'a{shot}.{fmt}'


def _geqdsk_exists(shot, time_ms):
    gfile, afile = _geqdsk_filenames(shot, time_ms)
    return (os.path.exists(os.path.join(_GEQDSK_DIR, gfile)) and
            os.path.exists(os.path.join(_GEQDSK_DIR, afile)))


def _download_geqdsk(shot, times_ms):
    """Download geqdsk files for any times not already cached."""
    os.makedirs(_GEQDSK_DIR, exist_ok=True)
    missing = [t for t in times_ms if not _geqdsk_exists(shot, t)]
    if not missing:
        return

    try:
        from_mds_plus(device='DIII-D', shot=shot, times=list(missing),
                      snap_file='EFIT01', get_afile=True, quiet=True, close=True)
    except Exception as exc:
        print(f'Warning: geqdsk download failed: {exc}')

    for pattern in [f'g{shot}.*', f'a{shot}.*']:
        for fpath in glob.glob(pattern):
            dest = os.path.join(_GEQDSK_DIR, os.path.basename(fpath))
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(fpath, _GEQDSK_DIR)


def _load_equilibrium(shot, time_ms):
    """Return an EqdskReader for the given shot/time (downloads if needed)."""
    gfile, afile = _geqdsk_filenames(shot, time_ms)
    if not _geqdsk_exists(shot, time_ms):
        _download_geqdsk(shot, [time_ms])

    src_g = os.path.join(_GEQDSK_DIR, gfile)
    src_a = os.path.join(_GEQDSK_DIR, afile)

    # EqdskReader requires files in the current working directory
    shutil.copy(src_g, gfile)
    shutil.copy(src_a, afile)
    try:
        eq = eqdskreader.EqdskReader(gfile=gfile, afile=afile)
    finally:
        if os.path.exists(gfile):
            os.remove(gfile)
        if os.path.exists(afile):
            os.remove(afile)
    return eq
