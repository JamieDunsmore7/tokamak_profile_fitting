"""Main entry point: master_fit().

Usage::

    from tokamak_profile_fitting import master_fit

    result = master_fit(shot=1100305023, device='cmod', t_min=900, t_max=1400)
    result = master_fit(shot=189627,     device='d3d',  t_min=2000, t_max=3000)

    # result is a plain dict:
    result['te_profiles']    # shape (N_times, N_psi)
    result['te_times_ms']
    result['ne_profiles']
    result['psi_grid']
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .profiles.fit_functions import get_fit_function, Cubic, Osborne_linear_initial_guesses
from .utils import (
    remove_zeros_ne_te, add_sol_zero,
    apply_2pt_shift, reduced_chi_squared_inside_separatrix,
)


_PSI_GRID = np.append(np.arange(0.0, 0.8, 0.01),
                      np.arange(0.8, 1.2001, 0.002))



def master_fit(
    shot,
    device,
    t_min,
    t_max,
    mode               = 'per_slice',
    core_order         = 3,
    sol_order          = 0,
    enforce_mtanh      = True,
    shift_to_2pt_model = False,
    scale_to_tci       = False,
    set_te_floor       = None,
    set_ne_floor       = None,
    set_min_errorbar   = True,
    remove_zeros       = True,
    add_sol_zero_flag  = True,
    use_edge_chi_squared = False,
    plot               = False,
    debug_plot         = False,
    return_raw_data    = False,
    return_errorbars   = False,
    psi_grid           = None,
):
    """Fit Te and ne pedestal profiles to Thomson scattering data.

    Parameters
    ----------
    shot   : shot number
    device : 'cmod' or 'd3d'
    t_min, t_max : time window in ms
    mode   : 'per_slice' — fit each Thomson time point independently and
             return a time series of profiles.
             'time_window' — pool all data in the window and return one fit.
    core_order : polynomial order for the inboard (core) region (0–3)
    sol_order  : polynomial order for the outboard SOL (0–2)
    enforce_mtanh : only accept mtanh fits; skip time points where it fails.
                    if set to False, the lowest chi squared out of the cubic and mtanh fits will be chosen
    shift_to_2pt_model : shift the psi axis post-fit so the separatrix Te
                         matches the 2-point model prediction
    scale_to_tci : (C-Mod only) scale core Thomson ne to match the line-averaged interferometry measurements
    set_te_floor : minimum Te value to enforce on fitted profiles (eV)
    set_ne_floor : minimum ne value (m^-3)
    set_min_errorbar : apply absolute and relative error floors before fitting
    remove_zeros : strip zero-valued data points inside the separatrix
    add_sol_zero_flag : add an anchor zero at psi=1.05 to constrain the SOL
    use_edge_chi_squared : evaluate chi-squared only in 0.6 < psi < 1.0
    plot     : show a plot of each fit as it is produced
    return_raw_data : include the processed per-slice raw data in the result
    return_errorbars : run Monte Carlo uncertainty estimation (slow)
    psi_grid : custom output psi grid (default: coarse core + fine edge)

    Returns
    -------
    dict with keys:
        psi_grid        (N_psi,)
        mode            str
        te_times_ms     (N_te,)
        te_profiles     (N_te, N_psi)
        te_chi_squared  (N_te,)
        te_fit_type     list of str  ('mtanh' or 'cubic')
        ne_times_ms     (N_ne,)
        ne_profiles     (N_ne, N_psi)
        ne_chi_squared  (N_ne,)
        ne_fit_type     list of str
        te_profiles_std (N_te, N_psi)  only if return_errorbars=True
        ne_profiles_std (N_ne, N_psi)  only if return_errorbars=True
        raw_data        list of per-slice dicts, only if return_raw_data=True
    """
    if psi_grid is None:
        psi_grid = _PSI_GRID.copy()

    name = device.lower().replace('-', '').replace(' ', '')
    if name == 'cmod':
        from .devices import cmod as dev
    elif name in ('d3d', 'diiid'):
        from .devices import d3d as dev
    else:
        raise ValueError(f"Unknown device '{device}'. Supported: 'cmod', 'd3d'.")

    fit_func, n_params = get_fit_function(core_order, sol_order)
    min_errs           = dev.get_min_errorbars()
    min_ne_err         = min_errs['ne']
    min_te_err         = min_errs['te']

    # ---- fetch data ----
    data      = dev.get_thomson_data(shot, t_min, t_max)
    times_ms  = data['times_ms']
    psi_all   = data['psi']
    ne_all    = data['ne']
    ne_err_all = data['ne_err']
    te_all    = data['te']
    te_err_all = data['te_err']
    edge_mask = data['edge_mask']

    if scale_to_tci and hasattr(dev, 'scale_to_tci'):
        n_core = int(np.sum(~edge_mask))
        ne_all[:n_core, :] = dev.scale_to_tci(shot, times_ms, ne_all[:n_core, :])

    # ---- clean each time slice ----
    slices = []
    for t_idx, time_ms in enumerate(times_ms):
        psi_t    = psi_all[:, t_idx]
        ne_t     = ne_all[:, t_idx]
        ne_err_t = ne_err_all[:, t_idx]
        te_t     = te_all[:, t_idx]
        te_err_t = te_err_all[:, t_idx]

        if np.all(np.isnan(psi_t)) or np.all(psi_t < 0):
            continue

        if remove_zeros:
            psi_t, ne_t, te_t, ne_err_t, te_err_t = remove_zeros_ne_te(
                psi_t, ne_t, te_t, ne_err_t, te_err_t, core_only=True)
            
        valid    = (ne_err_t != 0) & (te_err_t != 0)
        psi_t    = psi_t[valid]
        ne_t     = ne_t[valid];    ne_err_t = ne_err_t[valid]
        te_t     = te_t[valid];    te_err_t = te_err_t[valid]

        if len(psi_t) == 0 or np.sum(psi_t < 0.8) < 3:
            print(f'  t={time_ms} ms: insufficient core data, skipping')
            continue

        centre_ne = ne_t[np.argmin(psi_t)]
        centre_te = te_t[np.argmin(psi_t)]
        phys      = (ne_t < 1.5 * centre_ne) & (te_t < 1.5 * centre_te)
        psi_t     = psi_t[phys]
        ne_t      = ne_t[phys];    ne_err_t = ne_err_t[phys]
        te_t      = te_t[phys];    te_err_t = te_err_t[phys]

        if max(te_t) < 100 or max(ne_t) < 1e18:
            print(f'  t={time_ms} ms: values too low, skipping')
            continue

        psi_te, te, te_err = psi_t.copy(), te_t.copy(), te_err_t.copy()
        psi_ne, ne, ne_err = psi_t.copy(), ne_t.copy(), ne_err_t.copy()
        if add_sol_zero_flag:
            edge_sel = psi_t > 0.8
            core_sel = ~edge_sel
            # Add a zero anchor in the SOL to edge data only, then recombine
            # with core — matching the original repo (functions_fit_1D.py:200-210)
            psi_te_e, te_e, te_err_e = add_sol_zero(psi_te[edge_sel], te[edge_sel], te_err[edge_sel])
            psi_te = np.concatenate([psi_te[core_sel], psi_te_e])
            te     = np.concatenate([te[core_sel],     te_e])
            te_err = np.concatenate([te_err[core_sel], te_err_e])
            psi_ne_e, ne_e, ne_err_e = add_sol_zero(psi_ne[edge_sel], ne[edge_sel], ne_err[edge_sel])
            psi_ne = np.concatenate([psi_ne[core_sel], psi_ne_e])
            ne     = np.concatenate([ne[core_sel],     ne_e])
            ne_err = np.concatenate([ne_err[core_sel], ne_err_e])

        if set_min_errorbar:
            te_err = np.maximum(te_err, np.maximum(min_te_err, te * 0.05))
            ne_err = np.maximum(ne_err, np.maximum(min_ne_err, ne * 0.05))

        edge_sel = psi_t > 0.8
        slices.append(dict(
            time_ms     = time_ms,
            psi_te      = psi_te,  te     = te,    te_err  = te_err,
            psi_ne      = psi_ne,  ne     = ne,    ne_err  = ne_err,
            psi_edge    = psi_t[edge_sel],
            te_edge     = te_t[edge_sel],
            te_err_edge = te_err_t[edge_sel],
            ne_edge     = ne_t[edge_sel],
        ))

    if len(slices) == 0:
        raise ValueError('No valid time slices found in the requested window.')

    fit_kwargs = dict(
        psi_grid=psi_grid, fit_func=fit_func, n_params=n_params,
        enforce_mtanh=enforce_mtanh, use_edge_chi_squared=use_edge_chi_squared,
        set_te_floor=set_te_floor, set_ne_floor=set_ne_floor,
        shift_to_2pt_model=shift_to_2pt_model, dev=dev, shot=shot,
        plot=plot, debug_plot=debug_plot,
        return_raw_data=return_raw_data, return_errorbars=return_errorbars,
    )

    if mode == 'per_slice':
        return _fit_per_slice(slices, **fit_kwargs)
    elif mode == 'time_window':
        return _fit_time_window(slices, **fit_kwargs)
    else:
        raise ValueError(f"mode must be 'per_slice' or 'time_window', got '{mode}'")


# ---------------------------------------------------------------------------
# Per-slice mode
# ---------------------------------------------------------------------------

def _fit_per_slice(slices, psi_grid, fit_func, n_params, enforce_mtanh,
                   use_edge_chi_squared, set_te_floor, set_ne_floor,
                   shift_to_2pt_model, dev, shot,
                   plot, debug_plot, return_raw_data, return_errorbars):

    te_times, te_profiles, te_chisq, te_types = [], [], [], []
    ne_times, ne_profiles, ne_chisq, ne_types = [], [], [], []
    te_stds, ne_stds = [], []
    raw_out = [] if return_raw_data else None

    last_te_params = None
    last_ne_params = None

    for sl in slices:
        time_ms = sl['time_ms']

        te_profile, te_chi, te_type, last_te_params = _fit_one_profile(
            psi=sl['psi_te'], values=sl['te'], errors=sl['te_err'],
            psi_grid=psi_grid, fit_func=fit_func, n_params=n_params,
            enforce_mtanh=enforce_mtanh, use_edge_chi_squared=use_edge_chi_squared,
            profile_type='te', last_params=last_te_params, debug_plot=debug_plot,
        )

        ne_profile, ne_chi, ne_type, last_ne_params = _fit_one_profile(
            psi=sl['psi_ne'], values=sl['ne'], errors=sl['ne_err'],
            psi_grid=psi_grid, fit_func=fit_func, n_params=n_params,
            enforce_mtanh=enforce_mtanh, use_edge_chi_squared=use_edge_chi_squared,
            profile_type='ne', last_params=last_ne_params, debug_plot=debug_plot,
        )

        if te_profile is None and ne_profile is None:
            continue

        if set_te_floor is not None and te_profile is not None:
            te_profile[te_profile < set_te_floor] = set_te_floor
        if set_ne_floor is not None and ne_profile is not None:
            ne_profile[ne_profile < set_ne_floor] = set_ne_floor

        shift = 0.0
        if shift_to_2pt_model and te_profile is not None:
            try:
                te_sep = dev.get_Te_sep(shot, time_ms)
                # only_edge=False: original shifts entire psi grid (the
                # only_shift_edge branch is dead code in the original's
                # apply_2pt_shift — the last line always shifts everything)
                shifted_psi, shift = apply_2pt_shift(psi_grid, te_profile,
                                                     te_sep, only_edge=False)
                te_profile = interp1d(shifted_psi, te_profile,
                                      fill_value='extrapolate')(psi_grid)
                if ne_profile is not None:
                    ne_profile = interp1d(shifted_psi, ne_profile,
                                          fill_value='extrapolate')(psi_grid)
            except Exception as exc:
                print(f'    2pt model shift failed: {exc}')

        te_std = ne_std = None
        if return_errorbars:
            te_std = _monte_carlo_std(sl['psi_te'], sl['te'], sl['te_err'],
                                      psi_grid, fit_func, n_params, 'te')
            ne_std = _monte_carlo_std(sl['psi_ne'], sl['ne'], sl['ne_err'],
                                      psi_grid, fit_func, n_params, 'ne')

        if plot and te_profile is not None:
            _plot_fit(psi_grid, sl['psi_te'], sl['te'], sl['te_err'],
                      te_profile, te_chi, time_ms, shift, 'Te (eV)')
            if ne_profile is not None:
                _plot_fit(psi_grid, sl['psi_ne'], sl['ne'], sl['ne_err'],
                          ne_profile, ne_chi, time_ms, shift, 'ne (m$^{-3}$)')

        if te_profile is not None:
            te_times.append(time_ms)
            te_profiles.append(te_profile)
            te_chisq.append(te_chi)
            te_types.append(te_type)
            if return_errorbars:        
                te_stds.append(te_std)
        if ne_profile is not None:
            ne_times.append(time_ms)
            ne_profiles.append(ne_profile)
            ne_chisq.append(ne_chi)
            ne_types.append(ne_type)
            if return_errorbars:
                ne_stds.append(ne_std)
        if return_raw_data:
            # store raw data with the 2pt shift applied to psi, matching the
            # original code which stores total_psi_te + shift (line 752 of
            # functions_fit_1D.py) so raw data and fit are on the same psi scale
            if shift != 0.0:
                sl = dict(sl)
                for key in ('psi_te', 'psi_ne'):
                    sl[key] = sl[key] + shift
            raw_out.append(sl)

    result = dict(
        psi_grid       = psi_grid,
        mode           = 'per_slice',
        te_times_ms    = np.array(te_times),
        te_profiles    = np.array(te_profiles) if te_profiles else np.empty((0, len(psi_grid))),
        te_chi_squared = np.array(te_chisq),
        te_fit_type    = te_types,
        ne_times_ms    = np.array(ne_times),
        ne_profiles    = np.array(ne_profiles) if ne_profiles else np.empty((0, len(psi_grid))),
        ne_chi_squared = np.array(ne_chisq),
        ne_fit_type    = ne_types,
    )
    if return_errorbars:
        result['te_profiles_std'] = np.array(te_stds) if te_stds else None
        result['ne_profiles_std'] = np.array(ne_stds) if ne_stds else None
    if return_raw_data:
        result['raw_data'] = raw_out
    return result


# ---------------------------------------------------------------------------
# Time-window mode
# ---------------------------------------------------------------------------

def _align_slice_psi(sl, dev, shot):
    """Shift outer psi (> 0.8) of a slice to align its pedestal with the 2pt model.

    Matches the original repo's get_twopt_shift_from_edge_Te_fit logic:
    fits Osborne_Tanh_linear to the outer data (psi > 0.8, including the SOL
    zero anchor), finds where the fit crosses the 2pt model Te_sep, and shifts
    all psi > 0.8 by that amount.
    """
    from .profiles.fit_functions import Osborne_Tanh_linear

    time_ms = sl['time_ms']

    # Use the full psi_te array filtered to psi > 0.8 — this includes the SOL
    # zero anchor at psi=1.05, matching the original code's edge fit input
    outer = sl['psi_te'] > 0.8
    psi_outer   = sl['psi_te'][outer]
    te_outer    = sl['te'][outer]
    te_err_outer = sl['te_err'][outer]

    if len(psi_outer) < 5:
        return sl, 0.0

    te_max = float(np.max(te_outer[te_outer > 0])) if np.any(te_outer > 0) else 100.0

    # Same five guesses as the original get_twopt_shift_from_edge_Te_fit
    auto = Osborne_linear_initial_guesses(psi_outer[te_outer > 0],
                                          te_outer[te_outer > 0], n_params=5)
    guesses = [
        auto,
        [9.926e-01, 4.018e-02, 255.6, 12.85,  0.218],
        [1.006,     2.018e-02, 150.0, 10.0,   0.35 ],
        [1.006,     2.88e-03,  131.6, 31.1,   0.069],
        [1.0,       2.88e-03,  600.0, 60.0,   0.039],
    ]
    te_params = None
    for g in guesses:
        try:
            te_params, _ = curve_fit(
                Osborne_Tanh_linear, psi_outer, te_outer, p0=g,
                sigma=te_err_outer, absolute_sigma=True, maxfev=2000,
                bounds=([0.85, 0.005, 10, -0.001, -np.inf],
                        [1.1,  0.2,   te_max, te_max, np.inf]),
            )
            break
        except Exception:
            continue

    if te_params is None:
        return sl, 0.0

    try:
        te_sep = dev.get_Te_sep(shot, time_ms)
    except Exception:
        return sl, 0.0

    fine_psi = np.linspace(float(np.min(psi_outer)), float(np.max(psi_outer)), 100)
    te_fit   = Osborne_Tanh_linear(fine_psi, *te_params)
    _, shift = apply_2pt_shift(fine_psi, te_fit, te_sep, only_edge=False)


    if shift == 0.0:
        return sl, 0.0

    sl = dict(sl)
    for key in ('psi_te', 'psi_ne'):
        arr = sl[key].copy()
        arr[arr > 0.8] += shift
        sl[key] = arr

    return sl, shift


def _fit_time_window(slices, psi_grid, fit_func, n_params, enforce_mtanh,
                     use_edge_chi_squared, set_te_floor, set_ne_floor,
                     shift_to_2pt_model, dev, shot,
                     plot, debug_plot, return_raw_data, return_errorbars):

    # When shift_to_2pt_model is requested, align each slice individually
    # before pooling so the pedestal sits at a consistent psi position.
    # This matches the original code which shifted per-slice raw data before
    # pooling rather than shifting the fitted profile afterwards.
    aligned_slices = []
    slice_shifts   = []
    for sl in slices:
        if shift_to_2pt_model:
            sl, s = _align_slice_psi(sl, dev, shot)
            slice_shifts.append(s)
        aligned_slices.append(sl)
    mean_psi_shift = float(np.mean(slice_shifts)) if slice_shifts else 0.0

    psi_te_pool = np.concatenate([sl['psi_te'] for sl in aligned_slices])
    te_pool     = np.concatenate([sl['te']     for sl in aligned_slices])
    te_err_pool = np.concatenate([sl['te_err'] for sl in aligned_slices])
    psi_ne_pool = np.concatenate([sl['psi_ne'] for sl in aligned_slices])
    ne_pool     = np.concatenate([sl['ne']     for sl in aligned_slices])
    ne_err_pool = np.concatenate([sl['ne_err'] for sl in aligned_slices])

    t_min_w  = int(slices[0]['time_ms'])
    t_max_w  = int(slices[-1]['time_ms'])
    mean_t   = float(np.mean([sl['time_ms'] for sl in slices]))

    te_profile, te_chi, te_type, _ = _fit_one_profile(
        psi=psi_te_pool, values=te_pool, errors=te_err_pool,
        psi_grid=psi_grid, fit_func=fit_func, n_params=n_params,
        enforce_mtanh=enforce_mtanh, use_edge_chi_squared=use_edge_chi_squared,
        profile_type='te', debug_plot=debug_plot,
    )

    ne_profile, ne_chi, ne_type, _ = _fit_one_profile(
        psi=psi_ne_pool, values=ne_pool, errors=ne_err_pool,
        psi_grid=psi_grid, fit_func=fit_func, n_params=n_params,
        enforce_mtanh=enforce_mtanh, use_edge_chi_squared=use_edge_chi_squared,
        profile_type='ne', debug_plot=debug_plot,
    )

    if set_te_floor is not None and te_profile is not None:
        te_profile[te_profile < set_te_floor] = set_te_floor
    if set_ne_floor is not None and ne_profile is not None:
        ne_profile[ne_profile < set_ne_floor] = set_ne_floor

    if plot and te_profile is not None:
        _plot_fit(psi_grid, psi_te_pool, te_pool, te_err_pool,
                  te_profile, te_chi, f'{t_min_w}–{t_max_w} ms', 0.0, 'Te (eV)')
        if ne_profile is not None:
            _plot_fit(psi_grid, psi_ne_pool, ne_pool, ne_err_pool,
                      ne_profile, ne_chi, f'{t_min_w}–{t_max_w} ms', 0.0, 'ne (m$^{-3}$)')

    result = dict(
        psi_grid       = psi_grid,
        psi_shift      = mean_psi_shift,
        mode           = 'time_window',
        te_times_ms    = np.array([mean_t]),
        te_profiles    = np.array([te_profile]) if te_profile is not None
                         else np.empty((0, len(psi_grid))),
        te_chi_squared = np.array([te_chi if te_chi is not None else np.nan]),
        te_fit_type    = [te_type],
        ne_times_ms    = np.array([mean_t]),
        ne_profiles    = np.array([ne_profile]) if ne_profile is not None
                         else np.empty((0, len(psi_grid))),
        ne_chi_squared = np.array([ne_chi if ne_chi is not None else np.nan]),
        ne_fit_type    = [ne_type],
    )
    if return_errorbars:
        result['te_profiles_std'] = np.array([_monte_carlo_std(
            psi_te_pool, te_pool, te_err_pool, psi_grid, fit_func, n_params, 'te')])
        result['ne_profiles_std'] = np.array([_monte_carlo_std(
            psi_ne_pool, ne_pool, ne_err_pool, psi_grid, fit_func, n_params, 'ne')])
    if return_raw_data:
        result['raw_data'] = aligned_slices
    return result


# ---------------------------------------------------------------------------
# Core fitting logic
# ---------------------------------------------------------------------------

def _fit_one_profile(psi, values, errors,
                     psi_grid, fit_func, n_params,
                     enforce_mtanh, use_edge_chi_squared,
                     profile_type, last_params=None, debug_plot=False):
    """Fit one profile (Te or ne) with mtanh and optionally cubic.

    Returns (profile_on_grid, chi_squared, fit_type_str, updated_last_params).
    Any return value may be None if all fits fail.
    """
    is_ne   = (profile_type == 'ne')
    scale   = 1e20 if is_ne else 1.0
    vals_s  = values / scale
    errs_s  = errors / scale
    max_val = max(vals_s)

    # Bounds: [c0, c1, c2, c3] + polynomial terms
    # c0=centre, c1=width, c2=top, c3=bottom — matching original code exactly
    if is_ne:
        lb = [0.85, 0.01,  0.0, -0.001 ] + [-np.inf] * (n_params - 4)
        ub = [1.1,  0.25,  max_val, np.inf ] + [ np.inf] * (n_params - 4)
    else:
        lb = [0.85, 0.01,  10.0 / scale, -0.001 ] + [-np.inf] * (n_params - 4)
        ub = [1.1,  0.2,   max_val, max_val] + [ np.inf] * (n_params - 4)

    # Hardcoded initial guesses — from original repo
    if is_ne:
        hardcoded = [
            # 650 kA C-Mod
            [1.00604712, 3.7400836e-02, 2.10662412, 1.68897974e-02, -6.32778417e-02, 2.29233952e-03, -2.0627212e-05],
            # 1 MA C-Mod
            [1.02123755e+00, 5.02744526e-02, 2.54219267e+00, -9.99999694e-04, 2.58724602e-02, -2.32961078e-03,  4.20279037e-05],
            # D3D
            [0.99, 0.04, 1.0, 0.05, 0.0, 0.0, 0.0],
            [0.99, 0.04, 0.3, 0.05, 0.0, 0.0, 0.0],
        ]
    else:
        hardcoded = [
            # from fit_te_mtanh in original functions_profile_fitting.py
            [9.92614859e-01, 4.01791101e-02, 2.55550908e+02 / scale, 1.28542623e+01 / scale, 2.17777084e-01, -3.45196862e-03, 1.42947373e-04],
        ]

    guesses = []
    for g in hardcoded:
        guesses.append(np.array((list(g) + [0.0] * n_params)[:n_params]))
    # Osborne_linear auto-guess uses edge data including the SOL zero anchor,
    # matching the original which passes raw_te_psi_edge/raw_ne_psi_edge after
    # add_SOL_zeros_in_psi_coords (functions_fit_1D.py lines 333-334, 473)
    try:
        edge_sel = psi > 0.8
        auto = Osborne_linear_initial_guesses(psi[edge_sel], vals_s[edge_sel], n_params)
        guesses.insert(0, np.array(auto))
    except Exception:
        pass
    if last_params is not None:
        guesses.insert(0, np.array(last_params))

    # Clamp all guesses to the parameter bounds.  Hardcoded C-Mod/D3D values
    # have c2 (pedestal top) much higher than D3D data, putting them outside
    # the ub[2]=max_val bound and causing curve_fit to raise immediately.
    lb_arr = np.array(lb)
    ub_arr = np.array(ub)
    guesses = [np.clip(g, lb_arr, ub_arr) for g in guesses]

    # Prepend data-adaptive guesses with c2 tuned to the actual data maximum
    # so the optimizer starts from a physically reasonable point.
    c2_est = max_val * 0.85
    c3_est = max_val * 0.02
    for _c0, _c1 in [(0.99, 0.04), (0.98, 0.04), (0.99, 0.05), (1.00, 0.03)]:
        g = np.zeros(n_params)
        g[0] = _c0; g[1] = _c1; g[2] = c2_est
        if n_params > 3:
            g[3] = c3_est
        guesses.insert(0, g)

    # ---- mtanh fit ----
    params_mtanh = profile_mtanh = chi_mtanh = None
    for guess in guesses:
        try:
            params_mtanh, _ = curve_fit(fit_func, psi, vals_s, p0=guess, sigma=errs_s, absolute_sigma=True, maxfev=2000, bounds=(lb, ub),)
            fitted_at_data = fit_func(psi, *params_mtanh)
            chi_mtanh = reduced_chi_squared_inside_separatrix(
                psi, vals_s, fitted_at_data, errs_s, n_params,
                only_edge=use_edge_chi_squared)
            profile_mtanh = fit_func(psi_grid, *params_mtanh) * scale
            break
        except Exception:
            continue

    # ---- diagnostic plot: raw data, all guesses, and final fit ----
    if debug_plot:
        lbl  = 'ne' if is_ne else 'Te'
        unit = 'm$^{-3}$' if is_ne else 'eV'
        psi_plot = np.linspace(0.0, 1.25, 500)
        y_max    = float(np.nanmax(values)) if values.size else 1.0

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(psi, values, yerr=errors,
                    fmt='o', ms=5, color='steelblue', alpha=0.7,
                    label='Data', zorder=5)

        for i, g in enumerate(guesses):
            try:
                curve = fit_func(psi_plot, *g) * scale
                ok = np.isfinite(curve) & (curve >= 0) & (curve < 5 * y_max)
                if ok.any():
                    ax.plot(psi_plot[ok], curve[ok], lw=1.2, alpha=0.6,
                            label=f'Guess {i}')
            except Exception:
                pass

        if profile_mtanh is not None:
            ax.plot(psi_grid, profile_mtanh, color='crimson', lw=2.5,
                    label=rf'Fit  $\chi^2_\nu={chi_mtanh:.2f}$', zorder=6)
            title = f'{lbl} — fit succeeded'
        else:
            title = f'{lbl} — ALL GUESSES FAILED'

        ax.axvline(1.0, color='black', ls='--', alpha=0.5, label='Sep')
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(f'{lbl} ({unit})')
        ax.set_xlim(0.0, 1.25)
        ax.set_ylim(bottom=0, top=2.0 * y_max if y_max > 0 else 1)
        ax.set_title(title, color='black' if profile_mtanh is not None else 'red')
        ax.legend(fontsize=7, ncol=2, framealpha=0.8)
        ax.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # if enforce_mtanh is True and the fit has succeeed, we can return immediately here
    if enforce_mtanh:
        if profile_mtanh is not None:
            return profile_mtanh, chi_mtanh, 'mtanh', params_mtanh
        return None, None, None, last_params

    # ---- cubic fallback ----
    params_cubic = profile_cubic= chi_cubic = None
    try:
        params_cubic, _ = curve_fit(Cubic, psi, vals_s,
                                sigma=errs_s, absolute_sigma=True, maxfev=2000)
        chi_cubic      = reduced_chi_squared_inside_separatrix(
            psi, vals_s, Cubic(psi, *params_cubic), errs_s, 4,
            only_edge=use_edge_chi_squared)
        profile_cubic = Cubic(psi_grid, *params_cubic) * scale
    except Exception:
        pass

    # discard fits with nonsensical chi-squared
    def _bad_chi(chi):
        return chi is None or chi <= 0 or chi > 20

    if _bad_chi(chi_mtanh):
        params_mtanh = profile_mtanh = chi_mtanh = None
    if _bad_chi(chi_cubic):
        params_cubic  = profile_cubic = chi_cubic  = None

    # discard mtanh if fewer than 3 points in the pedestal region, or if either
    # shoulder (inner top / outer foot) has no coverage
    if params_mtanh is not None:
        lo, hi = params_mtanh[0] - params_mtanh[1], params_mtanh[0] + params_mtanh[1]
        mid = params_mtanh[0]
        n_pedestal = np.sum((psi >= lo)  & (psi <= hi))
        n_top      = np.sum((psi >= lo)  & (psi <  mid))
        n_bottom   = np.sum((psi >= mid) & (psi <= hi))
        if n_pedestal < 3 or n_top < 1 or n_bottom < 1:
            params_mtanh = profile_mtanh = chi_mtanh = None

    # choose best
    if profile_mtanh is not None and profile_cubic is not None:
        if chi_cubic < chi_mtanh:
            return profile_cubic, chi_cubic, 'cubic', last_params
        return profile_mtanh, chi_mtanh, 'mtanh', params_mtanh
    if profile_mtanh is not None:
        return profile_mtanh, chi_mtanh, 'mtanh', params_mtanh
    if profile_cubic is not None:
        return profile_cubic, chi_cubic, 'cubic', last_params
    return None, None, None, last_params


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty
# ---------------------------------------------------------------------------

def _monte_carlo_std(psi, values, errors, psi_grid,
                     fit_func, n_params, profile_type, n_samples=100):
    scale   = 1e20 if profile_type == 'ne' else 1.0
    results = []
    for _ in range(n_samples):
        perturbed = np.random.normal(loc=values, scale=np.abs(errors))
        try:
            params, _ = curve_fit(fit_func, psi, perturbed / scale,
                                  sigma=errors / scale,
                                  absolute_sigma=True, maxfev=2000)
            fitted = fit_func(psi_grid, *params) * scale
            fitted[np.isinf(fitted)] = np.nan
            results.append(fitted)
        except Exception:
            pass
    if len(results) < 2:
        return np.full(len(psi_grid), np.nan)
    return np.nanstd(np.array(results), axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fit(psi_grid, psi_raw, values_raw, errors_raw,
              fitted, chi_sq, time_label, shift, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(psi_raw + shift, values_raw, yerr=errors_raw,
                fmt='o', mfc='white', color='steelblue', alpha=0.7, label='Data')
    ax.plot(psi_grid, fitted, color='crimson', linewidth=2,
            label=rf'Fit  $\chi^2_\nu={chi_sq:.2f}$')
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5,
               label=f'Separatrix (2pt model, shift={shift:+.4f})' if shift else 'Separatrix')
    if shift:
        ax.axvline(1.0 - shift, color='purple', linestyle='--', alpha=0.5,
                   label='Separatrix (unmapped)')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(ylabel)
    t_str = time_label if isinstance(time_label, str) else f'{int(time_label)} ms'
    ax.set_title(f't = {t_str}')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
