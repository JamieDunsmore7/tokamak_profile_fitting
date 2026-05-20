# Functions for fitting the SOL profile in R - R_sep space.
# The purpose of this is to get better profiles in the SOL for KN1D runs.
#
# Usage:
#   x_grid, profile, ... = fit_sol_double_exponential(shot, time_ms, 'd3d', psi_grid, ...)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from .fit_functions import Osborne_Tanh_cubic, Osborne_Tanh_cubic_quadratic_SOL


def _resolve_device(device):
    if device == 'cmod':
        from ..devices import cmod as dev
    elif device in ('d3d', 'diii-d'):
        from ..devices import d3d as dev
    else:
        raise ValueError(f"Unknown device '{device}'. Supported: 'cmod', 'd3d'.")
    return dev


def get_rid_of_ELMs(raw_psi_values, raw_values, raw_errors):
    '''
    NOTE: this strategy doesn't work if the arrays are already sorted.
    Needs to be separated by shot so I can split them up.
    '''
    masking_indices = [0]
    for idx in range(1, len(raw_psi_values)):
        if raw_psi_values[idx] < raw_psi_values[idx - 1]:
            masking_indices.append(idx)
    masking_indices.append(len(raw_psi_values))

    average_SOL_values_list = []

    for mask_index in range(len(masking_indices)-1):
        masking_start = masking_indices[mask_index]
        masking_end   = masking_indices[mask_index + 1]
        raw_psi_masked    = raw_psi_values[masking_start:masking_end]
        raw_values_masked = raw_values[masking_start:masking_end]

        raw_values_masked_SOL = raw_values_masked[raw_psi_masked > 1.03]
        average_SOL_value = np.nanmean(raw_values_masked_SOL)
        average_SOL_values_list.append(average_SOL_value)

    average_SOL_values_list = np.array(average_SOL_values_list)

    # get rid of any data whose average is greater than x2 the total average
    total_average_SOL_value = np.nanmean(average_SOL_values_list)
    indices_to_remove = np.where(average_SOL_values_list > 2 * total_average_SOL_value)[0]

    # remove the values from the raw data
    raw_psi_values = raw_psi_values.copy().astype(float)
    raw_values     = raw_values.copy().astype(float)
    raw_errors     = raw_errors.copy().astype(float)

    for index in indices_to_remove:
        masking_start = masking_indices[index]
        masking_end   = masking_indices[index + 1]
        raw_psi_values[masking_start:masking_end] = np.nan
        raw_values[masking_start:masking_end]     = np.nan
        raw_errors[masking_start:masking_end]     = np.nan

    # remove any nans
    raw_psi_values = raw_psi_values[~np.isnan(raw_psi_values)]
    raw_values     = raw_values[~np.isnan(raw_values)]
    raw_errors     = raw_errors[~np.isnan(raw_errors)]

    assert len(raw_psi_values) == len(raw_values) == len(raw_errors), \
        "Raw data arrays must be the same length after removing ELMs"

    return raw_psi_values, raw_values, raw_errors


def fit_sol_exponential_decay(shot, time_ms, device,
                               psi_grid, fitted_profile,
                               raw_psi_values, raw_values, raw_errors,
                               R_wall, output_R_grid=None):
    '''
    WHAT DOES THIS FUNCTION DO?
    Fits a simple exponential decay to the SOL profile in R - R_sep space.
    Fitting parameters are the decay length and the constant offset.

    Expect that the fit will have been performed over multiple time slices,
    so the raw data will contain data from many time slices.

    NOTE: for speed and ease, this function uses a single equilibrium to map
    all time points. I think this would be an issue if mapping directly to Rmid
    (because Rmid can jitter quite a bit), but I checked and R - Rsep seems to
    be much less sensitive to the choice of equilibrium used.
    '''
    dev = _resolve_device(device)
    raw_psi_values, raw_values, raw_errors = get_rid_of_ELMs(raw_psi_values, raw_values, raw_errors)

    sep_value = interp1d(psi_grid, fitted_profile, fill_value='extrapolate')(1.0)

    Rmid_raw         = dev.psinorm_to_rmid(shot, time_ms, raw_psi_values)
    Rmid_sep         = float(dev.psinorm_to_rmid(shot, time_ms, np.array([1.0])))
    raw_R_minus_Rsep = Rmid_raw - Rmid_sep

    SOL_mask = raw_R_minus_Rsep > -0.001  # allow 1mm inside the separatrix
    raw_R_minus_Rsep_SOL = raw_R_minus_Rsep[SOL_mask]
    raw_values_SOL       = raw_values[SOL_mask]
    raw_errors_SOL       = raw_errors[SOL_mask]

    grid_resolution    = 0.001
    generated_SOL_grid = np.arange(0, (R_wall - Rmid_sep) + grid_resolution, grid_resolution)

    # exponential decay function
    def exponential_decay(x, a, const):
        return (sep_value - const) * np.exp(-x / a) + const

    fit_params, fit_cov = curve_fit(exponential_decay, raw_R_minus_Rsep_SOL, raw_values_SOL,
                                    p0=[0.1, 0.01], sigma=raw_errors_SOL, absolute_sigma=True,
                                    bounds=([-np.inf, 0], [np.inf, np.inf]))
    fitted_profile_SOL = exponential_decay(generated_SOL_grid, *fit_params)

    fitted_R_minus_Rsep = dev.psinorm_to_rmid(shot, time_ms, psi_grid) - Rmid_sep

    if np.average(fitted_profile_SOL) > 1e10:
        # this is a density profile so set the density limit
        fitted_profile_SOL[fitted_profile_SOL < 1e16] = 1e16
    else:
        # this is a temperature profile so set the temperature limit
        fitted_profile_SOL[fitted_profile_SOL < 3] = 3

    plt.scatter(raw_R_minus_Rsep, raw_values, label='Raw Data', color='red', alpha=0.1)
    plt.errorbar(raw_R_minus_Rsep, raw_values, yerr=raw_errors, fmt='o', color='red',
                 label='Raw Data Error', alpha=0.1)
    plt.scatter(0, sep_value, color='black', marker='x', s=100)
    plt.plot(fitted_R_minus_Rsep, fitted_profile, color='orange', label='Fitted Profile')
    plt.plot(generated_SOL_grid, fitted_profile_SOL, color='black', label='Fitted SOL Profile')
    plt.axvline(0, color='black', linestyle='--', label='Separatrix')
    plt.xlabel('R - Rsep (m)')
    plt.ylabel('Profile Value')
    plt.legend()
    plt.show()

    if output_R_grid is None:
        output_grid_resolution = 0.001
        output_R_grid = np.arange(-0.1, (R_wall - Rmid_sep) + output_grid_resolution,
                                   output_grid_resolution)

    output_profile_inside_LCFS = interp1d(fitted_R_minus_Rsep, fitted_profile,
                                           fill_value='extrapolate')(output_R_grid[output_R_grid < 0])
    output_profile_SOL = interp1d(generated_SOL_grid, fitted_profile_SOL,
                                   fill_value='extrapolate')(output_R_grid[output_R_grid >= 0])
    output_profile = np.concatenate([output_profile_inside_LCFS, output_profile_SOL])

    plt.plot(output_R_grid, output_profile, color='blue', label='Output Profile')
    plt.axvline(0, color='black', linestyle='--', label='Separatrix')
    plt.xlabel('R - Rsep (m)')
    plt.ylabel('Profile Value')
    plt.legend()
    plt.show()

    return output_R_grid, output_profile, raw_R_minus_Rsep_SOL, raw_values_SOL, raw_errors_SOL


def fit_sol_double_exponential(shot, time_ms, device,
                                psi_grid, fitted_profile,
                                raw_psi_values, raw_values, raw_errors,
                                R_wall, Rlim=None, output_R_grid=None,
                                stronger_decay_behind_limiter=False):
    '''
    WHAT DOES THIS FUNCTION DO?
    Fits a double exponential decay to the SOL profile in R - R_sep space
    (i.e. two different decay lengths either side of R_switch).
    Fitting parameters are the two different decay lengths and the switch point.

    Expect that the fit will have been performed over multiple time slices,
    so the raw data will contain data from many time slices.

    NOTE: for speed and ease, this function uses a single equilibrium to map
    all time points. I think this would be an issue if mapping directly to Rmid
    (because Rmid can jitter quite a bit), but I checked and R - Rsep seems to
    be much less sensitive to the choice of equilibrium used.
    '''
    dev = _resolve_device(device)
    raw_psi_values, raw_values, raw_errors = get_rid_of_ELMs(raw_psi_values, raw_values, raw_errors)

    sorted_raw_order = np.argsort(raw_psi_values)
    raw_psi_values   = raw_psi_values[sorted_raw_order]
    raw_values       = raw_values[sorted_raw_order]
    raw_errors       = raw_errors[sorted_raw_order]

    sep_value = interp1d(psi_grid, fitted_profile, fill_value='extrapolate')(1.0)

    Rmid_raw         = dev.psinorm_to_rmid(shot, time_ms, raw_psi_values)
    Rmid_sep         = float(dev.psinorm_to_rmid(shot, time_ms, np.array([1.0])))
    raw_R_minus_Rsep = Rmid_raw - Rmid_sep

    grid_resolution    = 0.001
    generated_SOL_grid = np.arange(0, (R_wall - Rmid_sep) + grid_resolution, grid_resolution)

    SOL_mask = raw_R_minus_Rsep > -0.001  # allow 1mm inside the separatrix
    raw_R_minus_Rsep_SOL = raw_R_minus_Rsep[SOL_mask]
    raw_values_SOL       = raw_values[SOL_mask]
    raw_errors_SOL       = raw_errors[SOL_mask]

    # log transform requires positive values and errors
    valid = (raw_values_SOL > 0) & (raw_errors_SOL > 0)
    raw_R_minus_Rsep_SOL = raw_R_minus_Rsep_SOL[valid]
    raw_values_SOL       = raw_values_SOL[valid]
    raw_errors_SOL       = raw_errors_SOL[valid]

    def double_linear_decay(x, m1, m2, R_switch):
        x = np.asarray(x)
        z_left  = np.log(sep_value) + m1 * x
        z_right = (np.log(sep_value) + m1 * R_switch) + m2 * (x - R_switch)  # continuous at R_switch
        return np.where(x <= R_switch, z_left, z_right)

    log_errors_SOL = raw_errors_SOL / raw_values_SOL  # convert to relative errors for the fit

    fit_params, fit_cov = curve_fit(double_linear_decay, raw_R_minus_Rsep_SOL,
                                    np.log(raw_values_SOL),
                                    p0=[-1e5, -1e5, 0.01], sigma=log_errors_SOL,
                                    absolute_sigma=True,
                                    bounds=([-np.inf, -np.inf, 0.001], [0, 0, 0.02]))
    fitted_profile_SOL = np.exp(double_linear_decay(generated_SOL_grid, *fit_params))

    fitted_R_minus_Rsep = dev.psinorm_to_rmid(shot, time_ms, psi_grid) - Rmid_sep

    if np.average(fitted_profile_SOL) > 1e10:
        # DENSITY PROFILE
        if stronger_decay_behind_limiter:
            # set a different decay length behind the limiter shadow
            limiter_shadow_decay_length = 0.025  # from Francesco's KN1D OMFIT module
            limiter_R   = Rlim - Rmid_sep
            shadow_mask = generated_SOL_grid > limiter_R
            output_profile_at_limiter = interp1d(generated_SOL_grid, fitted_profile_SOL,
                                                  fill_value='extrapolate')(limiter_R)
            fitted_profile_SOL[shadow_mask] = output_profile_at_limiter * \
                np.exp(-(generated_SOL_grid[shadow_mask] - limiter_R) / limiter_shadow_decay_length)

        # This is the default minimum value for the density profile
        fitted_profile_SOL[fitted_profile_SOL < 1e16] = 1e16
    else:
        # TEMPERATURE PROFILE
        if stronger_decay_behind_limiter:
            # set a different decay length behind the limiter shadow
            limiter_shadow_decay_length = 0.015  # from Francesco's KN1D OMFIT module
            limiter_R   = Rlim - Rmid_sep
            shadow_mask = generated_SOL_grid > limiter_R
            output_profile_at_limiter = interp1d(generated_SOL_grid, fitted_profile_SOL,
                                                  fill_value='extrapolate')(limiter_R)
            fitted_profile_SOL[shadow_mask] = output_profile_at_limiter * \
                np.exp(-(generated_SOL_grid[shadow_mask] - limiter_R) / limiter_shadow_decay_length)

        # This is the default minimum value for the temperature profile
        fitted_profile_SOL[fitted_profile_SOL < 3] = 3

    if output_R_grid is None:
        output_grid_resolution = 0.001
        output_R_grid = np.arange(-0.1, (R_wall - Rmid_sep) + output_grid_resolution,
                                   output_grid_resolution)

    output_profile_inside_LCFS = interp1d(fitted_R_minus_Rsep, fitted_profile,
                                           fill_value='extrapolate')(output_R_grid[output_R_grid < 0])
    output_profile_SOL = interp1d(generated_SOL_grid, fitted_profile_SOL,
                                   fill_value='extrapolate')(output_R_grid[output_R_grid >= 0])
    output_profile = np.concatenate([output_profile_inside_LCFS, output_profile_SOL])

    return output_R_grid, output_profile, raw_R_minus_Rsep_SOL, raw_values_SOL, raw_errors_SOL


def fit_sol_exponential_francesco(shot, time_ms, device,
                                   psi_grid, fitted_profile,
                                   raw_psi_values, raw_values, raw_errors,
                                   R_wall, R_lim, profile_type, output_R_grid=None):
    '''
    WHAT DOES THIS FUNCTION DO?
    Fits an exponential decay to the SOL profile in R - R_sep space using
    Francesco's coefficients from the KN1D OMFIT module.
    First exponential is inside the limiter, and second exponential is behind
    limiter shadow. Default decay lengths are used here, so no actual fitting
    is done.

    profile_type is either 'ne' or 'Te'
    '''
    dev = _resolve_device(device)

    sorted_raw_order = np.argsort(raw_psi_values)
    raw_psi_values   = raw_psi_values[sorted_raw_order]
    raw_values       = raw_values[sorted_raw_order]
    raw_errors       = raw_errors[sorted_raw_order]

    sep_value = interp1d(psi_grid, fitted_profile, fill_value='extrapolate')(1.0)

    Rmid_raw         = dev.psinorm_to_rmid(shot, time_ms, raw_psi_values)
    Rmid_sep         = float(dev.psinorm_to_rmid(shot, time_ms, np.array([1.0])))
    raw_R_minus_Rsep = Rmid_raw - Rmid_sep

    grid_resolution    = 0.001
    generated_SOL_grid = np.arange(0, (R_wall - Rmid_sep) + grid_resolution, grid_resolution)

    SOL_mask = raw_R_minus_Rsep > -0.001  # allow 1mm inside the separatrix
    raw_R_minus_Rsep_SOL = raw_R_minus_Rsep[SOL_mask]
    raw_values_SOL       = raw_values[SOL_mask]
    raw_errors_SOL       = raw_errors[SOL_mask]

    R_switch_value = R_lim - Rmid_sep
    if profile_type == 'ne':
        a1_value  = 0.04
        a2_value  = 0.025
        min_value = 1e16
    elif profile_type == 'Te':
        a1_value  = 0.025
        a2_value  = 0.015
        min_value = 3
    else:
        raise ValueError("Type must be either 'ne' or 'Te'")

    def double_exponential_decay(x, a1, a2, x_switch):
        x = np.asarray(x, dtype=float)
        return np.where(
            x <= x_switch,
            sep_value * np.exp(-x / a1),
            sep_value * np.exp(-x_switch / a1) * np.exp(-(x - x_switch) / a2)
        )

    fitted_profile_SOL  = double_exponential_decay(generated_SOL_grid, a1_value, a2_value,
                                                    R_switch_value)
    fitted_R_minus_Rsep = dev.psinorm_to_rmid(shot, time_ms, psi_grid) - Rmid_sep
    fitted_profile_SOL[fitted_profile_SOL < min_value] = min_value

    plt.scatter(raw_R_minus_Rsep, raw_values, label='Raw Data', color='red', alpha=0.1)
    plt.errorbar(raw_R_minus_Rsep, raw_values, yerr=raw_errors, fmt='o', color='red',
                 label='Raw Data Error', alpha=0.1)
    plt.scatter(0, sep_value, color='black', marker='x', s=100)
    plt.plot(fitted_R_minus_Rsep, fitted_profile, color='orange', label='Fitted Profile')
    plt.plot(generated_SOL_grid, fitted_profile_SOL, color='black', label='Fitted SOL Profile')
    plt.axvline(0, color='black', linestyle='--', label='Separatrix')
    plt.xlabel('R - Rsep (m)')
    plt.ylabel('Profile Value')
    plt.legend()
    plt.show()

    if output_R_grid is None:
        output_grid_resolution = 0.001
        output_R_grid = np.arange(-0.1, (R_wall - Rmid_sep) + output_grid_resolution,
                                   output_grid_resolution)

    output_profile_inside_LCFS = interp1d(fitted_R_minus_Rsep, fitted_profile,
                                           fill_value='extrapolate')(output_R_grid[output_R_grid < 0])
    output_profile_SOL = interp1d(generated_SOL_grid, fitted_profile_SOL,
                                   fill_value='extrapolate')(output_R_grid[output_R_grid >= 0])
    output_profile = np.concatenate([output_profile_inside_LCFS, output_profile_SOL])

    plt.plot(output_R_grid, output_profile, color='blue', label='Output Profile')
    plt.axvline(0, color='black', linestyle='--', label='Separatrix')
    plt.xlabel('R - Rsep (m)')
    plt.ylabel('Profile Value')
    plt.legend()
    plt.show()

    return output_R_grid, output_profile, raw_R_minus_Rsep_SOL, raw_values_SOL, raw_errors_SOL