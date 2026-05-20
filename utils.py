import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')


def chi_squared(ydata, yfit, yerr):
    return np.sum((ydata - yfit) ** 2 / yerr ** 2)


def reduced_chi_squared(ydata, yfit, yerr, num_params):
    return chi_squared(ydata, yfit, yerr) / (len(ydata) - num_params)


def reduced_chi_squared_inside_separatrix(psi, ydata, yfit, yerr, num_params,
                                           only_edge=False):
    """Chi-squared evaluated only inside the separatrix (psi < 1).
    If only_edge=True, restrict further to 0.6 < psi < 1.0."""
    if only_edge:
        mask = (psi > 0.6) & (psi < 1.0)
    else:
        mask = psi < 1.0
    n = np.sum(mask)
    if n <= num_params:
        return np.inf
    return chi_squared(ydata[mask], yfit[mask], yerr[mask]) / (n - num_params)


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def remove_zeros_ne_te(psi, ne, te, ne_err, te_err, core_only=True):
    """Remove points where ne or te is zero. If core_only, only remove inside
    the separatrix (psi < 1) so physical SOL zeros are preserved."""
    if core_only:
        mask = ((ne != 0) & (te != 0)) | (psi >= 1.0)
    else:
        mask = (ne != 0) & (te != 0)
    return psi[mask], ne[mask], te[mask], ne_err[mask], te_err[mask]


def add_sol_zero(psi, values, errors, psi_sol=1.05):
    """Append a zero-valued anchor point in the SOL to help the fit decay."""
    psi = np.append(psi, psi_sol)
    values = np.append(values, 0.0)
    errors = np.append(errors, 2*np.mean(errors))
    breakpoint()
    return psi, values, errors


# ---------------------------------------------------------------------------
# Post-fit shift to align separatrix Te with the 2-point model
# ---------------------------------------------------------------------------

def apply_2pt_shift(psi, te_fit, te_sep, only_edge=True):
    """Shift the psi axis so that the fitted Te at the separatrix equals te_sep.

    Parameters
    ----------
    psi : 1D array of psi values (output grid)
    te_fit : 1D array of fitted Te values on that grid
    te_sep : target Te at the separatrix (eV) from the 2-point model
    only_edge : if True only shift points with psi > 0.8

    Returns
    -------
    shifted_psi : shifted psi array
    shift : scalar shift applied (positive = move separatrix outward)
    """
    shift = 0.0
    for i in range(1, len(psi)):
        if te_fit[i] < te_sep:
            frac = (te_fit[i - 1] - te_sep) / (te_fit[i - 1] - te_fit[i])
            new_sep = psi[i - 1] + (psi[i] - psi[i - 1]) * frac
            shift = 1.0 - new_sep
            break

    shifted = psi.copy()
    if only_edge:
        shifted[psi > 0.8] += shift
    else:
        shifted += shift
    return shifted, shift
