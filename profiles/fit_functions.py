### Fitting functions for pedestal profiles ###

import numpy as np


def Osborne_Tanh_linear(x, c0, c1, c2, c3, c4):
    '''
    Osborne tanh with a linear inboard slope and flat outboard (SOL) term.
    c0: pedestal centre, c1: full width, c2: top, c3: bottom, c4: inboard linear
    '''
    z  = 2. * (c0 - x) / c1
    P1 = 1. + c4 * z
    P2 = 1.
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1*E1 - P2*E2) / (E1 + E2))


def Osborne_Tanh_cubic(x, c0, c1, c2, c3, c4, c5, c6):
    '''
    Osborne tanh with linear, quadratic and cubic inboard terms and flat SOL.
    c0: pedestal centre, c1: full width, c2: top, c3: bottom,
    c4: linear, c5: quadratic, c6: cubic inboard terms
    '''
    z  = 2. * (c0 - x) / c1
    P1 = 1. + c4*z + c5*z**2 + c6*z**3
    P2 = 1.
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1*E1 - P2*E2) / (E1 + E2))


def Osborne_Tanh_cubic_linear_SOL(x, c0, c1, c2, c3, c4, c5, c6, c7):
    '''
    Osborne tanh with cubic inboard terms and a linear outboard (SOL) term.
    c7: outboard linear term
    '''
    z  = 2. * (c0 - x) / c1
    P1 = 1. + c4*z + c5*z**2 + c6*z**3
    P2 = 1. + c7*z
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1*E1 - P2*E2) / (E1 + E2))


def Osborne_Tanh_cubic_quadratic_SOL(x, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    '''
    Osborne tanh with cubic inboard terms and a quadratic outboard (SOL) term.
    c7: outboard linear, c8: outboard quadratic
    '''
    z  = 2. * (c0 - x) / c1
    P1 = 1. + c4*z + c5*z**2 + c6*z**3
    P2 = 1. + c7*z + c8*z**2
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1*E1 - P2*E2) / (E1 + E2))


def Cubic(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x**2 + c3*x**3


def Osborne_linear_initial_guesses(psi_edge, values_edge, n_params=7):
    '''
    Rough initial guesses for an Osborne tanh fit based on edge-only data.
    Returns a list of length n_params with trailing zeros for polynomial terms.
    '''
    if len(values_edge) < 4:
        return [1.0, 0.04, float(np.nanmax(values_edge)), 0.0] + [0.0] * (n_params - 4)

    avg    = np.nanmean(values_edge)
    bottom = avg * 0.3
    top    = avg * 1.1

    max_r = psi_edge[-1]
    min_r = psi_edge[-1]
    for i in range(len(psi_edge) - 1, -1, -1):
        if values_edge[i] > bottom:
            max_r = psi_edge[i]
            break
    for i in range(len(psi_edge) - 1, -1, -1):
        if values_edge[i] > top:
            min_r = psi_edge[i]
            break

    width  = max_r - min_r
    if width <= 0:
        width = abs(psi_edge[-3] - psi_edge[-5]) if len(psi_edge) > 5 else 0.05
    centre = (max_r + min_r) / 2.0

    return [centre, width, top, bottom] + [0.0] * (n_params - 4)


# Map (core_order, sol_order) to the fit function and number of parameters.
# Used by master_fit to select the right function from the order arguments.
_FIT_FUNCTION_MAP = {
    (3, 0): (Osborne_Tanh_cubic,               7),
    (3, 1): (Osborne_Tanh_cubic_linear_SOL,    8),
    (3, 2): (Osborne_Tanh_cubic_quadratic_SOL, 9),
}


def get_fit_function(core_order, sol_order):
    '''
    Return (fit_function, n_params) for the given polynomial orders.
    core_order: polynomial order for the inboard region (only 3 currently supported)
    sol_order:  polynomial order for the outboard SOL (0, 1, or 2)
    '''
    key = (core_order, sol_order)
    if key not in _FIT_FUNCTION_MAP:
        raise ValueError(
            f"No fit function for core_order={core_order}, sol_order={sol_order}. "
            f"Supported: {list(_FIT_FUNCTION_MAP.keys())}"
        )
    return _FIT_FUNCTION_MAP[key]
