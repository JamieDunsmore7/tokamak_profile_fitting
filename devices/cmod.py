"""Alcator C-Mod device module.

Provides the four functions that master_fit expects from any device module:
  get_thomson_data(shot, t_min, t_max)
  get_Te_sep(shot, time_ms)
  psinorm_to_rmid(shot, time_ms, psi)
  get_min_errorbars()

Also provides scale_to_tci(shot, times_ms, ne_core), which is C-Mod specific.
"""

import numpy as np
import MDSplus
import eqtools
import eqtools.CModEFIT
from scipy.interpolate import interp1d

from ..utils import smooth


MIN_NE_ERRORBAR = 2e19   # m^-3
MIN_TE_ERRORBAR = 10.0   # eV


def get_min_errorbars():
    return {'ne': MIN_NE_ERRORBAR, 'te': MIN_TE_ERRORBAR}


# ---------------------------------------------------------------------------
# Thomson data
# ---------------------------------------------------------------------------

def get_thomson_data(shot, t_min, t_max, coordinate='psinorm'):
    """Fetch core + edge Thomson data and map to a normalised radial coordinate.

    Parameters
    ----------
    coordinate : 'psinorm' (default) or 'sqrtphinorm'
        Passed directly to eqtools rho2rho as the target coordinate.

    Returns a dict with keys:
      times_ms  (N_t,)
      psi       (N_spatial, N_t)   values in the chosen coordinate
      ne        (N_spatial, N_t)  [m^-3]
      ne_err    (N_spatial, N_t)
      te        (N_spatial, N_t)  [eV]
      te_err    (N_spatial, N_t)
      edge_mask (N_spatial,) bool — True for edge TS points
    """
    tree = MDSplus.Tree('CMOD', shot)

    # edge Thomson
    te_e     = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:TE').data()
    te_err_e = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:TE:ERROR').data()
    ne_e     = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE').data()
    ne_err_e = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE:ERROR').data()
    rmid_e   = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:RMID').data()
    t_raw    = tree.getNode('\\TOP.ELECTRONS.YAG_EDGETS.RESULTS:NE').dim_of().data()

    times_ms = np.round(t_raw * 1000).astype(int)
    mask     = (times_ms > t_min) & (times_ms < t_max)
    times_ms = times_ms[mask]
    te_e, te_err_e = te_e[:, mask],  te_err_e[:, mask]
    ne_e, ne_err_e = ne_e[:, mask],  ne_err_e[:, mask]
    rmid_e         = rmid_e[:, mask]

    # core Thomson
    if shot > 1020000000:
        te_c     = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:TE_RZ').data() * 1000
        te_err_c = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:TE_ERR').data() * 1000
        ne_c     = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').data()
        ne_err_c = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_ERR').data()
        rmid_c   = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:R_MID_T').data()
        t_core   = tree.getNode('\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ').dim_of().data()
    else:
        te_c     = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:TE_RZ_T').data() * 1000
        te_err_c = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:TE_ERR_ZT').data() * 1000
        ne_c     = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T').data()
        ne_err_c = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_ERR_ZT').data()
        rmid_c   = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:R_MID_T').data()
        t_core   = tree.getNode('\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T').dim_of().data()

    core_times_ms = np.round(t_core * 1000).astype(int)
    core_mask     = (core_times_ms > t_min) & (core_times_ms < t_max)
    if not np.array_equal(times_ms, core_times_ms[core_mask]):
        raise ValueError('Core and edge Thomson time bases do not match.')

    te_c, te_err_c = te_c[:, core_mask],  te_err_c[:, core_mask]
    ne_c, ne_err_c = ne_c[:, core_mask],  ne_err_c[:, core_mask]
    rmid_c         = rmid_c[:, core_mask]

    # equilibrium (created once for all time points)
    try:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except Exception:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')

    n_c, n_e = ne_c.shape[0], ne_e.shape[0]
    n_total  = n_c + n_e
    n_times  = len(times_ms)

    psi    = np.zeros((n_total, n_times))
    ne     = np.zeros_like(psi)
    ne_err = np.zeros_like(psi)
    te     = np.zeros_like(psi)
    te_err = np.zeros_like(psi)

    for t_idx, time_ms in enumerate(times_ms):
        t_s   = time_ms / 1000.0
        psi_c = eq.rho2rho('Rmid', coordinate, rmid_c[:, t_idx], t_s)
        psi_e = eq.rho2rho('Rmid', coordinate, rmid_e[:, t_idx], t_s)

        psi[:n_c, t_idx]    = psi_c;    psi[n_c:, t_idx]    = psi_e
        ne[:n_c,  t_idx]    = ne_c[:,  t_idx]; ne[n_c:,  t_idx]  = ne_e[:,  t_idx]
        ne_err[:n_c, t_idx] = ne_err_c[:, t_idx]; ne_err[n_c:, t_idx] = ne_err_e[:, t_idx]
        te[:n_c,  t_idx]    = te_c[:,  t_idx]; te[n_c:,  t_idx]  = te_e[:,  t_idx]
        te_err[:n_c, t_idx] = te_err_c[:, t_idx]; te_err[n_c:, t_idx] = te_err_e[:, t_idx]

    edge_mask = np.zeros(n_total, dtype=bool)
    edge_mask[n_c:] = True

    return dict(times_ms=times_ms, psi=psi, ne=ne, ne_err=ne_err,
                te=te, te_err=te_err, edge_mask=edge_mask)


# ---------------------------------------------------------------------------
# Separatrix Te (2-point model)
# ---------------------------------------------------------------------------

def get_Te_sep(shot, time_ms):
    """Brunner-scaling 2-point model for the C-Mod separatrix Te (eV)."""
    return _two_point_model(shot, time_ms / 1000.0)


def _two_point_model(shot, time_s):
    tree = MDSplus.Tree('cmod', shot)

    def _interp(path):
        node = tree.getNode(path)
        return float(np.interp(time_s, node.dim_of().data(), node.data()))

    try:
        P_rad = _interp('\\top.spectroscopy.bolometer:results:foil:main_power') / 1e6
    except Exception:
        P_rad = _interp('\\top.spectroscopy.bolometer:twopi_diode') * 3 * 1000 / 1e6

    P_RF = _interp('\\top.RF.antenna.results:pwr_net_tot')
    P_oh = float(np.interp(time_s, *_get_P_ohmic(shot)))

    W_node = tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:WPLASM')
    W_t = W_node.dim_of().data()
    W   = W_node.data() / 1e6
    dW_dt = float(np.interp(time_s, W_t, np.gradient(W, W_t)))

    P_RF_efficiency = 0.8 # value suggested by paper by Bonoli. This is the efficiency of the RF heating. NOTE: it may not always be 0.8


    Psol = P_RF_efficiency * P_RF + P_oh - dW_dt - P_rad
    if Psol < 0:
        print('Psol < 0 — 2pt model unreliable, returning 60 eV')
        return 60.0

    BTaxis = _interp('\\top.MHD.magnetics.diamag_coils:btor')
    betat  = _interp('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:betat')
    p_vol  = (betat / 100) * BTaxis ** 2 / (2.0 * 4.0 * np.pi * 1e-7)
    q95    = _interp('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:qpsib')

    try:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except Exception:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')

    Rlcfs = float(eq.rho2rho('psinorm', 'Rmid', 1, time_s))
    Bt    = abs(float(eq.rz2BT(Rlcfs, 0, time_s)))
    Bp    = abs(float(eq.rz2BZ(Rlcfs, 0, time_s)))

    lam_q_mm = (0.08 / p_vol) ** 0.5 * 1e3
    L_par    = np.pi * Rlcfs * q95
    q_par    = (0.5 * Psol / (2.0 * np.pi * Rlcfs * lam_q_mm * 1e-3)
                * np.hypot(Bt, Bp) / Bp)
    return float(((7.0 / 2.0) * q_par * 1e6 * L_par / (2.0 * 2000.0)) ** (2.0 / 7.0))


def _get_P_ohmic(shot):
    tree   = MDSplus.Tree('cmod', shot)
    node   = tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.G_EQDSK.SSIBRY')
    t      = node.dim_of(0).data()
    psi_b  = node.data()
    vsurf  = np.gradient(smooth(psi_b, 5), t) * 2 * np.pi
    ip     = np.abs(tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:CPASMA').data())
    li     = tree.getNode('\\top.MHD.ANALYSIS.EFIT.RESULTS.A_EQDSK:ali').data()
    L      = li * 2 * np.pi * 67.0 * 1e-9
    vi     = L * np.gradient(smooth(ip, 2), t)
    return t, ip * (vsurf - vi) / 1e6


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def psinorm_to_rmid(shot, time_ms, psi):
    """Map psi_N values to outboard midplane R (m)."""
    try:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except Exception:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')
    return eq.rho2rho('psinorm', 'Rmid', psi, time_ms / 1000.0)


# ---------------------------------------------------------------------------
# TCI scaling (C-Mod specific, optional)
# ---------------------------------------------------------------------------

def scale_to_tci(shot, times_ms, ne_core):
    """Scale core Thomson ne to match TCI interferometry (computed over 0.5–1.5 s)."""
    tree    = MDSplus.Tree('cmod', shot)
    ip_node = tree.getNode('\\ip')
    ip_t    = ip_node.dim_of().data()
    ip_data = ip_node.data()

    idx = np.where((ip_t > 0.5) & (ip_t < 1.5))[0]
    if len(idx) == 0 or abs(ip_data[idx[0]]) < 0.4e6 or abs(ip_data[idx[-1]]) < 0.4e6:
        print('Ip < 0.4 MA over 0.5–1.5 s; skipping TCI scaling.')
        return ne_core

    mf1, mf2, _ = _get_ts_tci_ratio(shot, 0.5, 1.5)
    n_yag1, n_yag2, idx1, idx2 = _parse_yags(shot)

    ts_node = tree.getNode(
        '\\TOP.ELECTRONS.YAG_NEW.RESULTS.PROFILES:NE_RZ'
        if shot > 1020000000 else
        '\\TOP.ELECTRONS.YAG.RESULTS.GLOBAL.PROFILE:NE_RZ_T'
    )
    ts_t = ts_node.dim_of().data()

    laser1_ms = np.round(ts_t[idx1] * 1000).astype(int) if n_yag1 > 0 else np.array([], dtype=int)
    laser2_ms = np.round(ts_t[idx2] * 1000).astype(int) if n_yag2 > 0 else np.array([], dtype=int)

    ne_out = ne_core.copy()
    for col, t in enumerate(times_ms):
        if t in laser1_ms:
            ne_out[:, col] *= mf1
        elif t in laser2_ms:
            ne_out[:, col] *= mf2
        else:
            raise ValueError(f'Cannot match t={t} ms to either laser for TCI scaling.')
    return ne_out


def _get_ts_tci_ratio(shot, tmin, tmax, nl_num=4):
    nl_ts1, nl_ts2, nl_tci1, nl_tci2, t1, t2 = _compare_ts_tci(shot, tmin, tmax, nl_num)
    nl_ts1 = np.where(nl_ts1 == 0, np.nan, nl_ts1)
    nl_ts2 = np.where(nl_ts2 == 0, np.nan, nl_ts2)
    r1, r2  = nl_tci1 / nl_ts1, nl_tci2 / nl_ts2
    in1 = (t1 > tmin) & (t1 < tmax)
    in2 = (t2 > tmin) & (t2 < tmax)
    mf1 = float(np.nanmean(r1[in1])) if np.any(in1) else 1.0
    mf2 = float(np.nanmean(r2[in2])) if np.any(in2) else 1.0
    return mf1, mf2, (mf1 + mf2) / 2.0


def _compare_ts_tci(shot, tmin, tmax, nl_num=4):
    electrons = MDSplus.Tree('electrons', shot)
    ch_str    = f'0{nl_num}' if nl_num < 10 else str(nl_num)
    try:
        ts_t = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        _ = len(ts_t)
    except Exception:
        ts_t = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0).data()

    tci   = electrons.getNode(f'\\electrons::top.tci.results:nl_{ch_str}').data().flatten()
    tci_t = electrons.getNode(f'\\electrons::top.tci.results:nl_{ch_str}').dim_of(0).data()

    nl_ts, nl_ts_t = _integrate_ts_to_tci(shot, tmin, tmax, nl_num)
    t0, t1_edge    = np.min(nl_ts_t), np.max(nl_ts_t)
    tci_fn = interp1d(tci_t, tci, bounds_error=False, fill_value=np.nan)
    ts_fn  = interp1d(nl_ts_t, nl_ts)

    n_yag1, n_yag2, idx1, idx2 = _parse_yags(shot)
    nan_arr = np.array([np.nan])
    nl_ts1 = nl_ts2 = nl_tci1 = nl_tci2 = t1_arr = t2_arr = nan_arr

    if n_yag1 > 0:
        ts_t1 = ts_t[idx1]
        sel   = (ts_t1 >= t0) & (ts_t1 <= t1_edge)
        if np.any(sel):
            t1_arr  = ts_t1[sel]
            nl_tci1 = tci_fn(t1_arr)
            nl_ts1  = ts_fn(t1_arr)

    if n_yag2 > 0:
        ts_t2 = ts_t[idx2]
        sel   = (ts_t2 >= t0) & (ts_t2 <= t1_edge)
        if np.any(sel):
            t2_arr  = ts_t2[sel]
            nl_tci2 = tci_fn(t2_arr)
            nl_ts2  = ts_fn(t2_arr)

    return nl_ts1, nl_ts2, nl_tci1, nl_tci2, t1_arr, t2_arr


def _integrate_ts_to_tci(shot, tmin, tmax, nl_num=4):
    from scipy.integrate import cumtrapz
    t, z, ne, ne_sig = _map_ts_to_tci(shot, tmin, tmax, nl_num)
    nl = np.zeros(len(t))

    def _uniq(x):
        mask = [True]
        for i in range(1, len(x)):
            mask.append(x[i] != x[i - 1])
        return np.array(mask)

    for i in range(len(t)):
        ne_sig[i, :] = np.where(ne_sig[i, :] == 0, np.nan, ne_sig[i, :])
        ok = (np.abs(z[i, :]) < 0.5) & (ne[i, :] > 0) & (ne[i, :] < 1e21) \
             & (ne[i, :] / ne_sig[i, :] > 2)
        if np.sum(ok) < 3:
            continue
        x_u = z[i, ok][_uniq(z[i, ok])]
        y_u = ne[i, ok][_uniq(z[i, ok])]
        nl[i] = cumtrapz(y_u, x_u)[-1]
    return nl, t


def _map_ts_to_tci(shot, tmin, tmax, nl_num=4):
    electrons = MDSplus.Tree('electrons', shot)
    analysis  = MDSplus.Tree('analysis', shot)
    psi_a_t   = analysis.getNode('\\efit_aeqdsk:sibdry').dim_of(0).data()
    t1, t2    = np.min(psi_a_t), np.max(psi_a_t)

    try:
        t_ts    = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        ne_core = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').data()
        ne_cerr = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_err').data()
        z_core  = electrons.getNode('\\electrons::top.yag_new.results.profiles:z_sorted').data()
    except Exception:
        t_ts    = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0).data()
        ne_core = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').data()
        ne_cerr = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_err_zt').data()
        z_core  = electrons.getNode('\\electrons::top.yag.results.global.profile:z_sorted').data()

    ne_edge  = electrons.getNode('\\electrons::top.yag_edgets.results:ne').data()
    ne_eedge = electrons.getNode('\\electrons::top.yag_edgets.results:ne:error').data()
    z_edge   = electrons.getNode('\\electrons::top.yag_edgets.data:fiber_z').data()
    r_ts     = electrons.getNode('\\electrons::top.yag.results.param:r').data()
    r_tci_all = electrons.getNode('\\electrons::top.tci.results:rad').data()

    m_c, m_e = len(z_core), len(z_edge)
    m_ts     = m_c + m_e
    z_ts     = np.concatenate([z_core, z_edge])

    ind  = (t_ts > t1) & (t_ts < t2)
    t_ts = t_ts[ind]
    n_ts = len(t_ts)
    ne   = np.zeros((n_ts, m_ts));  ne_sig = np.zeros_like(ne)
    ne[:, :m_c]  = np.transpose(ne_core)[ind, :]
    ne[:, m_c:]  = np.transpose(ne_edge)[ind, :]
    ne_sig[:, :m_c] = np.transpose(ne_cerr)[ind, :]
    ne_sig[:, m_c:] = np.transpose(ne_eedge)[ind, :]

    try:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='EFIT20')
    except Exception:
        eq = eqtools.CModEFIT.CModEFITTree(int(shot), tree='ANALYSIS')

    r_full   = np.full(m_ts, r_ts)
    psin_ts  = eq.rho2rho('RZ', 'psinorm', r_full, z_ts, t=t_ts, each_t=True)
    m_tci    = 501
    z_tci    = -0.4 + 0.8 * np.linspace(1, m_tci - 1, m_tci) / float(m_tci - 1)
    r_tci    = r_tci_all[nl_num - 1] + np.zeros(m_tci)
    psin_tci = eq.rho2rho('RZ', 'psinorm', r_tci, z_tci, t=t_ts, each_t=True)

    z_map  = np.zeros((n_ts, 2 * m_ts))
    ne_map = np.zeros_like(z_map);  ne_map_sig = np.zeros_like(z_map)

    for i in range(n_ts):
        i_min = np.argmin(psin_tci[i, :])
        for j in range(m_ts):
            try:
                a1 = interp1d(psin_tci[i, :i_min + 1], z_tci[:i_min + 1])(psin_ts[i, j])
                a2 = interp1d(psin_tci[i, i_min + 1:], z_tci[i_min + 1:])(psin_ts[i, j])
            except Exception:
                a1 = a2 = np.nan
            z_map[i, [j, j + m_ts]]      = [a1, a2]
            ne_map[i, [j, j + m_ts]]     = ne[i, j]
            ne_map_sig[i, [j, j + m_ts]] = ne_sig[i, j]
        order = np.argsort(z_map[i, :])
        z_map[i, :]      = z_map[i, order]
        ne_map[i, :]     = ne_map[i, order]
        ne_map_sig[i, :] = ne_map_sig[i, order]

    return t_ts, z_map, ne_map, ne_map_sig


def _parse_yags(shot):
    electrons = MDSplus.Tree('electrons', shot)
    try:
        n1 = int(electrons.getNode('\\knobs:pulses_q').data())
    except Exception:
        n1 = 0
    try:
        n2 = int(electrons.getNode('\\knobs:pulses_q_2').data())
    except Exception:
        n2 = 0

    dark    = int(electrons.getNode('\\n_dark_prior').data())
    n_total = int(electrons.getNode('\\n_total').data())
    n_t     = n_total - dark

    if n1 == 0 and n2 == 0:
        return 0, 0, np.array([], dtype=int), np.array([], dtype=int)
    elif n1 == 0:
        return 0, n2, np.array([], dtype=int), np.arange(n2, dtype=int)
    elif n2 == 0:
        return n1, 0, np.arange(n1, dtype=int), np.array([], dtype=int)
    elif n1 == n2:
        i1 = 2 * np.arange(n1)
        i2 = i1 + 1
    elif n1 < n2:
        i1 = 2 * np.arange(n1)
        i2 = np.concatenate([i1 + 1, 2 * n1 + np.arange(n2 - n1)])
    else:
        i2 = 2 * np.arange(n2)
        i1 = np.concatenate([i2, 2 * n2 + np.arange(n1 - n2)])

    i1 = i1[i1 < n_t].astype(int)
    i2 = i2[i2 < n_t].astype(int)
    return n1, n2, i1, i2
