"""v10 transform (2026-10-02). Ported from t2-work/candidates/v10_final.py (see its docstring for every component and
constant). History lookups read only the unit's own panels: c['H'](asset) -> Series up to the as-of date;
c['rates'] -> rates-panel levels (UST tenors) up to the as-of date."""
import numpy as np
import pandas as pd

RO = {'MKT': -1, 'SMB': -1, 'QMJ': 1, 'AUD': -1, 'NZD': -1, 'EUR': -1, 'GBP': -1, 'CAD': 1, 'NOK': 1, 'SEK': 1, 'DKK': 1, 'JPY': -1, 'CHF': -1}

NORM = {'lv20': (-0.115, 0.33), 'lv60': (-0.06, 0.25), 'lvm': (-0.09, 0.28), 'lt60': (-0.22, 0.68)}
TAU_GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 25))
TEN = {'UST_2Y': 2.0, 'UST_5Y': 5.0, 'UST_7Y': 7.0, 'UST_10Y': 10.0, 'UST_20Y': 20.0, 'UST_30Y': 30.0}
TERM_NORM = {'ldzc': (-0.7258, 0.6331), 'cabs': (0.0, 0.2654), 'lvlN': (0.0, 0.3095), 'lvl': (0.0552, 0.3201)}
NMAX = 3300
C3 = 0.5
ZA0, ZA1, ZS0, ZS1 = 0.75, 1.75, 2.2, 1.2
FAC = ('MKT', 'SMB', 'HML', 'MOM', 'QMJ', 'BAB')
FXSET = ('AUD', 'NZD', 'EUR', 'GBP', 'CAD', 'NOK', 'SEK', 'DKK', 'JPY', 'CHF')

# v10 per-family components (see docstring): ksl, lamS (pca), DP (f2f3), CV / KM (factor), fx_cap (wild)
CFG = {
    'F1': dict(CV=0.2, fx_cap=1.0),
    'F2': dict(ksl=0.2, DP=(0.75, 0.02, 0.09)),
    'F3': dict(ksl=0.3, lamS=0.5, DP=(0.75, 0.02, 0.09)),
    'F4': dict(CV=0.2, KM=0.7),
}


def _fam(**kw):
    q = dict(k=0.5, c=1.0, ct=1.3, ct2=1.3, beta=0.25, vlo=0.7, vhi=1.4, a=0.0, b=0.0, slope=0.0,
             rule=('lvm', 0.0, 0.0, 0.1, 0.0), shiftadj=False,
             G0=1.0, PW=0.5, CAP=2.5, gslope=0.0, rtail=0.0, ltail=0.0, LAM=0.5, TEXP=0.5,
             wv_ldzc=0.0, a_cabs=0.0, a_lvlN=0.0, wv_lvlN=0.0, m_lvl=0.0, xs=())
    q.update(kw)
    return q


P = {
    'F1': _fam(k=0.625, c=0.5, ct=1.0, ct2=1.15, a=0.1, b=-0.05, slope=0.05, rule=('lvm', 0.069, 0.0645, 0.1, 0.0),
               G0=0.75, ltail=0.3, LAM=0.3, wv_ldzc=0.1, a_cabs=0.0625, xs=(('u10', 750, -0.07),)),
    'F2': _fam(k=0.5, c=1.0, ct=1.3, ct2=1.3, rule=('lvm', -0.026, 0.178, 0.1, 0.0),
               G0=1.0, rtail=0.5, a_lvlN=0.3, wv_lvlN=0.125, xs=(('u10', 500, -0.12),)),
    'F3': _fam(k=0.375, c=0.9, ct=1.3, ct2=1.3, rule=('lvm', 0.094, 0.0, 0.1, 0.0),
               G0=1.0, rtail=0.5, gslope=0.3, xs=(('u10', 750, -0.135), ('lvpm', None, 0.36))),
    'F4': _fam(k=0.625, c=1.6, ct=1.5, ct2=1.7, beta=0.4375, a=0.3, b=0.25, slope=0.106, rule=('lv60', 0.0608, 0.3735, 0.1, 0.5),
               shiftadj=True, G0=1.4, PW=0.0, CAP=4.0, gslope=0.3, rtail=0.5, m_lvl=0.1),
}


# ---------------------------------------------------------------- as-of histories

def _steps(s):
    dt = s.index.to_series().diff().dt.days
    hole = max(10 * dt.iloc[1:].median(), 5) if len(s) > 2 else 5
    st = s.diff()
    st[dt > hole] = np.nan
    return st.iloc[1:]


def steps_of(c, a):
    """last <= 3300 clean step changes of asset a up to as-of (KeyError if the unit has no such asset)"""
    s = c['H'](a).iloc[-(NMAX + 60):]
    return _steps(s).dropna().to_numpy()[-NMAX:]


def ma1500(c):
    out = []
    for a, h in c['cells']:
        past = c['H'](a).iloc[-1500:]
        out.append(past.mean() if len(past) >= 500 else np.nan)
    return np.array(out)


def lvl_of(c, j):
    a, h = c['cells'][j]
    s = c['H'](a).iloc[-1700:]
    dt = s.index.to_series().diff().dt.days
    hole = max(10 * dt.iloc[1:].median(), 5) if len(s) > 2 else 5
    st = s.diff() if c['ttype'] == 'level' else s.copy()
    st[dt > hole] = np.nan
    st = st.iloc[1:].dropna().to_numpy()
    return np.log(max(st[-1500:].std(), 1e-300) / max(st[-300:].std(), 1e-300)) if len(st) >= 750 else 0.0


def _rates(c):
    """rates panel (UST levels) up to as-of, forward-filled; None if absent or stale (> 8 days before as-of)"""
    if '_rp' not in c:
        R = c.get('rates')
        out = None
        if R is not None and not R.empty and 'UST_10Y' in R.columns and 'UST_2Y' in R.columns:
            Lv = R.sort_index().ffill()
            if (pd.Timestamp(c['asof']) - Lv.index[-1]).days <= 8 and len(Lv) > 2:
                W = Lv.diff().iloc[1:].fillna(0.0)
                out = dict(Lv=Lv.iloc[1:], W=W)
        c['_rp'] = out
    return c['_rp']


def rates_feats(c, ns):
    rp = _rates(c)
    if rp is None:
        return {}
    W = rp['W']
    cs = W.cumsum().to_numpy(); vol = W.rolling(300, min_periods=200).std().to_numpy()
    v20 = W.rolling(20, min_periods=15).std().to_numpy(); v60 = W.rolling(60, min_periods=45).std().to_numpy()
    i = len(W) - 1; cols = list(W.columns); j10 = cols.index('UST_10Y')
    v = vol[i]; ok = v > 0
    out = {}
    for n in ns:
        if i >= n + 1:
            out[f'u10_{n}'] = float((cs[i, j10] - cs[i - n, j10]) / (v[j10] * np.sqrt(n))) if v[j10] > 0 else 0.0
    lv = [np.where(ok & (arr[i] > 0), np.log(np.where(arr[i] > 0, arr[i], 1.0) / np.where(ok, v, 1.0)), 0.0) for arr in (v20, v60)]
    out['lvpm'] = float(np.mean(0.5 * (lv[0] + lv[1])))
    return out


def reversal(c, sd, q, w=0.0, lamS=0.0):
    """(drift after reversal, correction in horizon-sd units cr = -corr/sd)"""
    d = np.array(c['drift'], float).copy()
    cr = np.zeros(len(d))
    if c['ttype'] != 'level':
        return d, cr
    mv = None
    if lamS and len({a for a, h in c['cells'] if a.startswith('UST')}) >= 2:
        mm = {}
        for a in TEN:
            try:
                st = steps_of(c, a)
            except KeyError:
                continue
            if len(st) >= 900:
                mm[a] = float(st[-600:-300].mean() + C3 * st[-900:-600].mean())
        if len(mm) >= 3:
            xs_ = np.log(np.array([TEN[a] for a in mm])); vals = np.array(list(mm.values()))
            xm = xs_.mean(); xc = xs_ - xm; den = (xc ** 2).sum()
            b0 = (xc * (vals - vals.mean())).sum() / den if den > 0 else 0.0
            mv = (vals.mean(), b0, xm)
    for j, (a, h) in enumerate(c['cells']):
        if a.startswith('UST'):
            st = steps_of(c, a)
            if len(st) >= 900:
                g = q['G0'] * (21.0 / h) ** q['PW']
                if q['gslope']:
                    g *= max(1.0 + q['gslope'] * np.log(10.0 / TEN[a]) / np.log(5.0), 0.0)
                g *= 1.0 - w
                cap = q['CAP'] * sd[j]
                mj = st[-600:-300].mean() + C3 * st[-900:-600].mean()
                if mv is not None:
                    a0_, b0_, xm_ = mv
                    xj = np.log(TEN[a]) - xm_
                    mj = a0_ + (h / 63.0) ** lamS * b0_ * xj + (mj - a0_ - b0_ * xj)
                corr = float(np.clip(g * h * mj, -cap, cap))
                d[j] -= corr
                cr[j] = -corr / sd[j]
    return d, cr



def _ss(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def zlb_w(c):
    """low-rate steep-curve weight in [0,1] from UST_2Y and UST_10Y levels as of the as-of date (0 if unavailable)"""
    rp = _rates(c)
    if rp is None:
        return 0.0
    y2 = float(rp['Lv']['UST_2Y'].iloc[-1]); y10 = float(rp['Lv']['UST_10Y'].iloc[-1])
    if not (np.isfinite(y2) and np.isfinite(y10)):
        return 0.0
    return float((1.0 - _ss((y2 - ZA0) / (ZA1 - ZA0))) * (1.0 - _ss((ZS0 - (y10 - y2)) / (ZS0 - ZS1))))


def xasset_shift(c, q, sd, isU):
    xs = np.zeros(len(isU))
    if not q['xs']:
        return xs
    f = rates_feats(c, (500, 750))
    for j in np.where(isU > 0)[0]:
        v = 0.0
        for name, n, coef in q['xs']:
            if name == 'u10':
                x = f.get(f'u10_{n}')
            else:
                x = f.get('lvpm')
                if x is not None:
                    ds = c['drift'][j] / sd[j]
                    x = x * ds / (abs(ds) + 0.1)
            if x is not None and np.isfinite(x):
                v += coef * float(np.clip(x, -3.0, 3.0))
        xs[j] = v * (c['hs'][j] / 63.0) ** 0.5
    return xs



def _x(f, key):
    cen, sc = TERM_NORM[key]
    return np.clip((f - cen) / sc, -2.5, 2.5)


def transform_card(c):
    fam = c['fam']; q = P[fam]; cf = CFG[fam]
    base = c['base']; sd0 = base.std(0, keepdims=True); sd = np.where(sd0 > 0, sd0, 1)
    z = base / sd
    az = np.abs(z); sgn = np.sign(z)
    cells = c['cells']
    d = base.shape[1]
    feat = c['feat']
    dr0 = np.asarray(c['drift'], float)
    ro = np.array([RO.get(x, 0) for x, h in cells], float)[None, :]
    isU = np.array([1.0 if (x.startswith('UST') and c['ttype'] == 'level') else 0.0 for x, h in cells])
    isr = isU > 0
    v300 = np.array([f['v300'] for f in feat]); ok = v300 > 0; v3 = np.where(ok, v300, 1.0)

    wz = zlb_w(c) if isr.any() else 0.0
    dadj, crs = reversal(c, sd[0], q, wz, cf.get('lamS', 0.0))                  # 2
    isf = np.array([1.0 if (x in FAC and c['ttype'] == 'log_return') else 0.0 for x, h in cells])
    isF = np.array([1.0 if x in FXSET else 0.0 for x, h in cells])

    m = 1 + q['a'] * sgn * ro                                                   # 1
    cm = 1.0
    if q['slope']:
        r60 = np.array([f['r60'] for f in feat])
        lt = np.log(np.where(ok, np.abs(r60) / (v3 * np.sqrt(60)), 0.0) + 0.15)
        cm = np.exp(q['slope'] * np.clip((lt - NORM['lt60'][0]) / NORM['lt60'][1], -2.5, 2.5))[None, :]
    if q['rtail'] and isr.any():                                                # 3
        m = m * (1 + q['rtail'] * sgn * np.clip(crs, -1, 1)[None, :])
    if q['ltail'] and isr.any():                                                # 4
        ma = ma1500(c)
        ma = np.where(np.isfinite(ma), ma, c['anchor'])
        m = m * (1 + q['ltail'] * (1.0 - wz) * sgn * np.where(isr, np.clip((ma - c['anchor']) / 3.0, -1, 1), 0.0)[None, :])

    fe, sa, sb, eps, gam = q['rule']                                            # 5
    lv20 = np.where(ok, np.log(np.maximum([f['v20'] for f in feat], 1e-300) / v3), 0.0)
    lv60 = np.where(ok, np.log(np.maximum([f['v60'] for f in feat], 1e-300) / v3), 0.0)
    lv = {'lv20': lv20, 'lv60': lv60, 'lvm': 0.5 * (lv20 + lv60)}[fe]
    dr = dadj if q['shiftadj'] else dr0
    shift = (sa + sb * np.clip((lv - NORM[fe][0]) / NORM[fe][1], -2.5, 2.5)) * dr / (np.abs(dr) + eps * sd[0] + 1e-300)
    if gam:
        shift = shift * (np.asarray(c['hs'], float) / 63.0) ** gam

    lvm = np.zeros(d)                                                           # 6
    dirdr = dr0 / (np.abs(dr0) + 0.1 * sd[0] + 1e-300)
    if q['wv_ldzc']:
        lvm += q['wv_ldzc'] * _x(np.log(np.abs(dadj / sd[0]) + 0.15), 'ldzc')
    if q['a_cabs']:
        shift = shift + q['a_cabs'] * _x(np.abs(crs), 'cabs') * dirdr
    if q['a_lvlN'] or q['wv_lvlN'] or q['m_lvl']:
        lvl = np.array([lvl_of(c, j) for j in range(d)])
        if q['a_lvlN'] or q['wv_lvlN']:
            xN = _x(lvl * (1 - isU), 'lvlN')
            shift = shift + q['a_lvlN'] * xN * dirdr
            lvm += q['wv_lvlN'] * xN
        if q['m_lvl']:
            shift = shift + q['m_lvl'] * _x(lvl, 'lvl') * np.clip(dadj / sd[0], -3, 3)
    shift = shift + xasset_shift(c, q, sd[0], isU)                              # 7
    hs_ = np.asarray(c['hs'], float)
    if cf.get('fx_cap') and c['ttype'] == 'level':
        for j in np.where(isF > 0)[0]:
            cp = cf['fx_cap'] * sd[0][j]
            shift[j] += q['k'] * (float(np.clip(dadj[j], -cp, cp)) - dadj[j]) / sd[0][j]
    if cf.get('CV') and isf.any():
        lvmf = np.clip(0.5 * (lv20 + lv60), -3.0, 3.0)
        shift = shift - cf['CV'] * isf * lvmf * np.sqrt(hs_ / 63.0)
    tailm = 1.0
    if cf.get('DP'):
        dcen, dw, dt = cf['DP']
        xd = np.clip((np.abs(dadj / sd[0]) / np.sqrt(hs_ / 63.0) - dcen) / 0.5, -2.5, 2.5)
        lvm = lvm + dw * xd
        tailm = (1 + dt * xd)[None, :]
    vm = np.exp(lvm)[None, :] if np.any(lvm) else 1.0

    zt = sgn * (q['c'] * cm * np.minimum(az, 1) + (q['ct'] * m * np.clip(az - 1, 0, 1) + q['ct2'] * m * np.maximum(az - 2, 0)) * tailm)
    vr = np.clip(np.array([f['v20'] / f['v300'] if f['v300'] > 0 else 1 for f in feat]) ** q['beta'], q['vlo'], q['vhi'])
    centre = c['anchor'] + q['k'] * np.where(isf > 0, cf.get('KM', 1.0), 1.0) * dadj
    if cf.get('ksl') and isr.any():
        for j in np.where(isr)[0]:
            fj = max(1.0 + cf['ksl'] * np.log(5.0 / TEN[cells[j][0]]) / np.log(2.5), 0.0)
            centre[j] += (fj - 1.0) * q['k'] * dr0[j]
    X = centre + sd * (zt * vr[None, :] * vm + q['b'] * ro + shift[None, :])
    if isr.any():                                                               # 8
        X = X.copy()
        for j in np.where(isr)[0]:
            a0 = c['anchor'][j]; L = max(a0 - 0.2, 0.05) * 2.0
            dl = X[:, j] - a0; neg = dl < 0
            X[neg, j] = a0 - L * np.tanh(-dl[neg] / L)

    m0 = c['anchor'] + c['drift'] + base                                        # 9
    if d == 2:
        if fam == 'F1' and len(set(a_ for a_, h in cells)) == 1:
            Dp = X[:, 1] - X[:, 0]
            mp = np.mean(np.abs(Dp) ** 0.5); mm = np.mean(np.abs(m0[:, 1] - m0[:, 0]) ** 0.5)
            Dn = ((mm / mp) ** 2 if mp > 0 else 1.0) * Dp
            LAM = q['LAM']
            Ab = LAM * X[:, 0] + (1 - LAM) * X[:, 1]
            X = np.column_stack([Ab - (1 - LAM) * Dn, Ab + LAM * Dn])
        else:
            X = m0 + (X[:, :1] - m0[:, :1])
    elif fam == 'F3' and c['ttype'] == 'log_return' and d >= 3:
        ctr = centre[None, :]
        u = (X - ctr) / sd; ub = u.mean(1, keepdims=True); dv = u - ub
        iu = np.triu_indices(d, 1)
        t0 = (np.abs(m0[:, iu[0]] - m0[:, iu[1]]) ** 0.5).mean(0)
        errs = [(((np.abs((ctr + sd * (ub + t * dv))[:, iu[0]] - (ctr + sd * (ub + t * dv))[:, iu[1]]) ** 0.5).mean(0) - t0) ** 2).sum() for t in TAU_GRID]
        X = ctr + sd * (ub + TAU_GRID[int(np.argmin(errs))] ** q['TEXP'] * dv)
    return X
