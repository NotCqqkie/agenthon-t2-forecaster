import json, pathlib, tomllib, zlib
import numpy as np
import pandas as pd

N_DRAWS = 500
WINDOW = 300
PARAMS = dict(k=0.5, c=0.7, ct=1.2)
FAMILY_PARAMS = {'F1': dict(k=0.625, c=0.5, ct=1.0, ct2=1.15), 'F2': dict(k=0.5, c=1.0, ct=1.3),
                 'F3': dict(k=0.375, c=0.9, ct=1.3), 'F4': dict(k=0.625, c=1.6, ct=1.7, beta=0.5)}
VOL_BETA, VOL_CLIP = 0.25, (0.7, 1.4)
RISK_OFF = {'MKT': -1, 'SMB': -1, 'QMJ': 1, 'AUD': -1, 'NZD': -1, 'EUR': -1, 'GBP': -1, 'CAD': 1, 'NOK': 1,
            'SEK': 1, 'DKK': 1, 'JPY': -1, 'CHF': -1}
F4_SKEW = (0.3, 0.25)
# v8 additions (2026-10-02 swarm, verified on a sealed holdout):
# A: Treasury prior-window reversal of the centre drift
REV_G0 = {'F1': 0.75, 'F2': 1.0, 'F3': 1.0, 'F4': 1.25}
REV_PW, REV_C3, REV_CAP = 0.5, 0.5, 1.5
# C: drift-direction centre shift linear in the as-of vol ratio; F4 inner width by trend strength
REGIME_RULE = {'F1': ('lvm', 0.049, 0.0645, 0.1, 0.0), 'F2': ('lvm', -0.026, 0.178, 0.1, 0.0),
               'F3': ('lvm', 0.094, 0.0, 0.1, 0.0), 'F4': ('lv60', 0.0608, 0.3735, 0.1, 0.5)}
REGIME_NORM = {'lv20': (-0.115, 0.33), 'lv60': (-0.06, 0.25), 'lvm': (-0.09, 0.28), 'lt60': (-0.22, 0.68)}
TREND_WIDTH = {'F4': 0.106}
# B: F1 one-asset two-horizon construction; F3 log-return dispersion match
PAIR_LAM = 0.5
TAU_GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 25))


def find_card(panels_dir):
    for cand in (panels_dir / 'card.toml', panels_dir.parent / 'card.toml'):
        if cand.exists():
            return cand
    raise FileNotFoundError('card.toml')


def read_panels(panels_dir):
    found = sorted(panels_dir.glob('*.parquet'))
    if not found:
        found = sorted(panels_dir.parent.glob('*.parquet'))
    out = []
    for p in found:
        df = pd.read_parquet(p)
        col = 'asset' if 'asset' in df.columns else ('asset_id' if 'asset_id' in df.columns else None)
        if col is None or 'date' not in df.columns or 'value' not in df.columns:
            continue
        df = df[['date', col, 'value']].rename(columns={col: 'asset'})
        df['asset'] = df['asset'].astype(str)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.slice(0, 10), errors='coerce')
        out.append((p.name, df))
    return out


def history(panels, asset, asof):
    for _, df in panels:
        sub = df[df['asset'] == asset]
        if len(sub):
            sub = sub[sub['date'] <= asof].dropna(subset=['date']).sort_values('date', kind='stable')
            sub = sub.drop_duplicates('date', keep='last')
            sub = sub[np.isfinite(pd.to_numeric(sub['value'], errors='coerce'))]
            return pd.Series(sub['value'].astype(float).to_numpy(), index=pd.DatetimeIndex(sub['date']))
    raise KeyError(f'asset {asset} not in panels')


def steps_of(s, ttype):
    s = s.iloc[-WINDOW:]
    if ttype == 'log_return' and np.median(np.abs(s.to_numpy())) >= 0.2:
        raise ValueError('log_return panel does not look like returns')
    gaps = s.index.to_series().diff().dt.days
    hole = max(10 * float(gaps.iloc[1:].median()), 5.0) if len(s) > 1 else 5.0
    st = s.diff() if ttype == 'level' else s.copy()
    st[gaps > hole] = np.nan
    return st.iloc[1:], s, hole


def panel_steps(s_win, hole, h, target_date, asof):
    if len(s_win) < 3:
        return h
    gaps = s_win.index.to_series().diff().dt.days.iloc[1:]
    gaps = gaps[gaps <= hole]
    spacing = float(gaps.mean()) if len(gaps) else 1.0
    if target_date is None:
        if spacing <= 20:
            return h
        target_date = asof + pd.offsets.BDay(h)
    last = s_win.index[-1]
    if spacing > 20:
        steps = 12 * (target_date.year - last.year) + (target_date.month - last.month)
    else:
        steps = int(round((target_date - last).days / spacing))
    if steps <= 0:
        return h
    ratio = max(steps, h) / max(min(steps, h), 1)
    return steps if ratio >= 2 else h


def target_dates(card, spec, asof, cells):
    """Per-cell target date if the card/spec names one; monthly periods map to month start."""
    out = {}
    t = card.get('targets', {})
    st = spec.get('targets', {}) if isinstance(spec, dict) else {}
    st = st if isinstance(st, dict) else {}
    hs = list(t.get('horizons', []))
    for src in (t, st):
        for key in ('target_dates', 'observation_periods'):
            vals = src.get(key)
            if isinstance(vals, list) and len(vals) == len(hs):
                for h, v in zip(hs, vals):
                    try:
                        out.setdefault(int(h), pd.Timestamp(str(v)[:10] if key == 'target_dates' else str(v)[:7] + '-01'))
                    except Exception:
                        pass
    return {c: out.get(c[1]) for c in cells}


def m0_parts(card, spec, panels, asof, unit_id):
    t = card['targets']
    ttype = t.get('target_type', 'level')
    assets = sorted(str(a) for a in t['asset_ids'])
    cells = sorted((str(a), int(h)) for a in t['asset_ids'] for h in t['horizons'])
    hist, steps, wins, holes = {}, {}, {}, {}
    for a in assets:
        hist[a] = history(panels, a, asof)
        steps[a], wins[a], holes[a] = steps_of(hist[a], ttype)
    frame = pd.DataFrame(steps).dropna()
    X = frame[assets].to_numpy()
    mu = X.mean(0)
    Sig = np.atleast_2d(np.cov(X, rowvar=False))
    idx = {a: i for i, a in enumerate(assets)}
    tds = target_dates(card, spec, asof, cells)
    s = {c: panel_steps(wins[c[0]], holes[c[0]], c[1], tds[c], asof) for c in cells}
    anchor = np.array([hist[a].iloc[-1] if ttype == 'level' else 0.0 for a, _ in cells])
    drift = np.array([s[c] * mu[idx[c[0]]] for c in cells])
    cov = np.array([[min(s[c1], s[c2]) * Sig[idx[c1[0]], idx[c2[0]]] for c2 in cells] for c1 in cells])
    cov[np.diag_indices_from(cov)] += 1e-10
    eye = 1e-9 * np.eye(len(cells))
    try:
        L = np.linalg.cholesky(cov + eye)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov + eye)))
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    Z = rng.standard_normal((N_DRAWS, len(cells)))
    return dict(cells=cells, anchor=anchor, drift=drift, base=Z @ L.T, ttype=ttype, steps=s,
                mu=mu, Sig=Sig, assets=assets, n_rows=len(frame), hist=hist)


def vol_ratio(p, beta=None):
    out = []
    for a, _ in p['cells']:
        s = p['hist'][a].to_numpy()
        st = np.diff(s) if p['ttype'] == 'level' else s
        st = st[np.isfinite(st)]
        v20, v300 = (st[-20:].std(), st[-300:].std()) if len(st) >= 40 else (1.0, 1.0)
        out.append(v20 / v300 if v300 > 0 and np.isfinite(v20 / v300) else 1.0)
    return np.clip(np.array(out) ** (VOL_BETA if beta is None else beta), *VOL_CLIP)


def transform(p, k, c, ct, vol=False, adj=None, skew=None, ct2=None, beta=None):
    base = p['base']
    sd = base.std(0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = base / sd
    a = np.abs(z)
    ro = np.array([RISK_OFF.get(str(x).upper(), 0) for x, _ in p['cells']], float)[None, :]
    m = (1 + skew[0] * np.sign(z) * ro) if skew else 1.0
    ct2 = ct if ct2 is None else ct2
    zt = np.sign(z) * (c * np.minimum(a, 1) + ct * m * np.clip(a - 1, 0, 1) + ct2 * m * np.maximum(a - 2, 0))
    if vol:
        zt = zt * vol_ratio(p, beta)[None, :]
    if skew:
        zt = zt + skew[1] * ro
    center = p['anchor'] + k * p['drift']
    if adj is not None:
        shift, width = adj
        zt = zt * width[None, :]
        center = center + shift
    X = center + sd * zt
    if X.shape[1] == 2:
        m0 = p['anchor'] + p['drift'] + base
        X = m0 + (X[:, :1] - m0[:, :1])
    return X


def cell_feats(p):
    out = []
    for a, _ in p['cells']:
        x = p['hist'][a].to_numpy()
        st = np.diff(x) if p['ttype'] == 'level' else x
        if len(st) < 61:
            out.append(None)
            continue
        out.append(dict(v20=st[-20:].std(), v60=st[-60:].std(), v300=st[-300:].std(),
                        r60=(x[-1] - x[-61]) if p['ttype'] == 'level' else st[-60:].sum()))
    return out


def reversal_drift(p, fam, k, sd):
    d = np.array(p['drift'], float).copy()
    if p['ttype'] != 'level' or fam not in REV_G0:
        return k * d
    for j, (a, h) in enumerate(p['cells']):
        if not str(a).upper().startswith('UST'):
            continue
        s = p['hist'][a].iloc[-960:]
        gaps = s.index.to_series().diff().dt.days
        hole = max(10 * float(gaps.iloc[1:].median()), 5.0) if len(s) > 2 else 5.0
        st = s.diff()
        st[gaps > hole] = np.nan
        st = st.iloc[1:].dropna().to_numpy()
        if len(st) < 900:
            continue
        corr = REV_G0[fam] * (21.0 / h) ** REV_PW * h * (st[-600:-300].mean() + REV_C3 * st[-900:-600].mean())
        if np.isfinite(corr):
            d[j] -= float(np.clip(corr, -REV_CAP * sd[j], REV_CAP * sd[j]))
    return k * d


def transform_v8(p, fam, k, c, ct, ct2=None, beta=None, skew=None):
    base = p['base']
    sd = base.std(0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = base / sd
    az = np.abs(z)
    ro = np.array([RISK_OFF.get(str(x).upper(), 0) for x, _ in p['cells']], float)[None, :]
    m = (1 + skew[0] * np.sign(z) * ro) if skew else 1.0
    ct2 = ct if ct2 is None else ct2
    f = cell_feats(p)
    ok = all(x is not None and x['v300'] > 0 for x in f)
    cm = 1.0
    if ok and fam in TREND_WIDTH:
        v3 = np.array([x['v300'] for x in f]); r60 = np.array([x['r60'] for x in f])
        lt = np.log(np.abs(r60) / (v3 * np.sqrt(60)) + 0.15)
        cm = np.exp(TREND_WIDTH[fam] * np.clip((lt - REGIME_NORM['lt60'][0]) / REGIME_NORM['lt60'][1], -2.5, 2.5))[None, :]
    zt = np.sign(z) * (c * cm * np.minimum(az, 1) + ct * m * np.clip(az - 1, 0, 1) + ct2 * m * np.maximum(az - 2, 0))
    zt = zt * vol_ratio(p, beta)[None, :]
    if skew:
        zt = zt + skew[1] * ro
    if ok and fam in REGIME_RULE:
        fe, sa, sb, eps, gam = REGIME_RULE[fam]
        v3 = np.array([x['v300'] for x in f])
        lv20 = np.log(np.maximum([x['v20'] for x in f], 1e-300) / v3)
        lv60 = np.log(np.maximum([x['v60'] for x in f], 1e-300) / v3)
        lv = {'lv20': lv20, 'lv60': lv60, 'lvm': 0.5 * (lv20 + lv60)}[fe]
        x = np.clip((lv - REGIME_NORM[fe][0]) / REGIME_NORM[fe][1], -2.5, 2.5)
        dr = np.asarray(p['drift'], float)
        shift = (sa + sb * x) * dr / (np.abs(dr) + eps * sd[0] + 1e-300)
        if gam:
            shift = shift * (np.array([h for _, h in p['cells']], float) / 63.0) ** gam
        zt = zt + shift[None, :]
    centre = p['anchor'] + reversal_drift(p, fam, k, sd[0])
    X = centre + sd * zt
    d = X.shape[1]
    m0 = p['anchor'] + p['drift'] + base
    if d == 2:
        if fam == 'F1' and p['cells'][0][0] == p['cells'][1][0]:
            D = X[:, 1] - X[:, 0]
            mp, mm = np.mean(np.abs(D) ** 0.5), np.mean(np.abs(m0[:, 1] - m0[:, 0]) ** 0.5)
            D = ((mm / mp) ** 2 if mp > 0 else 1.0) * D
            A = PAIR_LAM * X[:, 0] + (1 - PAIR_LAM) * X[:, 1]
            X = np.column_stack([A - (1 - PAIR_LAM) * D, A + PAIR_LAM * D])
        else:
            X = m0 + (X[:, :1] - m0[:, :1])
    elif fam == 'F3' and p['ttype'] == 'log_return' and d >= 3:
        ctr = centre[None, :]
        u = (X - ctr) / sd
        ub = u.mean(1, keepdims=True)
        dv = u - ub
        iu = np.triu_indices(d, 1)
        t0 = (np.abs(m0[:, iu[0]] - m0[:, iu[1]]) ** 0.5).mean(0)
        errs = []
        for t in TAU_GRID:
            Y = ctr + sd * (ub + t * dv)
            errs.append((((np.abs(Y[:, iu[0]] - Y[:, iu[1]]) ** 0.5).mean(0) - t0) ** 2).sum())
        X = ctr + sd * (ub + TAU_GRID[int(np.argmin(errs))] ** 0.5 * dv)
    return X


def params_for(card, panels_used):
    if any('macro' in n or 'em_' in n or 'transfer' in n for n in panels_used):
        return dict(k=1.0, c=1.0, ct=1.0)
    fam = str(card.get('metadata', {}).get('category', ''))[-2:]
    if fam in FAMILY_PARAMS:
        return dict(FAMILY_PARAMS[fam], fam=fam, skew=F4_SKEW if fam == 'F4' else None)
    return dict(PARAMS, vol=True)


def forecast(card, spec, panels, asof, unit_id):
    p = m0_parts(card, spec, panels, asof, unit_id)
    used = set()
    for a in p['assets']:
        for name, df in panels:
            if (df['asset'] == a).any():
                used.add(name)
                break
    prm = params_for(card, used)
    X = transform_v8(p, **prm) if 'fam' in prm else transform(p, **prm)
    if not np.isfinite(X).all():
        raise ValueError('non-finite draws')
    return p, X, prm


def fallback(card, panels, asof):
    """Last resort: independent Gaussian walk from each asset's last value."""
    t = card['targets']
    ttype = t.get('target_type', 'level')
    cells = sorted((str(a), int(h)) for a in t['asset_ids'] for h in t['horizons'])
    rng = np.random.default_rng(0)
    X = np.empty((N_DRAWS, len(cells)))
    for j, (a, h) in enumerate(cells):
        try:
            s = history(panels, a, asof).dropna()
            v = s.diff().dropna() if ttype == 'level' else s
            sd = float(v.iloc[-WINDOW:].std()) if len(v) > 2 else 0.0
            sd = sd if np.isfinite(sd) and sd > 0 else max(abs(float(s.iloc[-1])) * 0.01, 1e-4)
            last = float(s.iloc[-1]) if ttype == 'level' else 0.0
        except Exception:
            sd, last = 1.0, 0.0
        X[:, j] = last + sd * np.sqrt(h) * rng.standard_normal(N_DRAWS)
    return cells, X
