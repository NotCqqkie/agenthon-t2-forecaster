import json, pathlib, tomllib, zlib
import numpy as np
import pandas as pd

N_DRAWS = 500
WINDOW = 300
PARAMS = dict(k=0.5, c=0.7, ct=1.2)
FAMILY_PARAMS = {'F1': dict(k=0.75, c=0.5, ct=1.0), 'F2': dict(k=0.5, c=1.0, ct=1.3),
                 'F3': dict(k=0.25, c=0.9, ct=1.3), 'F4': dict(k=0.75, c=1.6, ct=1.7)}
VOL_BETA, VOL_CLIP = 0.25, (0.7, 1.4)


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


def vol_ratio(p):
    out = []
    for a, _ in p['cells']:
        s = p['hist'][a].to_numpy()
        st = np.diff(s) if p['ttype'] == 'level' else s
        st = st[np.isfinite(st)]
        v20, v300 = (st[-20:].std(), st[-300:].std()) if len(st) >= 40 else (1.0, 1.0)
        out.append(v20 / v300 if v300 > 0 and np.isfinite(v20 / v300) else 1.0)
    return np.clip(np.array(out) ** VOL_BETA, *VOL_CLIP)


def transform(p, k, c, ct, vol=False, adj=None):
    base = p['base']
    sd = base.std(0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = base / sd
    a = np.abs(z)
    zt = np.sign(z) * (c * np.minimum(a, 1) + ct * np.maximum(a - 1, 0))
    if vol:
        zt = zt * vol_ratio(p)[None, :]
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


def params_for(card, panels_used):
    if any('macro' in n or 'em_' in n or 'transfer' in n for n in panels_used):
        return dict(k=1.0, c=1.0, ct=1.0)
    fam = str(card.get('metadata', {}).get('category', ''))[-2:]
    return dict(FAMILY_PARAMS.get(fam, PARAMS), vol=True)


def forecast(card, spec, panels, asof, unit_id):
    p = m0_parts(card, spec, panels, asof, unit_id)
    used = set()
    for a in p['assets']:
        for name, df in panels:
            if (df['asset'] == a).any():
                used.add(name)
                break
    prm = params_for(card, used)
    X = transform(p, **prm)
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
