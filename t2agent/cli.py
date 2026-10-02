import argparse, json, pathlib, sys, tomllib, traceback
import numpy as np
import pandas as pd
from . import core


def rationale(unit_id, asof, p, prm, n_docs, note):
    lines = [f'# Forecast rationale — {unit_id}', '',
             f'As of **{asof}**. {core.N_DRAWS} joint draws.', '',
             '## Method', '',
             '1. Statistical backbone: joint Gaussian random walk on the trailing 300 panel steps '
             '(level changes, or per-step returns for log-return targets), steps spanning data holes dropped, '
             'assets aligned by date, cross-horizon path covariance min(s, s\') * Sigma.',
             f'2. Drift: the trailing-window mean step is shrunk by k = {prm["k"]} (trend extrapolation is noisy).',
             f'3. Shape: standardized draws are reshaped piecewise-linearly, slope {prm["c"]} inside one sd and '
             f'{prm["ct"]} beyond it (h-day financial changes are leptokurtic: peaked body, fat tails)'
             + ('; width scaled by (20-day vol / 300-day vol)^0.25.' if prm.get('vol') else '.'),
             '4. Parameters were chosen on pre-as-of historical backtests only; no unit-specific values are stored.',
             '', '## Ledger', '', '| asset | horizon | anchor | drift used | panel steps |', '|---|---|---|---|---|']
    if p is not None:
        for j, (a, h) in enumerate(p['cells']):
            lines.append(f'| {a} | {h} | {p["anchor"][j]:.6g} | {prm["k"] * p["drift"][j]:.6g} | {p["steps"][(a, h)]} |')
    lines += ['', '## Text', '', f'{n_docs} document(s) available; this version does not adjust on text.']
    if note:
        lines += ['', '## Notes', '', note]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(prog='forecast')
    ap.add_argument('verb', nargs='?')
    ap.add_argument('--panels', type=pathlib.Path, required=True)
    ap.add_argument('--text', type=pathlib.Path, default=None)
    ap.add_argument('--asof', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--card', type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    card_path = a.card or core.find_card(a.panels)
    card = tomllib.loads(card_path.read_text())
    spec = None
    sp = card_path.parent / 'forecast_spec.json'
    if sp.exists():
        try:
            spec = json.loads(sp.read_text())
        except Exception:
            spec = None
    unit_id = str(card.get('task', {}).get('id') or card_path.parent.name)
    asof = pd.Timestamp(a.asof)
    panels = core.read_panels(a.panels)
    t = card['targets']
    note, p, prm = '', None, dict(k=1.0, c=1.0, ct=1.0)
    try:
        p, X, prm = core.forecast(card, spec, panels, asof, unit_id)
        cells = p['cells']
    except Exception:
        note = 'Primary method failed; independent Gaussian walk fallback used.\n\n```\n' + traceback.format_exc()[-1500:] + '\n```'
        cells, X = core.fallback(card, panels, asof)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n, d = X.shape
    df = pd.DataFrame({
        'draw': np.repeat(np.arange(n, dtype=np.int32), d),
        'asset': pd.array([c[0] for c in cells] * n, dtype='string'),
        'horizon': np.tile(np.array([c[1] for c in cells], dtype=np.int32), n),
        'value': X.reshape(-1).astype(np.float64),
    })
    df.to_parquet(a.out, index=False)
    meta = {'unit_id': unit_id, 'asof': a.asof, 'representation': 'samples',
            'asset_ids': [str(x) for x in t['asset_ids']], 'horizons': [int(h) for h in t['horizons']],
            'n_draws': int(n), 'target': t.get('target_type', 'level'),
            'rationale': {'file': 'forecast_rationale.md', 'method': 'M0-backbone RW, shrunk drift, leptokurtic reshape'}}
    if t.get('value_unit'):
        meta['units'] = t['value_unit']
    (a.out.parent / 'forecast_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    n_docs = len(list(a.text.glob('*'))) if a.text and a.text.is_dir() else 0
    (a.out.parent / 'forecast_rationale.md').write_text(rationale(unit_id, a.asof, p, prm, n_docs, note))
    print(f'wrote {a.out} ({n} draws x {d} cells){" [fallback]" if note else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
