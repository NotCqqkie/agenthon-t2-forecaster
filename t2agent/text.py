import base64, http.client, json, os, pathlib, re, time, urllib.request
from urllib.parse import unquote, urlsplit

MAX_DOC_CHARS = 6000
MAX_TOTAL_CHARS = 60000
TIMEOUT = 240
OUT_TOKENS = 1500


def read_corpus(text_dir, asof):
    if text_dir is None:
        return []
    idx = pathlib.Path(text_dir) / 'corpus_index.json'
    if not idx.is_file():
        return []
    docs = []
    for d in json.loads(idx.read_text(encoding='utf-8')).get('documents', []):
        ts = str(d.get('timestamp', ''))[:10]
        p = pathlib.Path(text_dir) / str(d.get('file', ''))
        if not ts or ts > asof or not p.is_file():
            continue
        docs.append(dict(doc_id=d.get('doc_id'), ts=ts, kind=d.get('doc_type'),
                         text=p.read_text(encoding='utf-8', errors='replace')))
    docs.sort(key=lambda d: d['ts'], reverse=True)
    out, total = [], 0
    for d in docs:
        t = re.sub(r'\s+', ' ', d['text']).strip()[:MAX_DOC_CHARS]
        if total + len(t) > MAX_TOTAL_CHARS:
            break
        out.append(dict(d, text=t))
        total += len(t)
    return out


def card_context(card, card_dir):
    parts = []
    tx = card.get('text', {})
    if tx.get('notes'):
        parts.append(str(tx['notes']))
    md = pathlib.Path(card_dir) / 'forecast_card.md'
    if md.is_file():
        m = re.search(r'\*\*Text corpus role\.\*\*(.*?)(\n\n|$)', md.read_text(errors='replace'), re.S)
        if m and m.group(1).strip() not in parts:
            parts.append(m.group(1).strip())
    return ' '.join(parts)[:2000]


def build_prompt(asof, ttype, assets, horizons, stats, docs, context):
    L = ['You are a careful macro/markets analyst adjusting a statistical forecast using dated documents.',
         f'As-of date: {asof}. Treat nothing after this date as known. Do not use hindsight.',
         f'Target: {"cumulative log return" if ttype == "log_return" else "level"} at horizons {horizons} business days ahead.',
         '', 'Statistical baseline per asset (anchor, typical move = one standard deviation at the longest horizon, '
         'recent 60-day change):']
    for a in assets:
        s = stats[a]
        L.append(f'  {a}: anchor {s["anchor"]:.6g}, one-sd move {s["sd"]:.4g}, last-60d change {s["r60"]:.4g}')
    if context:
        L += ['', f'Task context: {context}']
    L += ['', f'Documents ({len(docs)}), newest first:']
    for d in docs:
        L += [f'--- {d["doc_id"]} ({d["ts"]}, {d["kind"]}) ---', d['text'], '']
    L += ['For EACH asset give:',
          '  drift_sd: expected move over the longest horizon in units of the one-sd move, between -1 and 1. '
          'Positive = higher value. Use 0 unless the documents give a clear directional reason.',
          '  width: multiplier on the baseline uncertainty, between 0.6 and 1.8. Above 1 if the documents point to a '
          'scheduled decision, vote, policy shift, crisis or stress inside the horizon; below 1 only if they point to '
          'an unusually calm, anchored regime.',
          'Reply with JSON only:',
          '{"assets": {"<asset>": {"drift_sd": <number>, "width": <number>, "why": "<short, cite doc_id>"}}}']
    return '\n'.join(L)


def _post_house(endpoint, token, body):
    target, proxy = urlsplit(endpoint), urlsplit(os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or '')
    url = endpoint.rstrip('/') + '/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token}
    if proxy.hostname and target.scheme == 'http':
        if proxy.username:
            cred = unquote(proxy.username) + ':' + unquote(proxy.password or '')
            headers['Proxy-Authorization'] = 'Basic ' + base64.b64encode(cred.encode()).decode()
        conn = http.client.HTTPConnection(proxy.hostname, proxy.port or 80, timeout=TIMEOUT)
        try:
            conn.request('POST', url, body, headers)
            r = conn.getresponse()
            raw = r.read()
            if r.status != 200:
                raise ValueError(f'HTTP {r.status}')
            return json.loads(raw)
        finally:
            conn.close()
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def call(prompt):
    """Return (parsed_json or None, reason). One request; one retry only on transport failure."""
    endpoint, model, token = (os.environ.get(k, '').strip() for k in ('MODEL_ENDPOINT', 'MODEL_NAME', 'MODEL_TOKEN'))
    if not (endpoint and model and token):
        return None, 'no House model configured'
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                       'max_tokens': OUT_TOKENS, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    last = ''
    for attempt in range(2):
        try:
            payload = _post_house(endpoint, token, body)
            break
        except Exception as e:
            last = f'request failed ({type(e).__name__})'
            if attempt == 0:
                time.sleep(5)
    else:
        return None, last
    try:
        content = payload['choices'][0]['message']['content'] or ''
        i, j = content.find('{'), content.rfind('}')
        return json.loads(content[i:j + 1]), ''
    except Exception as e:
        return None, f'unparseable reply ({type(e).__name__})'


def parse_adjust(reply, assets):
    out = {}
    for a in assets:
        r = (reply or {}).get('assets', {}).get(a, {}) if isinstance(reply, dict) else {}
        try:
            d = float(r.get('drift_sd', 0.0))
            w = float(r.get('width', 1.0))
        except Exception:
            d, w = 0.0, 1.0
        d = d if d == d and abs(d) < 1e6 else 0.0
        w = w if w == w and 0 < w < 1e6 else 1.0
        out[a] = (max(-1.0, min(1.0, d)), max(0.6, min(1.8, w)), str(r.get('why', ''))[:300])
    return out
