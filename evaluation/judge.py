"""Tier 3 AI judge: rubric scores, position-swapped pairwise verdicts, calibration.

Backends: `claude` (Anthropic API), `openjev` (a local OpenJev Multimodal
server, github.com/jev-skills/openjev-multimodal; set CHAMELEON_OPENJEV_URL,
and OPENJEV_API_KEY if the server requires one), `ollama` (a local Ollama
server; set CHAMELEON_OLLAMA_URL / CHAMELEON_OLLAMA_MODEL) and `local` (the
optional llama.cpp VLM in engine/vlm.py). OpenJev reads each score from one
token's probability distribution, so it returns numbers without reasons text. All are development evaluation only, never
the offline runtime. The judge scores rendered PNGs, so the model must accept
images: a vision model such as qwen3-vl:4b, not text-only qwen3:4b.

  python -m evaluation.judge score   output/evaluation/RUN [--backend claude] [--limit N]
  python -m evaluation.judge compare output/evaluation/NEW output/evaluation/OLD
  python -m evaluation.judge calibrate output/workspace/labels.jsonl
"""
import argparse, base64, io, json, os, statistics, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from engine.canonical import ASSETS_DIR
from engine.constitution import CRITERIA, CRITERIA_TEXT, PAIR_PROMPT, PAIR_SCHEMA, SCORE_PROMPT, SCORE_SCHEMA, SCORE_STATE
from engine.formats import aspect_bucket, name_for

MODEL = os.environ.get('CHAMELEON_JUDGE_MODEL', 'claude-opus-5-5')
OLLAMA_URL = os.environ.get('CHAMELEON_OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('CHAMELEON_OLLAMA_MODEL', 'qwen3-vl:4b')
OPENJEV_URL = os.environ.get('CHAMELEON_OPENJEV_URL', 'http://127.0.0.1:8000')
OPENJEV_MODEL = os.environ.get('CHAMELEON_OPENJEV_MODEL', 'jev-latest')
_CLIENT = None


def render(design, size=768):
    from renderer.converter.convert import render_design_to_png
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'design.png'
        render_design_to_png(design, ASSETS_DIR, path, max_size=size)
        with Image.open(path) as im:
            return im.convert('RGB')


def _png(image):
    buf = io.BytesIO(); image.save(buf, format='PNG'); return buf.getvalue()


def _ask_claude(prompt, images, schema):
    global _CLIENT
    import anthropic
    _CLIENT = _CLIENT or anthropic.Anthropic()
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64.standard_b64encode(_png(im)).decode()}} for im in images]
    response = _CLIENT.beta.messages.create(
        model=MODEL, max_tokens=8000, betas=['server-side-fallback-2026-07-01'], fallbacks='default',
        output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': schema}},
        messages=[{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}])
    if response.stop_reason == 'refusal':
        return None
    return json.loads(next(b.text for b in response.content if b.type == 'text'))


def ollama_check():
    """Fail early unless the server is up and the model can read images."""
    import httpx
    try:
        info = httpx.post(OLLAMA_URL + '/api/show', json={'model': OLLAMA_MODEL}, timeout=10)
    except httpx.HTTPError as exc:
        raise SystemExit(f'Ollama not reachable at {OLLAMA_URL}: {exc}')
    if info.status_code != 200:
        raise SystemExit(f'Ollama model {OLLAMA_MODEL!r} not found (pull it, or set CHAMELEON_OLLAMA_MODEL)')
    caps = info.json().get('capabilities', [])
    if 'vision' not in caps:
        raise SystemExit(f'{OLLAMA_MODEL} has no vision capability {caps}; the judge needs images (try qwen3-vl:4b)')


def _ask_ollama(prompt, images, schema):
    """qwen3-vl thinks even with think=false, so the context must hold two images, the rubric and its reasoning."""
    import httpx, sys
    small = []
    for im in images:
        im = im.copy(); im.thumbnail((512, 512)); small.append(base64.standard_b64encode(_png(im)).decode())
    body = {'model': OLLAMA_MODEL, 'stream': False, 'format': schema,
            'options': {'temperature': 0.2, 'num_ctx': int(os.environ.get('CHAMELEON_OLLAMA_CTX', 16384)), 'num_predict': 8000},
            'messages': [{'role': 'user', 'content': prompt + '\nRespond only with JSON matching the schema.', 'images': small}]}
    try:
        reply = httpx.post(OLLAMA_URL + '/api/chat', json=body, timeout=1200).json()
        answer = json.loads(reply['message']['content'])
        import jsonschema
        jsonschema.validate(answer, schema); return answer
    except Exception as exc:  # unreachable server, truncated or off-schema reply -> no verdict
        print('ollama: no verdict (' + type(exc).__name__ + ')', file=sys.stderr); return None


LEVELS = ['broken', 'broken', 'serious problems', 'obvious shortcomings', 'several noticeable issues', 'mostly fine, a few issues',
          'acceptable with minor issues', 'good', 'very good', 'flawless']
DEFECT_CHECKS = {
    'clipped': 'Some text or element in the RESIZED version is cut off, clipped or runs off the canvas',
    'tiny': 'Some text in the RESIZED version is too small to read comfortably',
    'overlap': 'Elements in the RESIZED version overlap or collide where the ORIGINAL kept them apart',
    'distorted': 'An image or logo in the RESIZED version is stretched, squashed, covered or badly cropped',
    'focal': 'The headline or hero is no longer the clear focal point in the RESIZED version',
}


def openjev_check():
    """Fail early unless the OpenJev server is up and accepts images."""
    import httpx
    try:
        info = httpx.get(OPENJEV_URL + '/health', timeout=10)
    except httpx.HTTPError as exc:
        raise SystemExit(f'OpenJev not reachable at {OPENJEV_URL}: {exc} (start it with `uv run openjev serve`)')
    if info.status_code != 200:
        raise SystemExit(f'OpenJev at {OPENJEV_URL} is not ready (HTTP {info.status_code})')
    if not info.json().get('multimodal'):
        raise SystemExit(f'OpenJev model {info.json().get("model")!r} cannot read images; the judge needs a multimodal profile')


def _ask_openjev(state, images, questions):
    """One /v1/systemone call: every question is read from a single output token, no generated text."""
    import sys
    from engine.evaluators.openjev import request_readout
    try:
        return request_readout(state,images,questions,url=OPENJEV_URL,model=OPENJEV_MODEL,timeout=180)
    except Exception as exc:  # unreachable server, queue full or bad readout -> no verdict
        print('openjev: no verdict (' + type(exc).__name__ + ')', file=sys.stderr); return None


def _level(answer):
    """Score answers are an expected index over LEVELS (lowest first); map onto 1-10."""
    keys = [int(k) for k in answer.get('legend', {}) if str(k).lstrip('-').isdigit()]
    return round(max(1.0, min(10.0, answer['score'] - (min(keys) if keys else 0) + 1)), 1)


def _why(answer):
    if not answer.get('probabilities'):
        return ''
    top, p = max(answer['probabilities'].items(), key=lambda kv: kv[1])
    return f"most likely {answer.get('legend', {}).get(top, top)} ({p:.0%}); confidence {answer.get('confidence', 0):.2f}"


def _score_openjev(source, target, images):
    levels = [f'{n}/10: {t}' for n, t in enumerate(LEVELS, 1)]
    questions = {k: {'type': 'score', 'instructions': f'Rate the RESIZED version on {k}: {" ".join(v.split())}.', 'criteria': levels}
                 for k, v in CRITERIA_TEXT.items()}
    questions['overall'] = {'type': 'score', 'instructions': 'Overall, how good is the RESIZED version as a professional adaptation of the ORIGINAL to this format?', 'criteria': levels}
    questions.update({'defect_' + k: {'type': 'noul', 'instructions': f'Is this true? {v}.'} for k, v in DEFECT_CHECKS.items()})
    answers = _ask_openjev(SCORE_STATE.format(**_frame(source, target)), images, questions)
    if not answers:
        return None
    found = sorted(((answers['defect_' + k]['noul'], v) for k, v in DEFECT_CHECKS.items() if answers['defect_' + k]['noul'] >= .5), reverse=True)
    return {'scores': {k: _level(answers[k]) for k in CRITERIA}, 'overall': _level(answers['overall']),
            'reasons': {k: _why(answers[k]) for k in CRITERIA}, 'defects': [f'{v} (p={p:.2f})' for p, v in found[:3]],
            'samples': 1, 'backend': 'openjev'}


def _pair_openjev(prompt, images):
    answers = _ask_openjev(prompt, images, {'verdict': {'type': 'choice', 'instructions': 'Which resize would a professional designer ship for this format?',
                                                       'criteria': {'A': 'Resize A (Image 2) is better', 'B': 'Resize B (Image 3) is better',
                                                                    'tie': 'They are equivalent', 'unsure': 'Cannot tell'}}})
    if not answers:
        return None
    v = answers['verdict']
    return {'verdict': v['choice'], 'reason': ', '.join(f'{k} {p:.0%}' for k, p in v['probabilities'].items())}


def _strip(images):
    """Local VLM takes one image: label and tile them left to right."""
    h = max(im.height for im in images); out = Image.new('RGB', (sum(im.width for im in images) + 16 * len(images), h + 28), 'white')
    draw = ImageDraw.Draw(out); x = 0
    for n, im in enumerate(images, 1):
        out.paste(im, (x, 28)); draw.text((x + 4, 6), f'Image {n}', fill='black'); x += im.width + 16
    return out


def ask(prompt, images, schema, backend):
    if backend == 'claude':
        return _ask_claude(prompt, images, schema)
    if backend == 'ollama':
        return _ask_ollama(prompt, images, schema)
    if backend == 'openjev':
        if schema is not PAIR_SCHEMA:
            raise ValueError('openjev rubric scores go through score(), not ask()')
        return _pair_openjev(prompt, images)
    from engine.vlm import describe
    return describe(_strip(images), prompt + '\nThe images are tiled left to right and labelled.', schema)


def _frame(source, target):
    return {'src_w': source['canvas']['width'], 'src_h': source['canvas']['height'], 'tgt_w': target['width'], 'tgt_h': target['height'],
            'format_name': name_for(target['width'], target['height'])}


def score(source, design, target, backend='claude', samples=3, images=None):
    """Median of `samples` rubric scorings (scores 1-10 per criterion)."""
    images = images or [render(source), render(design)]
    if backend == 'openjev':  # deterministic probability readout: extra samples would repeat the same answer
        return _score_openjev(source, target, images)
    runs = [r for r in (ask(SCORE_PROMPT.format(**_frame(source, target)), images, SCORE_SCHEMA, backend) for _ in range(samples)) if r]
    if not runs:
        return None
    clamp = lambda v: max(1, min(10, int(v)))
    return {'scores': {k: statistics.median(clamp(r['scores'][k]) for r in runs) for k in CRITERIA},
            'overall': statistics.median(clamp(r['overall']) for r in runs), 'defects': runs[0]['defects'],
            'reasons': runs[0]['reasons'], 'samples': len(runs), 'backend': backend}


def pairwise(source, a, b, target, backend='claude', samples=1, images=None):
    """Run both orders. A side wins only if it wins in both; otherwise tie."""
    src, ia, ib = images or (render(source), render(a), render(b))
    prompt = PAIR_PROMPT.format(**_frame(source, target)); votes = []
    for _ in range(samples):
        first = ask(prompt, [src, ia, ib], PAIR_SCHEMA, backend); second = ask(prompt, [src, ib, ia], PAIR_SCHEMA, backend)
        if not first or not second:
            continue
        v1, v2 = first['verdict'], {'A': 'B', 'B': 'A'}.get(second['verdict'], second['verdict'])
        votes.append(v1 if v1 == v2 and v1 in ('A', 'B') else 'tie')
    if not votes:
        return None
    verdict = Counter(votes).most_common(1)[0][0]
    return {'verdict': {'A': 'a', 'B': 'b'}.get(verdict, 'tie'), 'votes': votes, 'backend': backend}


def _cases(run):
    for path in sorted(Path(run).glob('*__*.json')):
        record = json.loads(path.read_text())
        yield path.stem, record


def score_run(run, backend, samples, limit=None):
    out = {}
    for n, (key, r) in enumerate(_cases(run)):
        if limit and n >= limit:
            break
        result = score(r['source'], r['result']['design'], r['target_canvas'], backend, samples)
        out[key] = {'bucket': aspect_bucket(r['source']['canvas'], r['target_canvas']), **(result or {'error': 'no verdict'})}
        print(key, (result or {}).get('overall'), flush=True)
    by = defaultdict(list)
    for v in out.values():
        if 'overall' in v:
            by[v['bucket']].append(v)
    summary = {'backend': backend, 'model': {'claude': MODEL, 'ollama': OLLAMA_MODEL, 'openjev': OPENJEV_MODEL}.get(backend, 'local'), 'cases': len(out),
               'median': {c: statistics.median(v['scores'][c] for v in out.values() if 'scores' in v) for c in CRITERIA} if any('scores' in v for v in out.values()) else {},
               'by_bucket': {b: statistics.median(v['overall'] for v in vs) for b, vs in by.items()}}
    (Path(run) / 'judge.json').write_text(json.dumps({'summary': summary, 'cases': out}, indent=2))
    return summary


def compare_runs(new, old, backend, samples=1, limit=None):
    old_cases = dict(_cases(old)); tally = defaultdict(Counter); rows = {}
    for n, (key, r) in enumerate(_cases(new)):
        if key not in old_cases or (limit and n >= limit):
            continue
        a, b = r['result']['design'], old_cases[key]['result']['design']
        if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
            verdict = {'verdict': 'tie', 'votes': ['identical']}
        else:
            verdict = pairwise(r['source'], a, b, r['target_canvas'], backend, samples) or {'verdict': 'tie', 'votes': ['no verdict']}
        bucket = aspect_bucket(r['source']['canvas'], r['target_canvas'])
        result = {'a': 'win', 'b': 'loss'}.get(verdict['verdict'], 'tie')
        tally[bucket][result] += 1; tally['all'][result] += 1; rows[key] = {**verdict, 'bucket': bucket, 'result': result}
        print(key, result, flush=True)
    summary = {'new': str(new), 'old': str(old), 'backend': backend, 'tally': {k: dict(v) for k, v in tally.items()},
               'no_worse_everywhere': all(v['win'] >= v['loss'] for v in tally.values())}
    (Path(new) / 'comparison.json').write_text(json.dumps({'summary': summary, 'cases': rows}, indent=2))
    return summary


def cohen_kappa(x, y):
    labels = sorted(set(x) | set(y)); n = len(x)
    if not n:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def calibrate(labels, backend, limit=None):
    """Agreement of the judge with human pairwise labels from the review workspace."""
    rows = [json.loads(l) for l in Path(labels).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get('kind') == 'human' and r['decision'] in ('a', 'b', 'tie')][:limit]
    from service.workspace import record
    human, judged = [], []
    for r in rows:
        run_id, index = r['pair'].split('-')[0], int(r['pair'].split('-')[1])
        task = record(run_id)['request']; target = record(run_id)['response']['results'][index]['target']
        verdict = pairwise(task['source'], r['a']['design'], r['b']['design'], target, backend)
        if verdict:
            human.append(r['decision']); judged.append(verdict['verdict'])
    decisive = [(h, j) for h, j in zip(human, judged) if h != 'tie']
    report = {'pairs': len(human), 'kappa': cohen_kappa(human, judged),
              'decisive_agreement': sum(h == j for h, j in decisive) / len(decisive) if decisive else None,
              'target': 'kappa >= 0.6 (0.8 strong); rerun whenever judge model, prompt or renderer changes'}
    print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('score', 'compare', 'calibrate'):
        p = sub.add_parser(name); p.add_argument('path', type=Path)
        if name == 'compare':
            p.add_argument('baseline', type=Path)
        p.add_argument('--backend', choices=['claude', 'openjev', 'ollama', 'local'], default='claude'); p.add_argument('--samples', type=int, default=3 if name == 'score' else 1)
        p.add_argument('--limit', type=int)
    a = ap.parse_args()
    if a.backend == 'ollama':
        ollama_check()
    if a.backend == 'openjev':
        openjev_check()
    if a.cmd == 'score':
        print(json.dumps(score_run(a.path, a.backend, a.samples, a.limit), indent=2))
    elif a.cmd == 'compare':
        print(json.dumps(compare_runs(a.path, a.baseline, a.backend, a.samples, a.limit), indent=2))
    else:
        calibrate(a.path, a.backend, a.limit)
