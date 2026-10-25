"""Tiered design guardrails (docs/chameleon-design-guardrails-and-eval-rubric.md).

Tier 0 disqualifiers and Tier 1 structure are deterministic gates. Tier 2
scores measurable design principles on the target-size layout. Candidate
ranking, the benchmark, the release gate and training rewards all call
`evaluate()`, so what we train toward is exactly what we measure.
"""
from __future__ import annotations

import hashlib, json, math
from functools import lru_cache
from statistics import median, pstdev
from typing import Any

from .canonical import ROOT, MUTABLE_FIELDS, Box, _asset_info, element_box, parse_alpha, rotated_extent, visual_box

SCHEMA = ROOT / 'chameleon_design_schema_v0.9.json'
WEIGHTS = {'legibility': .22, 'contrast': .10, 'hierarchy': .16, 'grouping': .12, 'overlap': .10,
           'alignment': .08, 'space': .12, 'imagery': .05, 'fidelity': .05}
TEXT_ROLES = ('headline', 'subhead', 'body', 'cta', 'legal', 'badge')


def _issue(code, message, ids=(), severity='defect'):
    return {'id': code, 'severity': severity, 'message': message, 'ids': sorted({str(i) for i in ids})}


def _order(design):
    """Paint order: z_index, then document order."""
    return {e['id']: (e.get('z_index', 0), n) for n, e in enumerate(design['elements'])}


def _visible_fraction(e, cw, ch):
    b = rotated_extent(e)
    return b.clip(cw, ch).area / b.area if b.area > 0 else 0.0


def _opaque(e):
    if float(e.get('opacity', 1)) < 1:
        return False
    if e['type'] == 'shape':
        return parse_alpha(e.get('fill_color')) >= 1 and not str(e.get('fill_color', '')).startswith('transparent')
    return e['type'] == 'image' and e.get('fit') in ('cover', 'fill') and _asset_info(e.get('asset_id') or '') is not None


def _contains(outer: Box, inner: Box, frac=1.0):
    return inner.area > 0 and outer.intersection_area(inner) >= frac * inner.area - 1e-6


def _covered(design):
    """IDs fully covered by an opaque element painted above them."""
    order = _order(design); els = design['elements']; out = set()
    for e in els:
        box = rotated_extent(e)
        if any(c is not e and order[c['id']] > order[e['id']] and _opaque(c) and _contains(rotated_extent(c), box) for c in els):
            out.add(e['id'])
    return out


# ---------------------------------------------------------------- Tier 0 --

def tier0(source, design):
    src = {e['id']: e for e in source['elements']}; out_list = design.get('elements', [])
    by = {}
    for e in out_list:
        by.setdefault(e.get('id'), e)
    fails = []
    images = [i for i, s in src.items() if s['type'] == 'image' and (by.get(i, {}).get('type') != 'image' or by[i].get('asset_id') != s.get('asset_id'))]
    foreign = [e.get('id') for e in out_list if e.get('asset_id') and e.get('id') not in src]
    if images or foreign:
        fails.append(_issue('D1', 'Image asset references changed or added', images + foreign))
    texts = [i for i, s in src.items() if s['type'] == 'text']
    copy = [i for i in texts if i in by and (by[i].get('content') != src[i].get('content') or by[i].get('style_runs') != src[i].get('style_runs'))]
    if copy:
        fails.append(_issue('D2', 'Text content or rich-text runs changed', copy))
    fonts = [i for i in texts if i in by and by[i].get('font_family') != src[i].get('font_family')]
    if fonts:
        fails.append(_issue('D3', 'Font family changed', fonts))
    ids = [e.get('id') for e in out_list]
    if sorted(map(str, ids)) != sorted(src) or len(ids) != len(src):
        missing = set(src) - set(ids); dupes = {i for i in ids if ids.count(i) > 1}; extra = set(ids) - set(src)
        fails.append(_issue('D4', 'Elements added, removed or duplicated', missing | dupes | extra))
    cw, ch = design['canvas']['width'], design['canvas']['height']; sw, sh = source['canvas']['width'], source['canvas']['height']
    covered_before = _covered(source)
    hidden = []
    for i, s in src.items():
        o = by.get(i)
        if not o or not all(isinstance(o.get(k), (int, float)) for k in ('x', 'y', 'width', 'height')):
            continue
        if float(o.get('opacity', 1)) != float(s.get('opacity', 1)):
            hidden.append(i); continue
        before = _visible_fraction(s, sw, sh)
        if _visible_fraction(o, cw, ch) < min(.5, before * .9):
            hidden.append(i)
    if not fails or all(f['id'] != 'D4' for f in fails):
        hidden += [i for i in _covered(design) - covered_before if i in src]
    if hidden:
        fails.append(_issue('D5', 'Element made effectively invisible (opacity, off-canvas or newly covered)', hidden))
    return {'passed': not fails, 'failures': fails}


# ---------------------------------------------------------------- Tier 1 --

@lru_cache(maxsize=1)
def _validator():
    import jsonschema
    schema = json.loads(SCHEMA.read_text())
    return jsonschema.validators.validator_for(schema)(schema)


def _schema_errors(design):
    try:
        return sorted({'/'.join(map(str, e.absolute_path)) + ': ' + e.validator for e in _validator().iter_errors(design)})
    except Exception as exc:  # jsonschema unavailable: report rather than hide
        return ['schema validator unavailable: ' + type(exc).__name__]


def tier1(source, design, target=None, rendered=None):
    from .invariants import validate
    from .scene import background_cover_allowed
    fails, warns = [], []
    src = {e['id']: e for e in source['elements']}; out = {e.get('id'): e for e in design.get('elements', [])}
    errors = _schema_errors(design)
    if errors:
        inherited = set(_schema_errors(source))
        new = [x for x in errors if x not in inherited]
        (fails if new else warns).append(_issue('S1', 'Schema violations' + ('' if new else ' inherited from source'), [], 'defect' if new else 'warn') | {'detail': (new or errors)[:8]})
    target = target or design['canvas']
    if [design['canvas'].get('width'), design['canvas'].get('height')] != [int(target['width']), int(target['height'])]:
        fails.append(_issue('S2', 'Canvas does not match the requested size'))
    names = {'Required fields present': 'S1', 'Element IDs preserved': 'S3', 'Valid geometry': 'S4', 'Text, fonts & assets unchanged': 'S3',
             'Styles, layering & groups unchanged': 'S7', 'Nothing hidden': 'S4'}
    for c in validate(source, design):
        if c['name'] in names and not c['passed']:
            fails.append(_issue(names[c['name']], c['name'] + ' failed: ' + c['detail']))
    cw, ch = design['canvas']['width'], design['canvas']['height']; sw, sh = source['canvas']['width'], source['canvas']['height']
    off, bleed = [], []
    for i, o in out.items():
        if i not in src or not all(isinstance(o.get(k), (int, float)) for k in ('x', 'y', 'width', 'height')):
            continue
        b = rotated_extent(o)
        if b.x < -.5 or b.y < -.5 or b.x1 > cw + .5 or b.y1 > ch + .5:
            src_out = 1 - _visible_fraction(src[i], sw, sh); now_out = 1 - _visible_fraction(o, cw, ch)
            (bleed if src_out > 0 and now_out <= src_out + .05 else off).append(i)
    if off:
        fails.append(_issue('S5', 'Element extends outside the canvas', off))
    if bleed:
        warns.append(_issue('S5', 'Intentional source bleed retained (pending organizer bleed policy)', bleed, 'warn'))
    stretched = []
    for i, s in src.items():
        o = out.get(i)
        if not o or s['type'] != 'image' or o.get('fit', 'contain') != 'fill' or background_cover_allowed(source, s, o):
            continue
        if min(s['width'], s['height'], o['width'], o['height']) > 0 and abs(math.log((o['width'] / o['height']) / (s['width'] / s['height']))) > .01:
            stretched.append(i)
    if stretched:
        fails.append(_issue('S6', 'Foreground image aspect ratio changed by more than 1%', stretched))
    so, oo = _order(source), _order(design); flipped = set()
    ids = [i for i in src if i in out]
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            if rotated_extent(out[a]).intersection_area(rotated_extent(out[b])) > 0 and (so[a] < so[b]) != (oo[a] < oo[b]):
                flipped |= {a, b}
    if flipped:
        fails.append(_issue('S8', 'Stacking order changed between overlapping elements', flipped))
    extra_top = set(design) - {'canvas', 'elements'}
    extra_fields = [f'{i}.{k}' for i, o in out.items() if i in src for k in set(o) - set(src[i]) - MUTABLE_FIELDS - {'fit'}]
    if extra_top or extra_fields:
        fails.append(_issue('S11', 'Diagnostics or extra fields inside the design document', list(extra_top) + extra_fields))
    if rendered is False:
        fails.append(_issue('S10', 'Official renderer failed on the serialized design'))
    return {'passed': not fails, 'failures': fails, 'warnings': warns}


# ---------------------------------------------------------------- Tier 2 --

_SRC_CACHE: dict[str, Any] = {}


def _source_context(source, A):
    from .quality import _file_hash
    from .canonical import ASSETS_DIR
    resources=[]
    for e in source['elements']:
        if e.get('type')=='image':
            path=(ASSETS_DIR/str(e.get('asset_id',''))).resolve()
            resources.append((str(path),_file_hash(path) if path.is_relative_to(ASSETS_DIR.resolve()) and path.is_file() else 'missing'))
    key = hashlib.sha256(json.dumps([source,resources], sort_keys=True).encode()).hexdigest()
    if key not in _SRC_CACHE:
        from .quality import text_contrast, units
        ratios, busy = text_contrast(source)
        _SRC_CACHE.clear()
        _SRC_CACHE[key] = {'ratios': ratios, 'busy': busy, 'units': {b.id: r for b, r in units(A, source)}}
    return _SRC_CACHE[key]


def safe_zones(w, h):
    """Areas where platform UI covers content (community conventions)."""
    if (w, h) == (1080, 1920):
        return [Box(0, 0, w, 250), Box(0, h - 340, w, 340), Box(0, 0, 65, h), Box(w - 65, 0, 65, h)]
    if (w, h) == (1280, 720):
        return [Box(w - 200, h - 90, 200, 90)]
    if (w, h) == (2560, 1440):
        cx, cy = (w - 1235) / 2, (h - 338) / 2
        return [Box(0, 0, w, cy), Box(0, cy + 338, w, cy), Box(0, cy, cx, 338), Box(cx + 1235, cy, cx, 338)]
    if (w, h) == (728, 90):
        return [Box(0, 0, w, 10), Box(0, h - 10, w, 10), Box(0, 0, 10, h), Box(w - 10, 0, 10, h)]
    return []


def _gap(a: Box, b: Box):
    dx = max(0.0, max(a.x, b.x) - min(a.x1, b.x1)); dy = max(0.0, max(a.y, b.y) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def _bites(text: Box, other: Box):
    """`other` covers part of the glyphs, not just the empty line box edge (descender space, padding)."""
    iw = min(text.x1, other.x1) - max(text.x, other.x); ih = min(text.y1, other.y1) - max(text.y, other.y)
    return iw > 4 and ih > 4 and (ih > .35 * text.h or iw > .35 * text.w)


def _grid_mask(boxes, cw, ch, n=32):
    grid = [[0] * n for _ in range(n)]
    for b in boxes:
        b = b.clip(cw, ch)
        if b.area <= 0:
            continue
        for gy in range(int(b.y / ch * n), min(n, math.ceil(b.y1 / ch * n))):
            for gx in range(int(b.x / cw * n), min(n, math.ceil(b.x1 / cw * n))):
                grid[gy][gx] = 1
    return grid


def _largest_empty(grid):
    n = len(grid); heights = [0] * n; best = 0
    for row in grid:
        heights = [0 if v else h + 1 for v, h in zip(row, heights)]
        stack = []
        for i, h in enumerate(heights + [0]):
            start = i
            while stack and stack[-1][1] >= h:
                start, hh = stack.pop(); best = max(best, hh * (i - start))
            stack.append((start, h))
    return best / (n * n)


def _weighted_centroid(boxes, cw, ch):
    area = sum(b.area for b in boxes) or 1
    return (sum(b.cx * b.area for b in boxes) / area / cw - .5, sum(b.cy * b.area for b in boxes) / area / ch - .5)


def tier2(source, design, A, diagnostics=None, score=None):
    from .blocks import ButtonBlock, RigidBlock
    from .judging import judge_quality
    from .quality import diagnose, ink, units
    from .typography import measure
    cw, ch = float(design['canvas']['width']), float(design['canvas']['height']); sw, sh = A.sw, A.sh
    short = min(cw, ch); src = {e['id']: e for e in source['elements']}; out = {e['id']: e for e in design['elements']}
    d = diagnostics or diagnose(source, design, A); q = judge_quality(source, design, A, d); ctx = _source_context(source, A)
    if score is None:
        from .metrics import score as metric_score
        from .pipeline import _placement
        score = metric_score(source, design, A, _placement(A, design))
    parts = score['parts']; issues = []; stats = {}
    role = lambda i: A.roles.get(i, 'body')
    texts = [i for i, e in out.items() if e['type'] == 'text' and str(e.get('content', '')).strip()]

    # Legibility
    if q['defects']['overflow']:
        issues.append(_issue('legibility.overflow', 'Text is clipped or wider than its box', q['defects']['overflow']))
    if q['defects']['unreadable']:
        issues.append(_issue('legibility.floor', 'Text below its readability floor', q['defects']['unreadable']))
    tiny = [i for i in texts if out[i]['font_size'] < 7]
    if tiny:
        issues.append(_issue('legibility.tiny', 'Text under 7 px', tiny, 'warn'))
    long_lines = []
    for i in texts:
        if role(i) in ('body', 'subhead', 'legal'):
            fit = measure(out[i], int(out[i]['font_size']), max(1, out[i]['width']))
            if fit.lines >= 2 and len(out[i]['content']) / fit.lines > 90:
                long_lines.append(i)
    if long_lines:
        issues.append(_issue('legibility.measure', 'Body lines longer than 90 characters', long_lines, 'warn'))
    legibility = max(0.0, parts['readability'] - 25 * len(q['defects']['overflow']) - 15 * len(q['defects']['unreadable']) - 5 * len(long_lines))

    # Contrast (WCAG 2.2 1.4.3 on composited pixels; regressions only are defects)
    aa = []; kept = []
    def passes_aa(e, ratio):
        large = e['font_size'] >= 24 or (str(e.get('font_weight', '')).lower() in ('bold', '700', '800', '900', 'black') and e['font_size'] >= 18.66)
        return ratio >= (3 if large else 4.5)
    for i in texts:
        ok = passes_aa(out[i], d['contrast'].get(i, 21)); aa.append(ok)
        # Only text that met AA in the source is scored here; brand colours that
        # already failed are preserved intent and fall under the regression rule (§2.2).
        if passes_aa(src[i], ctx['ratios'].get(i, 21)):
            # Same colours at a smaller size can drop out of WCAG's "large text" band;
            # colours are immutable here, so an unchanged ratio counts as kept.
            kept.append(ok or d['contrast'].get(i, 21) >= .95 * ctx['ratios'].get(i, 21))
    stats['wcag_aa_share'] = round(sum(aa) / len(aa), 3) if aa else 1.0
    stats['wcag_aa_kept'] = round(sum(kept) / len(kept), 3) if kept else 1.0
    if d['contrast_regressions']:
        issues.append(_issue('contrast.regression', 'Text loses contrast against what is behind it', d['contrast_regressions']))
    busier = [i for i in texts if d['busyness'].get(i, 0) > ctx['busy'].get(i, 0) + .1]
    if busier:
        issues.append(_issue('contrast.busy', 'Text moved onto a busier background', busier, 'warn'))
    contrast = 100 * (.5 + .5 * stats['wcag_aa_kept']) - 30 * len(d['contrast_regressions']) - 8 * len(busier)

    # Hierarchy
    pairs = inverted = 0; bad = set(); head = [i for i in texts if role(i) == 'headline']
    for n, a in enumerate(texts):
        for b in texts[n + 1:]:
            fa, fb = float(src[a]['font_size']), float(src[b]['font_size'])
            if abs(fa - fb) / max(fa, fb) < .15:
                continue
            hi, lo = (a, b) if fa > fb else (b, a); pairs += 1
            if out[hi]['font_size'] < out[lo]['font_size']:
                inverted += 1; bad |= {hi, lo}
    if inverted:
        sev = 'defect' if inverted >= 3 or bad & set(head) else 'warn'
        issues.append(_issue('hierarchy.inversion', 'Type size order inverted', bad, sev))
    body = [i for i in texts if role(i) in ('body', 'subhead')]
    ratio_ok = 1.0
    if head and body:
        r0 = src[head[0]]['font_size'] / median(src[i]['font_size'] for i in body)
        r1 = out[head[0]]['font_size'] / max(1, median(out[i]['font_size'] for i in body))
        stats['headline_body_ratio'] = [round(r0, 2), round(r1, 2)]
        if r0 > 1.15:
            kept = r1 / r0
            # Shrinking the headline toward body size flattens hierarchy (doc §2.3); growth is milder.
            ratio_ok = max(0.0, (kept - .3) / .7) if kept < 1 else max(.5, 1 - max(0, kept - 1.35))
            if kept < .65:
                issues.append(_issue('hierarchy.scale', f'Headline lost most of its size advantage over body copy ({r0:.1f}:1 to {r1:.1f}:1)', head + body))
            elif abs(kept - 1) > .35:
                issues.append(_issue('hierarchy.scale', 'Headline-to-body size contrast changed by more than 35%', head + body, 'warn'))
    levels0 = len({round(src[i]['font_size']) for i in texts}); levels1 = len({round(out[i]['font_size']) for i in texts})
    split = [i for i in texts for j in texts if i < j and src[i]['font_size'] == src[j]['font_size'] and out[i]['font_size'] != out[j]['font_size']]
    if levels1 > levels0:
        issues.append(_issue('hierarchy.levels', f'Type size levels grew from {levels0} to {levels1}', split, 'warn'))
    hierarchy = 100 * (.4 * (1 - inverted / pairs if pairs else 1) + .25 * ratio_ok + .05 * (levels1 <= levels0)) + .3 * parts['hierarchy']

    # Grouping
    unit = {b.id: r for b, r in units(A, design)}; groups = broken = 0; broken_ids = []
    rigid = [b for b in A.all_blocks() if isinstance(b, (ButtonBlock, RigidBlock)) or (len(b.ids) > 1 and len({src[i].get('group_id') for i in b.ids}) == 1 and src[b.ids[0]].get('group_id'))]
    for b in rigid:
        s0, s1 = ctx['units'].get(b.id), unit.get(b.id)
        if not s0 or not s1 or s0.w <= 0 or s0.h <= 0 or s1.w <= 0 or s1.h <= 0:
            continue
        groups += 1
        err = max(max(abs((element_box(out[i]).cx - s1.x) / s1.w - (element_box(src[i]).cx - s0.x) / s0.w),
                       abs((element_box(out[i]).cy - s1.y) / s1.h - (element_box(src[i]).cy - s0.y) / s0.h)) for i in b.ids)
        if err > .05:
            broken += 1; broken_ids += b.ids
    if broken:
        issues.append(_issue('grouping.rigid', 'A fixed group (button, badge, grouped unit) came apart', broken_ids))
    region = []
    shapes = [e for e in source['elements'] if e['type'] == 'shape' and element_box(e).area < .85 * sw * sh]
    for i in texts:
        hosts = [s for s in shapes if _order(source)[s['id']] < _order(source)[i] and _contains(element_box(s), ink(src[i]), .95)]
        if hosts and not any(_contains(element_box(out[s['id']]), ink(out[i]), .95) for s in hosts):
            region.append(i)
    if region:
        issues.append(_issue('grouping.region', 'Text left the shape or card it sat in', region))
    for key, msg in (('unsupported_text', 'Text left its photo background'), ('scene_displacement', 'Text moved out of its quiet area of the photo')):
        if q['defects'].get(key):
            issues.append(_issue('grouping.photo', msg, q['defects'][key]))
    loose = []
    for b in A.all_blocks():
        if len(b.ids) < 2 or b.id not in unit:
            continue
        boxes = [ink(out[i]) for i in b.ids]
        intra = median(min(_gap(x, y) for y in boxes if y is not x) for x in boxes)
        others = [r for k, r in unit.items() if k != b.id]
        inter = min((_gap(unit[b.id], r) for r in others), default=math.inf)
        if intra > 2 and inter < 1.5 * intra:
            loose.append(b.id)
    if loose:
        issues.append(_issue('grouping.proximity', 'Group members are as far apart as neighbouring groups', loose, 'warn'))
    grouping = 100 * (1 - broken / groups if groups else 1) - 25 * len(region) - 20 * len(q['defects'].get('unsupported_text', [])) - 5 * len(loose)

    # Overlap
    clash = []
    for n, a in enumerate(texts):
        for b in texts[n + 1:]:
            if ink(out[a]).intersection_area(ink(out[b])) > 1 and ink(src[a]).intersection_area(ink(src[b])) <= 1:
                clash += [a, b]
    if clash:
        issues.append(_issue('overlap.text', 'Text overlaps other text', clash))
    # Text newly touching an element of another block (a button, badge or image
    # landing on copy). Block-level collisions miss this when boxes barely touch.
    owner = {i: b.id for b in A.all_blocks() for i in b.ids}; ambient = {e['id'] for e in A.background + A.decorations}
    covered = []
    for t in texts:
        for e in design['elements']:
            o = e['id']
            if o == t or o in ambient or e['type'] == 'text' or owner.get(o) == owner.get(t):
                continue
            if _bites(ink(out[t]), rotated_extent(e)) and not _bites(ink(src[t]), rotated_extent(src[o])):
                covered += [t, o]
    if covered:
        issues.append(_issue('overlap.text', 'Text runs into an element of another group', covered))
    if d['collisions']:
        issues.append(_issue('overlap.collision', 'Blocks collide that did not overlap in the source', [i for c in d['collisions'] for i in c['ids']]))
    occluded = []
    oo, so = _order(design), _order(source)
    for i, e in out.items():
        if e['type'] != 'image' or role(i) not in ('hero', 'image', 'logo'):
            continue
        vb = visual_box(e); vb0 = visual_box(src[i])
        cover = sum(vb.intersection_area(rotated_extent(c)) for c in design['elements'] if oo[c['id']] > oo[i] and c['type'] != 'image')
        cover0 = sum(vb0.intersection_area(rotated_extent(c)) for c in source['elements'] if so[c['id']] > so[i] and c['type'] != 'image')
        if vb.area and cover / vb.area > cover0 / max(1, vb0.area) + .15:
            occluded.append(i)
    if occluded:
        issues.append(_issue('overlap.occlusion', 'Foreground image is newly covered by text or shapes', occluded, 'warn'))
    overlap = 100 - 40 * len(set(clash)) / 2 - 30 * len(set(covered)) / 2 - 15 * len(d['collisions']) - 10 * len(occluded)

    # Alignment
    boxes = [r for r in unit.values() if r.area > 0 and r.w < .98 * cw]
    aligned = near = 0
    for n, a in enumerate(boxes):
        hit = False
        for b in boxes[:n] + boxes[n + 1:]:
            for va, vb_, dim in ((a.x, b.x, cw), (a.cx, b.cx, cw), (a.x1, b.x1, cw), (a.y, b.y, ch), (a.cy, b.cy, ch), (a.y1, b.y1, ch)):
                gap = abs(va - vb_)
                if gap <= 1:
                    hit = True
                elif gap <= .01 * dim:
                    near += 1
        aligned += hit
    near //= 2
    stats['aligned_share'] = round(aligned / len(boxes), 3) if boxes else 1.0
    stats['near_misses'] = near
    stats['grid4_share'] = round(sum(abs(r.x / 4 - round(r.x / 4)) < .13 and abs(r.y / 4 - round(r.y / 4)) < .13 for r in boxes) / len(boxes), 3) if boxes else 1.0
    if near:
        issues.append(_issue('alignment.near_miss', f'{near} near-miss alignments (1 px to 1% apart)', [], 'warn'))
    alignment = 100 * (.7 * stats['aligned_share'] + .3 * (1 - min(1, near / max(1, len(boxes)))))

    # White space, margins, balance
    margin = max(8.0, .03 * short); tight = []
    for b in A.all_blocks():
        r = unit.get(b.id); s0 = ctx['units'].get(b.id)
        if not r or b.role in ('background', 'decoration', 'sticker') or (s0 and (s0.x < 1 or s0.y < 1 or s0.x1 > sw - 1 or s0.y1 > sh - 1)):
            continue
        if min(r.x, r.y, cw - r.x1, ch - r.y1) < margin - .5:
            tight.append(b.id)
    if tight:
        issues.append(_issue('space.margin', f'Content within {margin:.0f} px of the edge', tight, 'warn'))
    zones = safe_zones(int(cw), int(ch)); unsafe = []
    for b in A.all_blocks():
        r = unit.get(b.id)
        if r and b.role in TEXT_ROLES + ('logo',) and any(z.intersection_area(r) > .1 * r.area for z in zones):
            unsafe.append(b.id)
    if unsafe:
        issues.append(_issue('space.safe_zone', 'Important content inside a platform UI zone', unsafe, 'warn'))
    grid = _grid_mask(list(unit.values()), cw, ch); coverage = sum(map(sum, grid)) / (len(grid) ** 2)
    src_grid = _grid_mask(list(ctx['units'].values()), sw, sh); src_cov = sum(map(sum, src_grid)) / (len(src_grid) ** 2)
    stats['coverage'] = round(coverage, 3)
    if coverage < .35 and src_cov >= .35:
        issues.append(_issue('space.empty', f'Content covers only {coverage:.0%} of the canvas', [], 'warn'))
    elif coverage > .85 and src_cov <= .85:
        issues.append(_issue('space.cramped', f'Content covers {coverage:.0%} of the canvas', [], 'warn'))
    dead, dead0 = _largest_empty(grid), _largest_empty(src_grid); stats['largest_empty'] = round(dead, 3)
    if dead > .4 and dead0 <= .4:
        issues.append(_issue('space.dead_zone', f'Empty region covering {dead:.0%} of the canvas', [], 'warn'))
    cols = sorted(boxes, key=lambda r: r.y); gaps = []
    for a, b in zip(cols, cols[1:]):
        if min(a.x1, b.x1) - max(a.x, b.x) > .5 * min(a.w, b.w) and b.y >= a.y1:
            gaps.append(b.y - a.y1)
    if len(gaps) >= 3 and median(gaps) > 0 and pstdev(gaps) / max(1e-6, sum(gaps) / len(gaps)) > .25:
        issues.append(_issue('space.uneven_gaps', 'Vertical gaps between stacked blocks are uneven', [], 'warn'))
    off = math.hypot(*_weighted_centroid(boxes, cw, ch)) if boxes else 0
    off0 = math.hypot(*_weighted_centroid(list(ctx['units'].values()), sw, sh)) if ctx['units'] else 0
    balance = 1 - min(1, max(0, off - off0) * 2.5); stats['balance'] = round(balance, 3)
    # Composition fit: when the aspect changes, content should spread along the axis that grew.
    # Measured relative to the source's own span, so sparse sources are not penalised; a
    # letterbox (uniform shrink) uses exp(-|log aspect change|) of it.
    change = abs(math.log((cw / ch) / (sw / sh))); fit_short = 0.0
    fg = [b for b in (r.clip(cw, ch) for r in unit.values()) if b.area > 0]; fg0 = [b for b in (r.clip(sw, sh) for r in ctx['units'].values()) if b.area > 0]
    if change >= .05 and fg and fg0:
        wide = cw / ch > sw / sh
        ext = lambda rs, dim: (max(r.x1 if wide else r.y1 for r in rs) - min(r.x if wide else r.y for r in rs)) / dim
        span0 = ext(fg0, sw if wide else sh)
        if span0 > .05:
            use = min(1.0, ext(fg, cw if wide else ch) / span0); stats['long_axis_use'] = round(use, 3); fit_short = max(0.0, .85 - use)
            if use < .7:
                issues.append(_issue('space.letterbox', f'Content spans {use:.0%} of its source extent along the {"width" if wide else "height"}; empty bands on a reshaped canvas', [], 'warn'))
    space = 100 * (.35 * min(1, coverage / .35) * (1 if coverage <= .85 else .8) + .25 * balance + .2 * (1 - min(1, len(tight) / 3)) + .2 * (dead <= .4 or dead0 > .4)) - 10 * len(unsafe) - 100 * fit_short

    # Imagery
    weak = []; cropped = []
    for i, e in out.items():
        if e['type'] != 'image' or i in {x['id'] for x in A.background}:
            continue
        share0 = element_box(src[i]).area / (sw * sh); share1 = element_box(e).area / (cw * ch)
        if share0 > .01 and share1 < .25 * share0:
            weak.append(i)
        info = _asset_info(e.get('asset_id') or '')
        if info and e.get('fit') == 'cover' and e['height'] > 0:
            img, box = info[0] / info[1], e['width'] / e['height']
            visible = min(img, box) / max(img, box)
            if min(1, visible / .6) < .9:
                cropped.append(i)
        if role(i) == 'logo' and visual_box(e).h < min(12, visual_box(src[i]).h) - .5:
            weak.append(i)
    if weak:
        issues.append(_issue('imagery.presence', 'Image or logo shrank too far', weak, 'warn'))
    if cropped:
        issues.append(_issue('imagery.crop', 'Cover crop likely cuts the subject', cropped, 'warn'))
    imagery = 100 - 20 * len(set(weak)) - 20 * len(cropped)

    # Fidelity
    focal = []
    if head and any(out[i]['font_size'] > out[head[0]]['font_size'] for i in texts):
        focal.append(head[0])
    hero = A.block('hero')
    if hero and hero.id in unit and any(unit[b.id].area > unit[hero.id].area for b in A.all_blocks() if b.id in unit and b.role in ('image',)):
        focal += hero.ids
    if focal:
        issues.append(_issue('fidelity.focal', 'The original focal point is no longer the most prominent element', focal, 'warn'))
    fidelity = parts['fidelity'] - 20 * bool(focal)

    scores = {k: round(max(0.0, min(100.0, v)), 1) for k, v in (('legibility', legibility), ('contrast', contrast), ('hierarchy', hierarchy), ('grouping', grouping),
              ('overlap', overlap), ('alignment', alignment), ('space', space), ('imagery', imagery), ('fidelity', fidelity))}
    defects = [x for x in issues if x['severity'] == 'defect']
    return {'passed': not defects, 'scores': scores, 'overall': round(sum(WEIGHTS[k] * v for k, v in scores.items()), 1),
            'defects': defects, 'warnings': [x for x in issues if x['severity'] != 'defect'], 'stats': stats,
            'relaxed_floors': bool(q['defects']['unreadable'])}


def evaluate(source, design, A=None, diagnostics=None, score=None, target=None, rendered=None):
    """Full Tier 0-2 report. `gate` is True only when Tier 0 and Tier 1 pass."""
    if A is None:
        from .analysis import analyze
        A = analyze(source)
    t0 = tier0(source, design); t1 = tier1(source, design, target, rendered)
    t2 = tier2(source, design, A, diagnostics, score) if t0['passed'] or not any(f['id'] == 'D4' for f in t0['failures']) else None
    return {'gate': t0['passed'] and t1['passed'], 'passed': t0['passed'] and t1['passed'] and bool(t2 and t2['passed']),
            'tier0': t0, 'tier1': t1, 'tier2': t2}


def summary(report):
    """Compact line for logs."""
    t2 = report['tier2'] or {}
    return {'gate': report['gate'], 'tier0': [f['id'] for f in report['tier0']['failures']], 'tier1': [f['id'] for f in report['tier1']['failures']],
            'defects': [f['id'] for f in t2.get('defects', [])], 'overall': t2.get('overall')}
