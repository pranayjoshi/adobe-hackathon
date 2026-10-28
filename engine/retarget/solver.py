"""Constrained layout decoding for learned retargeting (docs/retarget/post-processing-plan.md).

The model gives a distribution for every element's size, position and font size. This module
finds the layout that is most likely under those distributions while satisfying the engine's hard
checks, keeping the source's structure (rigid units, hosts, alignment, reading order, hierarchy)
and the designer priors, in one CP-SAT model.

Geometry is integer: positions in milli-pixels, unit scale in per-mille of the source size, font
sizes in whole pixels. Every member box is linear in its unit's (X, Y, S), so images keep their
aspect exactly and rigid groups stay rigid. Free text (not part of a rigid unit, not rotated)
chooses a line count; each line-count option has exact width/height coefficients measured with the
renderer's own metrics, including absolute letter-spacing. Costs are convex piecewise-linear
envelopes of the model's -log p, plus structural and aesthetic terms with tunable weights
(WEIGHTS). The same engine checks then verify the result; failing pairs are added and the model is
re-solved warm.
"""
import copy
import math
from dataclasses import dataclass, field

import numpy as np

from ..blocks import FLOORS
from ..canonical import backdrop_patch, rotated_extent
from ..scene import background_cover_allowed
from .features import LOG_RANGE, N_BINS, OFF_RANGE, TEXT, base_box

M = 1000  # milli-pixels per pixel; per-mille scale

# Cost weights, in nats-equivalent per unit of the named quantity. Tuned in P5.
WEIGHTS = {
    'model': 2.0,            # x the model's -log p surrogate per field (0.5 per spread squared)
    'text_box': 0.0,         # x model cost on a free text box's own width/height (it follows from font + wrap; matching a narrow predicted box shrinks the font)
    'align': 0.1,            # per 1% of canvas width that a source alignment drifts apart
    'order_flip': 3.0,       # reversing the source reading order of two copy blocks
    'axis_switch': 0.5,      # stack <-> row for a pair (scaled down for large aspect changes)
    'margin': 0.05,          # per 1% of the short side below the format's copy-margin prior
    'fill': 0.0,             # bounding-box fill: off, it is gamed by pushing one small element to a corner
    'balance': 0.05,         # per 1% of canvas the visual centre of mass sits off the optical centre
    'rhythm': 0.5,           # per font height that a stacked gap differs from the source gap ratio
    'typo': 0.3,             # x line-break quality cost of the chosen option
    'lines': 0.05,           # per line away from the source's line count (designers re-wrap freely)
    'headline': 0.2,         # per 1% of the short side outside the format's headline band
    'host': 3.0,             # per text-width of copy outside the photo or panel it sat on
}
STRIP_WEIGHTS = {'model': 0.35, 'strip_stack': 1.0, 'rhythm': 0.1, 'align': 0.1}
RELAX = [(), ('hier',), ('hier', 'contain'), ('hier', 'contain', 'guard')]  # dropped in this order when infeasible
GUARD = 0.6  # no unit shrinks below this share of its warm-start scale (overlap is never fixed by collapsing)
STOPWORDS = {'a', 'an', 'the', 'of', 'and', 'or', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'from', '&',
             'is', 'are', 'your', 'our', 'my', 'de', 'la', 'le', 'et'}


def _clip(v, lo, hi):
    return min(hi, max(lo, v))


def _bin(v, lo, hi):
    return _clip((v - lo) / (hi - lo), 0.0, 1.0) * (N_BINS - 1)


def _gauss(p, window=24):
    """Mode and local spread (in bins) of a field distribution; the spread is floored at 2 bins.

    Centred on the mode, as greedy decoding is (the mean of a broad or skewed distribution sits
    away from the designer-like answer); the spread is measured within +-window bins of it."""
    p = np.asarray(p, dtype=float)
    sm = np.convolve(p, np.ones(3) / 3, mode='same')
    mu = int(np.argmax(sm))
    lo, hi = max(0, mu - window), min(len(p), mu + window + 1)
    idx = np.arange(lo, hi)
    q = p[lo:hi] / max(float(p[lo:hi].sum()), 1e-12)
    sd = math.sqrt(max(float((q * (idx - mu) ** 2).sum()), 4.0))
    return float(mu), sd


def _nll(p, b):
    """Convex surrogate of -log p at continuous (unclipped) bin b: a Gaussian fitted to p.

    The raw -log p flattens far from the mode, and its convex envelope then makes huge moves
    almost free (layouts collapse); the quadratic keeps growing."""
    if not isinstance(p, tuple):
        p = _gauss(p)
    mu, sd = p
    return 0.5 * ((b - mu) / sd) ** 2


def _cbin(v, lo, hi):
    """Continuous bin position without clipping."""
    return (v - lo) / (hi - lo) * (N_BINS - 1)


def _hull(xs, ys):
    """Lower convex hull of points (sorted by x) -> list of (slope, intercept)."""
    pts = sorted(zip(xs, ys))
    h = []
    for x, y in pts:
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (x - x1) >= (y - y1) * (x2 - x1):
                h.pop()
            else:
                break
        h.append((x, y))
    segs = []
    for (x1, y1), (x2, y2) in zip(h, h[1:]):
        if x2 > x1:
            m = (y2 - y1) / (x2 - x1)
            segs.append((m, y1 - m * x1))
    if not segs:
        segs = [(0.0, min(ys))]
    return segs


# ------------------------------------------------------------------------------------------ text
@dataclass
class TextOption:
    lines: int
    a: float          # width = a * font + b  (px)
    b: float
    c: float          # height = c * font     (px)
    w_ref: float      # needed width at the reference font
    cost: float       # line-break quality cost


def _wrap_lines(el, font, width):
    from ..typography import _style_key, _widths, wrap_text
    content = str(el.get('content', ''))
    weight, style = _style_key(el)
    widths = list(_widths(el.get('font_family', ''), weight, style, int(font), content, bool(el.get('capitalize', False))))
    return wrap_text(content, widths, float(width)).split('\n') if content else ['']


def _line_cost(lines, align):
    words = [w for l in lines for w in l.split()]
    cost = 0.0
    if len(words) >= 3 and len(lines) >= 2 and len(lines[-1].split()) == 1:
        cost += 1.0  # widow
    for l in lines[:-1]:
        ws = l.split()
        if ws and ws[-1].lower().strip(',.:;') in STOPWORDS:
            cost += 0.5
    if len(lines) >= 2:
        lens = [len(l.strip()) for l in lines]
        cv = float(np.std(lens) / max(np.mean(lens), 1))
        cost += 0.6 * cv if align != 'left' else 0.25 * cv
    return cost


def text_options(el, f_ref, max_lines):
    """Line-count options for el at reference font f_ref; widths/heights linear in the font size."""
    from ..typography import measure
    f_ref = max(4, int(round(f_ref)))
    wide = measure(el, f_ref, 1e6)
    l_min = wide.lines
    hi = wide.line_width + 2
    # The narrowest width that gives each line count without overflow. Wrapping ignores tracking
    # while the drawn width includes it, so validity is not monotonic in the width: scan a fine
    # ladder of widths instead of bisecting (bisection skips the narrow windows between counts).
    best = {}
    width = hi
    while width >= max(4.0, hi / 16):
        f = measure(el, f_ref, width)
        if not f.overflow and f.lines <= max(l_min, max_lines):
            best[f.lines] = width
        width *= 0.98
    opts, seen = [], set()
    for L in sorted(best):
        b_ = best[L]
        fit = measure(el, f_ref, b_)
        if fit.lines in seen:
            continue
        seen.add(fit.lines)
        lines = _wrap_lines(el, f_ref, b_)
        # width coefficients from two sizes with the same breaks: tracking is the constant term
        widest = max(lines, key=lambda l: measure(dict(el, content=l), f_ref, 1e6).line_width) if lines else ''
        w1 = measure(dict(el, content=widest), f_ref, 1e6).line_width
        w2 = measure(dict(el, content=widest), 2 * f_ref, 1e6).line_width
        a = (w2 - w1) / f_ref
        b = w1 - a * f_ref
        c = fit.height / f_ref
        opts.append(TextOption(fit.lines, a, b, c, b_, _line_cost(lines, el.get('text_align', 'left'))))
    return opts


# ------------------------------------------------------------------------------------- structure
@dataclass
class Unit:
    idx: int
    ids: list
    sx: float
    sy: float
    sw: float
    sh: float
    free_text: bool = False
    rel: dict = field(default_factory=dict)  # id -> (rx, ry, rw, rh) of the rotated extent in px


def _drawn(e):
    """The box the renderer will need: copy whose measured height exceeds its stored box grows
    about its centre (as materialization does), so the solver reserves that space."""
    if e['type'] != 'text':
        return e
    from ..typography import measure
    fit = measure(e, int(e.get('font_size') or 12), max(1.0, float(e['width'])))
    h = float(e['height'])
    if fit.height <= h + 0.5:
        return e
    return dict(e, y=float(e['y']) - (fit.height - h) / 2, height=fit.height)


def _ext(e):
    b = rotated_extent(e)
    return b.x, b.y, b.w, b.h


def _frac(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min(a[2] * a[3], b[2] * b[3]))


def _inside(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, a[2] * a[3])


class Problem:
    """Everything about one (source, target) that does not depend on the seed."""

    def __init__(self, source, target, A, recs, priors=None):
        from .deoverlap import DECOR_MIN, backed, merge
        from .propose import _units
        self.source, self.target, self.A = source, target, A
        self.W, self.H = float(source['canvas']['width']), float(source['canvas']['height'])
        self.tW, self.tH = float(target['width']), float(target['height'])
        self.by = {e['id']: e for e in source['elements']}
        self.rec = {r['id']: r for r in recs}
        self.recs = recs
        self.order = [r['id'] for r in recs]
        self.priors = priors or {}
        from .strip import is_strip, long_axis
        self.strip = is_strip(target)
        self.long_axis = long_axis(target)
        cover = {r['id'] for r in recs if r['cover']}
        self.cover = cover
        W, H = self.W, self.H
        groups = [u for u in _units(source, A) if not (set(u) & cover)]
        grouped = {i for u in groups for i in u}
        units = [list(u) for u in groups]
        for e in source['elements']:
            if e['id'] not in cover and e['id'] not in grouped:
                units.append([e['id']])
        self.units = []
        self.drawn = {e['id']: _drawn(e) for e in source['elements']}
        for k, ids in enumerate(units):
            ex = [_ext(self.drawn[i]) for i in ids]
            x0 = min(b[0] for b in ex); y0 = min(b[1] for b in ex)
            x1 = max(b[0] + b[2] for b in ex); y1 = max(b[1] + b[3] for b in ex)
            u = Unit(k, ids, x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0))
            u.free_text = len(ids) == 1 and self.by[ids[0]]['type'] == 'text' and not self.by[ids[0]].get('rotation')
            u.rel = {i: (b[0] - x0, b[1] - y0, b[2], b[3]) for i, b in zip(ids, ex)}
            self.units.append(u)
        self.owner = {i: u.idx for u in self.units for i in u.ids}
        # collision blocks: analysis blocks + decorations >= DECOR_MIN of the canvas (deoverlap's set)
        ids_all = set(self.owner)
        blocks = [b for b in ([i for i in blk.ids if i in ids_all] for blk in A.all_blocks()) if b]
        seen = {i for b in blocks for i in b}
        blocks += [[e['id']] for e in source['elements'] if e['id'] in ids_all and e['id'] not in seen and e['type'] != 'text'
                   and DECOR_MIN * W * H <= e['width'] * e['height'] < 0.9 * W * H and not e.get('rotation')]
        # a block spanning two units would bind them; keep only blocks inside one unit
        self.blocks = [b for b in blocks if len({self.owner[i] for i in b}) == 1]
        self.bsrc = [self._src_box(b) for b in self.blocks]
        self.texts = [e['id'] for e in source['elements'] if e['type'] == 'text' and e['id'] in self.owner]
        self.role = {i: A.roles.get(i, 'body') for i in self.by}
        area = math.sqrt(self.tW * self.tH / (A.sw * A.sh))
        # the engine's "unreadable" floor (judging.judge_quality), with a small safety margin
        self.floor = {i: min(FLOORS.get(self.role[i], 9), self.by[i]['font_size'] * max(.5, min(1, area))) for i in self.texts}
        self._hosts()
        self._alignment()
        self._stacks()

    def _src_box(self, ids):
        ex = [_ext(self.drawn[i]) for i in ids]
        x0 = min(b[0] for b in ex); y0 = min(b[1] for b in ex)
        return [x0, y0, max(b[0] + b[2] for b in ex) - x0, max(b[1] + b[3] for b in ex) - y0]

    def unit_text_option(self, t):
        """The source's own line option for copy inside a rigid unit (cached)."""
        if not hasattr(self, '_uto'):
            self._uto = {}
        if t not in self._uto:
            from ..typography import measure
            e = self.by[t]
            n = measure(e, int(e['font_size']), float(e['width'])).lines
            opts = text_options(e, e['font_size'], n)
            match = [o for o in opts if o.lines == n] or [o for o in opts if o.lines <= n]
            self._uto[t] = max(match, key=lambda o: o.lines) if match else None
        return self._uto[t]

    def _hosts(self):
        """Copy sitting on a large photo or panel (>= 25% of the canvas) in another unit."""
        self.host = {}
        big = [e for e in self.source['elements'] if e['type'] != 'text' and e['id'] in self.owner
               and e['width'] * e['height'] >= 0.25 * self.W * self.H]
        for t in self.texts:
            tb = _ext(self.by[t])
            best = None
            for h in big:
                if self.owner[h['id']] == self.owner[t] or h.get('z_index', 0) >= self.by[t].get('z_index', 0):
                    continue
                if _inside(tb, _ext(h)) >= 0.9 and (best is None or h['width'] * h['height'] < best['width'] * best['height']):
                    best = h
            if best is not None:
                self.host[t] = best['id']

    def _alignment(self):
        """Source text pairs sharing a left, centre or right edge on different rows."""
        self.aligned = []
        tol = 0.01 * self.W
        bx = {t: _ext(self.by[t]) for t in self.texts}
        for i, a in enumerate(self.texts):
            for b in self.texts[i + 1:]:
                if self.owner[a] == self.owner[b]:
                    continue
                A_, B_ = bx[a], bx[b]
                if min(A_[1] + A_[3], B_[1] + B_[3]) - max(A_[1], B_[1]) > 0.5 * min(A_[3], B_[3]):
                    continue  # same row
                for kind, fa, fb in (('l', A_[0], B_[0]), ('r', A_[0] + A_[2], B_[0] + B_[2]),
                                     ('c', A_[0] + A_[2] / 2, B_[0] + B_[2] / 2)):
                    if abs(fa - fb) <= tol:
                        self.aligned.append((a, b, kind))
                        break

    def _stacks(self):
        """Consecutive copy stacked vertically: (upper, lower, gap / upper font) in the source."""
        self.stacked = []
        bx = {t: _ext(self.by[t]) for t in self.texts}
        for a in self.texts:
            A_ = bx[a]
            best = None
            for b in self.texts:
                if b == a or self.owner[a] == self.owner[b]:
                    continue
                B_ = bx[b]
                if min(A_[0] + A_[2], B_[0] + B_[2]) - max(A_[0], B_[0]) <= 0:
                    continue
                gap = B_[1] - (A_[1] + A_[3])
                fa = self.by[a]['font_size']
                if -0.2 * fa <= gap <= 2.5 * fa and (best is None or gap < best[1]):
                    best = (b, gap)
            if best:
                self.stacked.append((a, best[0], best[1] / max(self.by[a]['font_size'], 1.0)))


# ---------------------------------------------------------------------------------------- solve
class Solution:
    def __init__(self, design, status, cost, parts, notes):
        self.design, self.status, self.cost, self.parts, self.notes = design, status, cost, parts, notes


def _margin(P, box):
    from .deoverlap import margin
    return margin(box, P.W, P.H, P.tW, P.tH)


def _aclass(w, h):
    a = w / h
    return 'tall<0.5' if a < 0.5 else 'portrait0.5-0.9' if a < 0.9 else 'square0.9-1.1' if a < 1.1 else \
        'landscape1.1-2' if a < 2 else 'wide2-3.5' if a < 3.5 else 'banner>3.5'


def _prior(P, key, q, default):
    pri = P.priors.get('ac:' + _aclass(P.tW, P.tH), {}).get(key) or P.priors.get('all', {}).get(key) or {}
    return pri.get(q, default) if isinstance(pri, dict) else default


def solve(P, seed_boxes, probs, weights=None, time_limit=0.25, extra_pairs=(), hint=None, rounds=6, disable=(), workers=8,
          deadline=None):
    """Solve, relaxing soft-able hard constraints (hierarchy, containment) if the model is infeasible.
    deadline: time.monotonic() value after which no new round or relaxation starts."""
    import time
    sol = fallback = None
    for extra in RELAX:
        if deadline is not None and time.monotonic() > deadline and (sol is not None or fallback is not None):
            break
        sol = _solve_once(P, seed_boxes, probs, weights, time_limit, extra_pairs, hint, rounds, tuple(disable) + extra, workers,
                          deadline)
        if sol is not None and sol.design is not None:
            if extra:
                sol.notes.append('relaxed:' + '+'.join(extra))
            if not getattr(sol, 'incomplete', False):
                return sol
            fallback = fallback or sol
    return fallback or sol


def _solve_once(P, seed_boxes, probs, weights=None, time_limit=0.25, extra_pairs=(), hint=None, rounds=6, disable=(), workers=8,
                deadline=None):
    """hint: a design (e.g. today's pipeline output or the previous round) to warm-start the search."""
    """Solve for one seed. seed_boxes: decoded model boxes per record (same order as P.recs);
    probs: [n, 5, N_BINS] distributions for that seed. Returns a Solution."""
    from ortools.sat.python import cp_model
    w = dict(WEIGHTS, **(weights or {}))
    if P.strip:
        # Out of the model's distribution: trust its sizes less; the reading-order composer is the
        # warm start and strip ordering costs replace the source-axis ones.
        w = dict(w, **STRIP_WEIGHTS, **{k: v for k, v in (weights or {}).items() if k.startswith('strip_')})
    tW, tH = P.tW, P.tH
    short = min(tW, tH)
    seed = {r['id']: b for r, b in zip(P.recs, seed_boxes)}
    pr = {r['id']: probs[k] for k, r in enumerate(P.recs)}
    pairs = set(extra_pairs)
    tight = {}  # pair -> factor on its allowed overlap; tightened when the engine still flags it
    fref = {}   # free-text unit -> reference font for its line options (re-measured at the solution)
    force = set()  # pairs the rubric flags at element level: always fully separated
    notes = []
    best = None
    import time
    for rnd in range(rounds):
        if deadline is not None and best is not None and time.monotonic() > deadline:
            # out of time: the last solution keeps the violations this round would have fixed
            notes.append(f'round{rnd}:deadline')
            best.incomplete = best.design is not None
            break
        m = cp_model.CpModel()
        X, Y, S, F, TW, TH, OPT = {}, {}, {}, {}, {}, {}, {}
        UT = {}  # unit text id -> (left, top, width, height) expressions
        BB = {}  # block union boxes that contain flexible text
        cost_terms = []
        parts = {}

        def add_cost(name, expr_var, weight):
            parts.setdefault(name, []).append((expr_var, weight))

        # ---------- units
        for u in P.units:
            e0 = P.by[u.ids[0]]
            texts = [i for i in u.ids if P.by[i]['type'] == 'text']
            # canvas-fit scale bound and minimum scale (hairlines >= 1 px, copy >= its floor)
            mar = _margin(P, (u.sx, u.sy, u.sw, u.sh)) if texts else 0.0
            if u.free_text:
                el = e0
                f_seed = fref.get(u.idx) or _clip(seed[el['id']]['font_size'] or el['font_size'], P.floor[el['id']], 4 * el['font_size'] + 1)
                src_lines = max(1, int(P.rec[el['id']].get('lines') or 1))
                opts = text_options(el, f_seed, max(src_lines * 2 + 2, 3))
                if not opts:
                    opts = text_options(el, f_seed, 60)
                fmin = max(1, int(math.ceil(P.floor[el['id']] / 0.98)))
                fmax = max(fmin, int(4 * el['font_size'] + 1))
                f = m.NewIntVar(fmin, fmax, f'f{u.idx}')
                tw = m.NewIntVar(M, int((tW - 2 * mar) * M), f'tw{u.idx}')
                th = m.NewIntVar(M, int((tH - 2 * mar) * M), f'th{u.idx}')
                bo = [m.NewBoolVar(f'o{u.idx}_{j}') for j in range(len(opts))]
                m.AddExactlyOne(bo)
                for j, o in enumerate(opts):
                    m.Add(tw >= int(round(o.a * M)) * f + int(round(o.b * M)) + M).OnlyEnforceIf(bo[j])
                    # stay below the next-fewer-lines width so the renderer keeps this break count
                    fewer = [q for q in opts if q.lines < o.lines]
                    if fewer:
                        q = max(fewer, key=lambda z: z.lines)
                        m.Add(tw <= int(round(q.a * M)) * f + int(round(q.b * M)) - M).OnlyEnforceIf(bo[j])
                    m.Add(th >= int(math.ceil(o.c * M)) * f + M // 2).OnlyEnforceIf(bo[j])
                    m.Add(th <= int(math.ceil(o.c * M)) * f + 2 * M).OnlyEnforceIf(bo[j])
                F[u.idx], TW[u.idx], TH[u.idx], OPT[u.idx] = f, tw, th, (bo, opts)
                x = m.NewIntVar(int(mar * M), int((tW - mar) * M), f'x{u.idx}')
                y = m.NewIntVar(int(mar * M), int((tH - mar) * M), f'y{u.idx}')
                m.Add(x + tw <= int((tW - mar) * M)); m.Add(y + th <= int((tH - mar) * M))
                X[u.idx], Y[u.idx] = x, y
                # typography and line-count costs
                for j, o in enumerate(opts):
                    c = w['typo'] * o.cost + w['lines'] * abs(o.lines - src_lines)
                    if c > 0:
                        add_cost('typo', bo[j], c)
            else:
                # nothing below 1 px after rounding: the element's own thinner side, with headroom
                thin = min(min(float(P.by[i]['width']), float(P.by[i]['height'])) for i in u.ids)
                kmin = max(1, int(math.ceil(1.1 * M / max(thin, 1e-3))))
                for t in texts:
                    kmin = max(kmin, int(math.ceil(M * P.floor[t] / 0.98 / max(P.by[t]['font_size'], 1))))
                kmax = int(M * min((tW - 2 * mar) / u.sw, (tH - 2 * mar) / u.sh, 6.0))
                if kmin > kmax:
                    notes.append(f'unit{u.idx}:floor_infeasible')
                    kmin = max(1, min(kmin, kmax))
                s = m.NewIntVar(kmin, max(kmin, kmax), f's{u.idx}')
                x = m.NewIntVar(int(mar * M) - 10 ** 7, int(tW * M), f'x{u.idx}')
                y = m.NewIntVar(int(mar * M) - 10 ** 7, int(tH * M), f'y{u.idx}')
                m.Add(x >= int(mar * M)); m.Add(y >= int(mar * M))
                m.Add(x + int(round(u.sw)) * s <= int((tW - mar) * M))
                m.Add(y + int(round(u.sh)) * s <= int((tH - mar) * M))
                X[u.idx], Y[u.idx], S[u.idx] = x, y, s
                # copy inside a rigid unit: its centre is rigid, its box must fit its measured
                # line option at the scaled font (absolute letter-spacing included)
                for t in texts:
                    te = P.by[t]
                    if te.get('rotation'):
                        continue
                    o = P.unit_text_option(t)
                    if o is None:
                        continue
                    rx, ry, rw, rh = u.rel[t]
                    wt = m.NewIntVar(M, int(tW * M), ''); ht = m.NewIntVar(M, int(tH * M), '')
                    fs_mpx = int(round(te['font_size'])) * s  # font in milli-px
                    m.Add(wt >= int(round(rw)) * s)
                    m.Add(1000 * wt >= int(round(o.a * 1000)) * fs_mpx + int(round(o.b * 1000)) * M + 2 * 1000 * M)
                    m.Add(ht >= int(round(rh)) * s)
                    m.Add(1000 * ht >= int(math.ceil(o.c * 1000)) * fs_mpx + 1000 * M // 2)
                    lx = m.NewIntVar(-10 ** 8, 10 ** 8, ''); ty = m.NewIntVar(-10 ** 8, 10 ** 8, '')
                    cx2 = 2 * x + int(round(2 * rx + rw)) * s
                    cy2 = 2 * y + int(round(2 * ry + rh)) * s
                    m.Add(2 * lx <= cx2 - wt); m.Add(2 * lx >= cx2 - wt - 1)
                    m.Add(2 * ty <= cy2 - ht); m.Add(2 * ty >= cy2 - ht - 1)
                    m.Add(lx >= int(mar * M)); m.Add(ty >= int(mar * M))
                    m.Add(lx + wt <= int((tW - mar) * M)); m.Add(ty + ht <= int((tH - mar) * M))
                    UT[t] = (lx, ty, wt, ht)

        def box(i):
            """(x, y, w, h) of element i's extent as linear expressions in milli-px."""
            u = P.units[P.owner[i]]
            if u.free_text:
                return X[u.idx], Y[u.idx], TW[u.idx], TH[u.idx]
            if i in UT:
                return UT[i]
            rx, ry, rw, rh = u.rel[i]
            s = S[u.idx]
            return (X[u.idx] + int(round(rx)) * s, Y[u.idx] + int(round(ry)) * s, int(round(rw)) * s, int(round(rh)) * s)

        def font(i):
            u = P.units[P.owner[i]]
            if u.free_text:
                return F[u.idx] * M, M  # expression in milli-px, scale
            return int(round(P.by[i]['font_size'])) * S[u.idx], M

        def bbox(ids):
            # union of a block's member extents (rigid members of one unit)
            u = P.units[P.owner[ids[0]]]
            if u.free_text:
                return box(ids[0])
            if any(i in UT for i in ids):
                key = tuple(sorted(ids))
                if key not in BB:
                    bs = [box(i) for i in ids]
                    x0 = m.NewIntVar(-10 ** 8, 10 ** 8, ''); y0 = m.NewIntVar(-10 ** 8, 10 ** 8, '')
                    x1 = m.NewIntVar(-10 ** 8, 10 ** 8, ''); y1 = m.NewIntVar(-10 ** 8, 10 ** 8, '')
                    m.AddMinEquality(x0, [_as_var(m, q[0]) for q in bs]); m.AddMinEquality(y0, [_as_var(m, q[1]) for q in bs])
                    m.AddMaxEquality(x1, [_as_var(m, q[0] + q[2]) for q in bs]); m.AddMaxEquality(y1, [_as_var(m, q[1] + q[3]) for q in bs])
                    BB[key] = (x0, y0, x1 - x0, y1 - y0)
                return BB[key]
            ex = [u.rel[i] for i in ids]
            x0 = min(b[0] for b in ex); y0 = min(b[1] for b in ex)
            x1 = max(b[0] + b[2] for b in ex); y1 = max(b[1] + b[3] for b in ex)
            s = S[u.idx]
            return (X[u.idx] + int(round(x0)) * s, Y[u.idx] + int(round(y0)) * s, int(round(x1 - x0)) * s, int(round(y1 - y0)) * s)

        # ---------- model likelihood (convex envelopes of -log p)
        def pwl(expr, xs, ys, scale, name, weight):
            """cost var >= every segment of the lower hull of (xs, ys); xs in the units of expr."""
            segs = _hull(xs, ys)
            lo = min(ys)
            c = m.NewIntVar(0, 10 ** 9, '')
            for slope, icpt in segs:
                # M*c >= slope*M*expr/scale + (icpt-lo)*M   (c in milli-nats)
                num = int(round(slope * M * M / scale))
                m.Add(M * c >= num * expr + int(round((icpt - lo) * M * M)) - 0)
            add_cost(name, c, weight / M)
            return c

        for i in P.order:
            if i in P.cover or i not in P.owner:
                continue
            r = P.rec[i]
            u = P.units[P.owner[i]]
            p = pr[i]
            b = base_box(r, P.W, P.H, tW, tH)
            bx, by_, bw, bh = box(i)
            # centre offsets: dx = (cx - base cx) / tW, in bins
            # cost as a function of the element centre (milli-px)
            for axis, tot, base_c, expr_pos, expr_size in ((2, tW, b['x'] + b['w'] / 2, bx, bw), (3, tH, b['y'] + b['h'] / 2, by_, bh)):
                g = _gauss(p[axis])
                xs = np.linspace(-0.3 * tot * M, 1.3 * tot * M, 65)
                ys = [_nll(g, _cbin((v / M - base_c) / tot, *OFF_RANGE)) for v in xs]
                centre = m.NewIntVar(-10 ** 8, 10 ** 8, '')
                m.Add(2 * centre <= 2 * expr_pos + expr_size)
                m.Add(2 * centre >= 2 * expr_pos + expr_size - 1)
                pwl(centre, xs, ys, 1.0, 'model', w['model'])
            if u.free_text:
                # width and height of the text box (px*M) and font (px)
                for fld, expr, ref in ((0, TW[u.idx], b['w']), (1, TH[u.idx], b['h'])):
                    xs = np.exp(np.linspace(-3.0, 2.5, 56)) * ref * M
                    g = _gauss(p[fld])
                    ys = [_nll(g, _cbin(math.log(max(v / M, 1e-3) / max(ref, 1e-3)), *LOG_RANGE)) for v in xs]
                    pwl(expr, xs, ys, 1.0, 'model', w['text_box'] * w['model'])  # box size follows from font and wrap
                if b['font_size'] > 0:
                    xs = np.exp(np.linspace(-3.0, 2.5, 56)) * b["font_size"]
                    g = _gauss(p[4])
                    ys = [_nll(g, _cbin(math.log(v / b['font_size']), *LOG_RANGE)) for v in xs]
                    pwl(F[u.idx], xs, ys, 1.0, 'model', w['model'])
            else:
                s = S[u.idx]
                e = P.by[i]
                ks = np.exp(np.linspace(-4.0, 2.5, 66)) * M
                gw, gh, gf = _gauss(p[0]), _gauss(p[1]), _gauss(p[4])
                ys = []
                for kk in ks:
                    sc = kk / M
                    c = _nll(gw, _cbin(math.log(max(float(e['width']) * sc, 1e-3) / max(b['w'], 1e-3)), *LOG_RANGE))
                    c += _nll(gh, _cbin(math.log(max(float(e['height']) * sc, 1e-3) / max(b['h'], 1e-3)), *LOG_RANGE))
                    if e['type'] == 'text' and b['font_size'] > 0:
                        c += _nll(gf, _cbin(math.log(max(e['font_size'] * sc, 1e-3) / b['font_size']), *LOG_RANGE))
                    ys.append(c)
                pwl(s, ks, ys, 1.0, 'model', w['model'] / max(1, len(u.ids)) ** 0.5)

        # ---------- block separation (the engine's collision definition) and occlusion
        nb = len(P.blocks)
        before = {}
        for a in range(nb):
            for c in range(a + 1, nb):
                if P.owner[P.blocks[a][0]] != P.owner[P.blocks[c][0]]:
                    before[a, c] = _frac(P.bsrc[a], P.bsrc[c])
        gap = int(0.012 * short * M)
        text_blocks = {k for k, bl in enumerate(P.blocks) if any(P.by[i]['type'] == 'text' for i in bl)}
        # initial pairs: every pair involving copy (reading order needs their relation), plus
        # pairs the seed or the warm start puts on top of each other
        if rnd == 0:
            starts = []
            seed_box = []
            for bl in P.blocks:
                bs = [seed[i] for i in bl]
                x0 = min(q['x'] for q in bs); y0 = min(q['y'] for q in bs)
                seed_box.append([x0, y0, max(q['x'] + q['w'] for q in bs) - x0, max(q['y'] + q['h'] for q in bs) - y0])
            starts.append(seed_box)
            if hint is not None:
                starts.append(_block_boxes(P, hint))
            for (a, c), f0 in before.items():
                if (a in text_blocks or c in text_blocks) or any(_frac(b_[a], b_[c]) > f0 + 0.02 for b_ in starts):
                    pairs.add((a, c))
        rel_bools = {}
        hint_boxes, conflicts = None, set()
        if hint is not None:
            hint_boxes = _block_boxes(P, hint)
            conflicts = {k for k in pairs if k[0] < len(hint_boxes) and k[1] < len(hint_boxes)
                         and _frac(hint_boxes[k[0]], hint_boxes[k[1]]) > 0}
        bucket = abs(math.log((P.W / P.H) / (tW / tH)))
        switch_cost = w['axis_switch'] * (1.0 if bucket < 0.3 else 0.6 if bucket < 0.7 else 0.25 if bucket < 1.1 else 0.1)
        for a, c in sorted(pairs):
            f0 = before.get((a, c))
            if f0 is None or 'pairs' in disable:
                continue
            A_, C_ = bbox(P.blocks[a]), bbox(P.blocks[c])
            sa, sc_ = P.bsrc[a], P.bsrc[c]
            if f0 <= 0.0 or (a, c) in force:
                # disjoint in the source (or flagged by the rubric's text-overlap rules): separate.
                # Directions the warm start already separates stay fixed; only conflicts are open.
                l, rr, ab, be = (m.NewBoolVar('') for _ in range(4))
                m.AddBoolOr([l, rr, ab, be])
                if hint_boxes is not None and (a, c) not in conflicts:
                    ha, hc = hint_boxes[a], hint_boxes[c]
                    free = [ha[0] + ha[2] <= hc[0], hc[0] + hc[2] <= ha[0], ha[1] + ha[3] <= hc[1], hc[1] + hc[3] <= ha[1]]
                    if any(free):
                        for v, keep in zip((l, rr, ab, be), free):
                            if not keep:
                                m.Add(v == 0)
                m.Add(A_[0] + A_[2] + gap <= C_[0]).OnlyEnforceIf(l)
                m.Add(C_[0] + C_[2] + gap <= A_[0]).OnlyEnforceIf(rr)
                m.Add(A_[1] + A_[3] + gap <= C_[1]).OnlyEnforceIf(ab)
                m.Add(C_[1] + C_[3] + gap <= A_[1]).OnlyEnforceIf(be)
                rel_bools[a, c] = (l, rr, ab, be)
                # (gap is required only between pairs that conflicted; see the verify loop)
                # reading order between copy blocks: source "a before c" on its separating axis
                if a in text_blocks and c in text_blocks and P.strip:
                    # strips: copy runs in source reading order along the long side; stacking
                    # across the short side is allowed at a cost, reversing the order is not
                    a_first = ((round((sa[1] + sa[3] / 2) / max(P.H * 0.08, 1.0)), sa[0]) <=
                               (round((sc_[1] + sc_[3] / 2) / max(P.H * 0.08, 1.0)), sc_[0]))
                    along = (l, rr) if P.long_axis == 0 else (ab, be)
                    across = (ab, be) if P.long_axis == 0 else (l, rr)
                    add_cost('order', along[1] if a_first else along[0], w['order_flip'])
                    for v in across:
                        add_cost('order', v, w['strip_stack'])
                elif a in text_blocks and c in text_blocks:
                    gx = max(sc_[0] - (sa[0] + sa[2]), sa[0] - (sc_[0] + sc_[2]))
                    gy = max(sc_[1] - (sa[1] + sa[3]), sa[1] - (sc_[1] + sc_[3]))
                    vertical = gy >= gx
                    a_first = (sa[1] < sc_[1]) if vertical else (sa[0] < sc_[0])
                    keep_same = (ab if a_first else be) if vertical else (l if a_first else rr)
                    keep_other = (l if a_first else rr) if vertical else (ab if a_first else be)
                    flip = [v for v in (l, rr, ab, be) if v is not keep_same and v is not keep_other]
                    for v in flip:
                        add_cost('order', v, w['order_flip'])
                    add_cost('order', keep_other, switch_cost)
            elif 'contain' in disable:
                pass
            elif _inside(sa, sc_) >= 0.9 or _inside(sc_, sa) >= 0.9:
                inner, outer = (A_, C_) if _inside(sa, sc_) >= 0.9 else (C_, A_)
                m.Add(inner[0] >= outer[0]); m.Add(inner[1] >= outer[1])
                m.Add(inner[0] + inner[2] <= outer[0] + outer[2]); m.Add(inner[1] + inner[3] <= outer[1] + outer[3])
            else:
                # partial overlap in the source: keep the order on the shallowest axis, bounded overlap
                from .deoverlap import _relation
                # the overlap along the chosen axis is bounded by the source overlap fraction of the
                # smaller block's own (variable) size, so the bound scales with the layout
                ta = seed_box_k(P, seed, P.blocks[a])
                tc = seed_box_k(P, seed, P.blocks[c])
                axis, a_first, _ = _relation(sa, sc_, ta, tc, f0)
                first, second = (A_, C_) if a_first else (C_, A_)
                small = A_ if sa[2] * sa[3] <= sc_[2] * sc_[3] else C_
                frac = min(1.0, (f0 + 0.02) * max(tight.get((a, c), 1.0), 0.2))
                k = int(round(1000 * frac))
                m.Add(1000 * (second[axis] - first[axis] - first[axis + 2]) + k * small[axis + 2] >= 0)

        # ---------- hierarchy: clearly larger source copy stays larger
        for i in P.texts:
            for j in P.texts:
                if i != j and P.by[i]['font_size'] >= 1.25 * P.by[j]['font_size'] and P.owner[i] != P.owner[j] and 'hier' not in disable:
                    fi, _ = font(i); fj, _ = font(j)
                    m.Add(10 * fi >= 11 * fj)

        # ---------- hosts: copy stays on the photo/panel it sat on (soft, per side)
        for t, h in P.host.items():
            T_, H_ = box(t), box(h)
            ref = max(float(P.by[t]['width']), 1.0) * M
            for expr in (H_[0] - T_[0], H_[1] - T_[1], (T_[0] + T_[2]) - (H_[0] + H_[2]), (T_[1] + T_[3]) - (H_[1] + H_[3])):
                sl = m.NewIntVar(0, 10 ** 8, '')
                m.Add(sl >= expr)
                add_cost('host', sl, w['host'] / ref)

        # ---------- alignment kept
        for a, b, kind in P.aligned:
            A_, B_ = box(a), box(b)
            if kind == 'l':
                d = A_[0] - B_[0]
            elif kind == 'r':
                d = A_[0] + A_[2] - B_[0] - B_[2]
            else:
                d = 2 * A_[0] + A_[2] - 2 * B_[0] - B_[2]
            v = m.NewIntVar(0, 10 ** 8, '')
            m.AddAbsEquality(v, d)
            add_cost('align', v, w['align'] * 100 / (tW * M) / (2 if kind == 'c' else 1))

        # ---------- spacing rhythm: stacked copy keeps its gap in font heights (when still stacked)
        for a, b, ratio in P.stacked:
            A_, B_ = box(a), box(b)
            fa, _ = font(a)
            dev = 100 * (B_[1] - A_[1] - A_[3]) - int(round(_clip(ratio, -1, 6) * 100)) * fa
            v = m.NewIntVar(0, 10 ** 9, '')
            m.Add(100 * v >= dev)
            m.Add(100 * v >= -dev)
            ref = max(P.by[a]['font_size'] * min(tW / P.W, tH / P.H), 4.0) * M
            add_cost('rhythm', v, w['rhythm'] / ref)

        # ---------- content box: fill and balance; copy margins near the format median
        rights, bottoms, lefts, tops = [], [], [], []
        mass_terms, mass_tot = [], 0
        for u in P.units:
            bx_ = bbox(u.ids) if not u.free_text else box(u.ids[0])
            lefts.append(bx_[0]); tops.append(bx_[1])
            rv = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.Add(rv == bx_[0] + bx_[2]); rights.append(rv)
            bv = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.Add(bv == bx_[1] + bx_[3]); bottoms.append(bv)
            sb = seed_box_k(P, seed, u.ids)
            wt = 1.6 if any(P.by[i]['type'] == 'text' for i in u.ids) else 1.0
            mass = max(1, int(round(100 * wt * sb[2] * sb[3] / (tW * tH))))
            mass_terms.append((mass, bx_)); mass_tot += mass
        if P.units:
            L = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.AddMinEquality(L, [_as_var(m, e) for e in lefts])
            T = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.AddMinEquality(T, [_as_var(m, e) for e in tops])
            R = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.AddMaxEquality(R, rights)
            B = m.NewIntVar(-10 ** 8, 10 ** 8, ''); m.AddMaxEquality(B, bottoms)
            # designers' content box covers q50 of the canvas area; ask for most of it on each axis
            share_w = share_h = 0.95 * math.sqrt(_prior(P, 'content_bbox_share', 'q50', 0.83))
            for span, tot, share in ((R - L, tW, share_w), (B - T, tH, share_h)):
                sl = m.NewIntVar(0, 10 ** 8, '')
                m.Add(sl >= int(share * tot * M) - span)
                add_cost('fill', sl, w['fill'] * 100 / (tot * M))
            # visual balance: weighted centre of mass near the canvas centre (optical: slightly above)
            for axis, tot, centre in ((0, tW, 0.5), (1, tH, 0.48)):
                dev = m.NewIntVar(-10 ** 12, 10 ** 12, '')
                m.Add(dev == sum(ms * (2 * bx_[axis] + bx_[axis + 2]) for ms, bx_ in mass_terms) - int(2 * centre * tot * M) * mass_tot)
                v = m.NewIntVar(0, 10 ** 12, '')
                m.AddAbsEquality(v, dev)
                add_cost('balance', v, w['balance'] * 100 / (2 * tot * M * mass_tot))
        med_margin = _prior(P, 'text_margin_min', 'q25', 0.08)
        for t in P.texts:
            T_ = box(t)
            want = int(med_margin * short * M)
            for expr in (T_[0], T_[1], int(tW * M) - T_[0] - T_[2], int(tH * M) - T_[1] - T_[3]):
                sl = m.NewIntVar(0, 10 ** 8, '')
                m.Add(sl >= want - expr)
                add_cost('margin', sl, w['margin'] * 100 / (short * M))
        # headline inside the format's designer band
        heads = [t for t in P.texts if P.role[t] == 'headline'] or sorted(P.texts, key=lambda t: -P.by[t]['font_size'])[:1]
        for t in heads[:1]:
            lo = _prior(P, 'headline_fs_over_short', 'q05', 0.05) * short
            hi = _prior(P, 'headline_fs_over_short', 'q95', 0.35) * short
            fx, _ = font(t)
            for expr in (int(lo * M) - fx, fx - int(hi * M)):
                sl = m.NewIntVar(0, 10 ** 8, '')
                m.Add(sl >= expr)
                add_cost('headline', sl, w['headline'] * 100 / (short * M))

        # ---------- warm start, and the collapse guard relative to it
        if hint is not None:
            _add_hint(m, P, hint, X, Y, S, F, TW, TH, OPT)
            if 'guard' not in disable:
                hb = {e['id']: e for e in hint['elements']}
                for u in P.units:
                    if u.free_text or any(i not in hb for i in u.ids):
                        continue
                    sh = sum(float(hb[i]['width']) for i in u.ids) / max(sum(float(P.by[i]['width']) for i in u.ids), 1e-6)
                    m.Add(S[u.idx] >= int(GUARD * sh * M))

        # ---------- objective
        terms = []
        for name, lst in parts.items():
            for var, wt in lst:
                terms.append((var, wt))
        scale = 1000.0
        m.Minimize(sum(int(round(wt * scale * 1e6)) * var for var, wt in terms if int(round(wt * scale * 1e6)) != 0))
        sv = cp_model.CpSolver()
        # deterministic work budget (reproducible under any machine load); wall clock only as a cap
        sv.parameters.max_deterministic_time = time_limit
        cap = 4 * time_limit + 2
        if deadline is not None:
            cap = min(cap, max(0.3, deadline - time.monotonic()))
        sv.parameters.max_time_in_seconds = cap
        sv.parameters.num_workers = workers
        sv.parameters.random_seed = 1
        st = sv.Solve(m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            notes.append(f'round{rnd}:{sv.StatusName(st)}')
            if best is None:
                best = Solution(None, sv.StatusName(st), None, {}, list(notes))
            else:
                best.incomplete = True  # the last solution still has the violations that made it fail
            break
        values = {
            'X': {k: sv.Value(v) for k, v in X.items()}, 'Y': {k: sv.Value(v) for k, v in Y.items()},
            'S': {k: sv.Value(v) for k, v in S.items()}, 'F': {k: sv.Value(v) for k, v in F.items()},
            'TW': {k: sv.Value(v) for k, v in TW.items()}, 'TH': {k: sv.Value(v) for k, v in TH.items()},
            'OPT': {k: next(j for j, b in enumerate(bo) if sv.Value(b)) for k, (bo, _) in OPT.items()},
            'UT': {i: tuple(sv.Value(q) for q in qs) for i, qs in UT.items()},
        }
        part_vals = {name: round(sum(sv.Value(var) * wt for var, wt in lst), 3) for name, lst in parts.items()}
        design = _materialize(P, values, OPT)
        hint = design
        best = Solution(design, sv.StatusName(st), sv.ObjectiveValue(), part_vals, list(notes))
        # verify with the engine's own collision and occlusion checks; failing pairs join the model
        rub = _rubric_violations(P, design)
        force |= rub
        found = _violations(P, design, before) | _engine_violations(P, design) | rub
        new = found - pairs
        again = found & pairs
        remeasure = _text_mismatch(P, design, values, OPT)
        for k in remeasure:
            fref[k] = values['F'][k]
        if not new and not again and not remeasure:
            break
        for k in again:
            tight[k] = tight.get(k, 1.0) * 0.3
        pairs |= new
        notes.append(f'round{rnd}:+{len(new)}pairs,{len(again)}tightened,{len(remeasure)}remeasured')
    if best is not None:
        best.notes = notes
    return best


def _add_hint(m, P, design, X, Y, S, F, TW, TH, OPT):
    by = {e['id']: e for e in design['elements']}

    proto = m.Proto()

    def put(var, val):
        dom = proto.variables[var.Index()].domain
        m.AddHint(var, int(min(dom[-1], max(dom[0], round(val)))))
    for u in P.units:
        if any(i not in by for i in u.ids):
            continue
        if u.free_text:
            e = by[u.ids[0]]
            put(X[u.idx], e['x'] * M); put(Y[u.idx], e['y'] * M)
            put(TW[u.idx], e['width'] * M); put(TH[u.idx], e['height'] * M)
            put(F[u.idx], e['font_size'])
            bo, opts = OPT[u.idx]
            from ..typography import measure
            n = measure(e, int(e['font_size']), e['width']).lines
            j = min(range(len(opts)), key=lambda k: abs(opts[k].lines - n))
            for k, b in enumerate(bo):
                m.AddHint(b, int(k == j))
        else:
            src_w = sum(float(P.by[i]['width']) for i in u.ids)
            s = sum(float(by[i]['width']) for i in u.ids) / max(src_w, 1e-6)
            ex = [_ext(by[i]) for i in u.ids]
            put(S[u.idx], s * M)
            put(X[u.idx], min(b[0] for b in ex) * M); put(Y[u.idx], min(b[1] for b in ex) * M)


def _as_var(m, expr):
    v = m.NewIntVar(-10 ** 8, 10 ** 8, '')
    m.Add(v == expr)
    return v


def seed_box_k(P, seed, ids):
    bs = [seed[i] for i in ids if i in seed]
    if not bs:
        return [0, 0, 1, 1]
    x0 = min(q['x'] for q in bs); y0 = min(q['y'] for q in bs)
    return [x0, y0, max(q['x'] + q['w'] for q in bs) - x0, max(q['y'] + q['h'] for q in bs) - y0]


def _block_boxes(P, design):
    by = {e['id']: e for e in design['elements']}
    boxes = []
    for bl in P.blocks:
        ex = [_ext(by[i]) for i in bl]
        x0 = min(b[0] for b in ex); y0 = min(b[1] for b in ex)
        boxes.append([x0, y0, max(b[0] + b[2] for b in ex) - x0, max(b[1] + b[3] for b in ex) - y0])
    return boxes


def _violations(P, design, before):
    boxes = _block_boxes(P, design)
    out = set()
    for (a, c), f0 in before.items():
        if _frac(boxes[a], boxes[c]) > f0 + 0.02:
            out.add((a, c))
    return out


def _text_mismatch(P, design, v, OPT):
    """Free-text units whose re-measured wrap differs from the chosen option (other line count,
    overflow, or taller than solved): their options are rebuilt at the solved font."""
    from ..typography import measure
    by = {e['id']: e for e in design['elements']}
    out = []
    for k, (bo, opts) in OPT.items():
        u = P.units[k]
        e = by[u.ids[0]]
        f = measure(e, int(e['font_size']), e['width'])
        o = opts[v['OPT'][k]]
        if f.overflow or f.lines != o.lines or f.height > v['TH'][k] / M + 0.5:
            out.append(k)
    return out


def _rubric_violations(P, design):
    """Element pairs behind the rubric's blocking 'overlap.text' rules: copy whose ink newly
    touches other copy, or that a graphic of another block newly bites into."""
    from ..quality import ink
    from ..rubric import _bites
    by = {e['id']: e for e in design['elements']}
    src = P.by
    texts = [t for t in P.texts if t in by]
    out = set()

    def add(a, b):
        if a in P.owner and b in P.owner and P.owner[a] != P.owner[b]:
            ka, kb = _block_index(P, [a]), _block_index(P, [b])
            out.add((min(ka, kb), max(ka, kb)))
    for n, a in enumerate(texts):
        for b in texts[n + 1:]:
            if ink(by[a]).intersection_area(ink(by[b])) > 1 and ink(src[a]).intersection_area(ink(src[b])) <= 1:
                add(a, b)
    owner = {i: b.id for b in P.A.all_blocks() for i in b.ids}
    ambient = {e['id'] for e in P.A.background + P.A.decorations}
    for t in texts:
        for e in design['elements']:
            o = e['id']
            if o == t or o in ambient or e['type'] == 'text' or owner.get(o) == owner.get(t):
                continue
            if _bites(ink(by[t]), rotated_extent(e)) and not _bites(ink(src[t]), rotated_extent(src[o])):
                add(t, o)
    return out


def _block_index(P, ids):
    """Index of a (possibly new) separation block made of ids that share one unit."""
    key = tuple(sorted(ids))
    for k, bl in enumerate(P.blocks):
        if tuple(sorted(bl)) == key:
            return k
    P.blocks.append(list(ids))
    P.bsrc.append(P._src_box(list(ids)))
    return len(P.blocks) - 1


def _engine_violations(P, design):
    """Block pairs behind the engine's collision and new-occlusion findings."""
    from ..pipeline import _new_occlusion
    from ..quality import collisions
    out = set()

    def split(ids):
        by_unit = {}
        for i in ids:
            if i in P.owner:
                by_unit.setdefault(P.owner[i], []).append(i)
        return [_block_index(P, v) for v in by_unit.values()]
    try:
        found = collisions(P.source, design, P.A)
    except Exception:
        found = []
    blocks = {b.id: [i for i in b.ids] for b in P.A.all_blocks()}
    for c in found:
        for ka in split(blocks.get(c['a'], [])):
            for kb in split(blocks.get(c['b'], [])):
                if P.owner[P.blocks[ka][0]] != P.owner[P.blocks[kb][0]]:
                    out.add((min(ka, kb), max(ka, kb)))
    try:
        hidden = _new_occlusion(P.source, design)
    except Exception:
        hidden = []
    if hidden:
        by = {e['id']: e for e in design['elements']}
        z = {e['id']: e.get('z_index', 0) for e in P.source['elements']}
        for h in hidden:
            if h not in P.owner:
                continue
            hb = _ext(by[h])
            for e in design['elements']:
                o = e['id']
                if o in P.owner and o != h and e['type'] != 'text' and z[o] > z[h] and P.owner[o] != P.owner[h] \
                        and _inside(hb, _ext(e)) > 0.0 and _inside(hb, _ext(e)) >= 0.2:
                    ka, kb = _block_index(P, [h]), _block_index(P, [o])
                    out.add((min(ka, kb), max(ka, kb)))
    return out


def _materialize(P, v, OPT):
    """Engine design from solver values."""
    from ..typography import measure
    tW, tH = P.tW, P.tH
    out = []
    for e in P.source['elements']:
        o = copy.deepcopy(e)
        i = e['id']
        if i in P.cover:
            if background_cover_allowed(P.source, e, {**e, 'fit': 'cover'}):
                o.update(x=0.0, y=0.0, width=tW, height=tH, fit='cover')
            else:
                b = backdrop_patch(e, tW, tH)
                o.update(x=b['x'], y=b['y'], width=b['width'], height=b['height'])
            out.append(o)
            continue
        u = P.units[P.owner[i]]
        if u.free_text:
            fs = int(v['F'][u.idx])
            x, y = v['X'][u.idx] / M, v['Y'][u.idx] / M
            w, h = v['TW'][u.idx] / M, v['TH'][u.idx] / M
            fit = measure(o, fs, w)
            h = max(h, fit.height)
            o.update(x=round(x, 2), y=round(y, 2), width=round(w, 2), height=round(h, 2), font_size=fs)
            out.append(o)
            continue
        s = v['S'][u.idx] / M
        rx, ry, rw, rh = u.rel[i]
        ex_x = v['X'][u.idx] / M + round(rx) * s
        ex_y = v['Y'][u.idx] / M + round(ry) * s
        cx, cy = ex_x + round(rw) * s / 2, ex_y + round(rh) * s / 2
        dr = P.drawn[i]
        w, h = float(dr['width']) * s, float(dr['height']) * s
        o.update(x=round(cx - w / 2, 2), y=round(cy - h / 2, 2), width=round(w, 2), height=round(h, 2))
        if e['type'] == 'text' and i in v.get('UT', {}):
            lx, ty, wt, ht = (q / M for q in v['UT'][i])
            floor = math.ceil(P.floor.get(i, 1) / 0.98)
            fs = max(1, floor, int(round(e['font_size'] * s)))
            o.update(x=round(lx, 2), y=round(ty, 2), width=round(wt, 2), height=round(ht, 2), font_size=fs)
            fit = measure(o, fs, wt)
            if fit.height > ht + 0.5:
                o['height'] = round(fit.height, 2)
            out.append(o)
            continue
        if e['type'] == 'text':
            # Scaled copy keeps its absolute letter-spacing, so it can be wider than its scaled box.
            # Keep the font (never below the engine's floor) and widen the box about its centre.
            floor = math.ceil(P.floor.get(i, 1) / 0.98)
            fs = max(1, floor, int(round(e['font_size'] * s)))
            o['font_size'] = fs
            fit = measure(o, fs, w)
            if fit.overflow:
                need = min(tW, measure(o, fs, 1e6).line_width + 2)
                if measure(o, fs, need).lines == fit.lines or need >= tW:
                    o['x'] = round(min(max(0.0, o['x'] - (need - w) / 2), tW - need), 2)
                    o['width'] = round(need, 2)
                    w = need
                fit = measure(o, fs, w)
            if fit.height > h + 0.5:
                grow = fit.height - h
                o['y'] = round(max(0.0, min(tH - fit.height, o['y'] - grow / 2)), 2)
                o['height'] = round(fit.height, 2)
        out.append(o)
    return {'canvas': {'width': int(tW), 'height': int(tH)}, 'elements': out}
