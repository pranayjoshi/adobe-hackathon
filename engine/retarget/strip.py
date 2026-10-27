"""Extreme strips (800x90 banners, 90x800 skyscrapers): a reading-order composer.

Crello has almost no designs this elongated (no tall strip beyond 1:3.75 at all), so the model's
predictions are out of distribution there. Designers rebuild such formats as a single row (or
column): the content in reading order along the long side, each piece sized to the short side.
`compose` builds that layout from the solver's units; it is the warm start the constrained solver
then refines under every engine check (solver strip mode).
"""
import copy
import math

STRIP_ASPECT = 4.0


def is_strip(target):
    a = float(target['width']) / float(target['height'])
    return a >= STRIP_ASPECT or a <= 1.0 / STRIP_ASPECT


def long_axis(target):
    return 0 if float(target['width']) >= float(target['height']) else 1


def reading_order(P, ids):
    """Source reading order of units: rows top to bottom, left to right within a row."""
    def key(u):
        x, y, w, h = u.sx, u.sy, u.sw, u.sh
        return (round((y + h / 2) / max(P.H * 0.08, 1.0)), x + w / 2)
    return sorted(ids, key=lambda k: key(P.units[k]))


MAIN_ROLES = {'hero', 'image', 'logo'}
ROLE_SHARE = {'headline': 0.55, 'subhead': 0.32, 'cta': 0.32, 'body': 0.24, 'legal': 0.18}


def _slots(P, order):
    """Main content in reading order, copy stacked in the source kept together as one slot."""
    canvas_area = P.W * P.H
    main = []
    for k in order:
        u = P.units[k]
        has_text = any(P.by[i]['type'] == 'text' for i in u.ids)
        if has_text or (any(P.role[i] in MAIN_ROLES for i in u.ids) and u.sw * u.sh >= 0.03 * canvas_area):
            main.append(k)
    # stacked copy chains (upper -> lower) become one column slot
    below = {}
    for a, b, _ in P.stacked:
        ka, kb = P.owner[a], P.owner[b]
        if P.units[ka].free_text and P.units[kb].free_text:
            below.setdefault(ka, kb)
    slots, used = [], set()
    for k in main:
        if k in used:
            continue
        col = [k]
        while col[-1] in below and below[col[-1]] in main and below[col[-1]] not in used and len(col) < 3:
            col.append(below[col[-1]])
        used |= set(col)
        slots.append(col)
    return slots


def compose(P):
    """Reading-order row/column layout of P's units for an extreme strip target -> design dict."""
    from ..typography import measure
    from .solver import _materialize, M
    tW, tH = P.tW, P.tH
    ax = long_axis(P.target)
    L, S = (tW, tH) if ax == 0 else (tH, tW)
    mar = 0.1 * S
    short = S - 2 * mar
    order = reading_order(P, [u.idx for u in P.units])
    slots = _slots(P, order)
    texts = [P.units[k].ids[0] for col in slots for k in col if P.units[k].free_text]
    head = max(texts, key=lambda t: P.by[t]['font_size']) if texts else None
    # one global type scale: the largest copy reaches its role's share of the strip height (wide
    # strips) or fits the strip width in a few lines (tall strips); hierarchy is kept
    def font_for(t, k_scale):
        e = P.by[t]
        return max(int(math.ceil(P.floor[t] / 0.98)), int(e['font_size'] * k_scale))
    if head is not None:
        e = P.by[head]
        target_h = ROLE_SHARE.get(P.role[head], 0.4) * short if ax == 0 else 0.12 * L
        k_scale = target_h / max(e['font_size'] * float(e.get('line_height') or 1.2), 1.0)
    else:
        k_scale = 1.0
    plan = []  # (slot, [(k, kind, value, w, h)], slot_w, slot_h)
    for col in slots:
        items = []
        for k in col:
            u = P.units[k]
            if u.free_text:
                t = u.ids[0]; e = P.by[t]
                f = font_for(t, k_scale)
                width = short if ax == 1 else 1e6
                fit = measure(e, f, width)
                while (fit.overflow or (ax == 0 and fit.height > short / len(col))) and f > math.ceil(P.floor[t] / 0.98):
                    f -= 1
                    fit = measure(e, f, width)
                w = short if ax == 1 else fit.line_width + 2
                items.append((k, 'text', f, w, fit.height))
            else:
                s = min(short / u.sh, (short / u.sw) if ax == 1 else 1e9, (0.45 * L) / (u.sw if ax == 0 else u.sh))
                items.append((k, 'unit', s, u.sw * s, u.sh * s))
        if ax == 0:
            sw_, sh_ = max(i[3] for i in items), sum(i[4] for i in items) + 0.08 * short * (len(items) - 1)
        else:
            sw_, sh_ = max(i[3] for i in items), sum(i[4] for i in items) + 0.08 * short * (len(items) - 1)
        plan.append((col, items, sw_, sh_))
    gap = 0.4 * short
    along = lambda pl: pl[2] if ax == 0 else pl[3]
    total = sum(along(pl) for pl in plan) + gap * max(0, len(plan) - 1)
    fitk = min(1.0, (L - 2 * mar) / max(total, 1e-6))
    free = max(0.0, L - 2 * mar - total * fitk)
    gap_k = gap * fitk + (free / max(1, len(plan) - 1) if len(plan) > 1 else 0) * 0.5
    pos = mar + (free - (gap_k - gap * fitk) * max(0, len(plan) - 1)) / 2
    v = {'X': {}, 'Y': {}, 'S': {}, 'F': {}, 'TW': {}, 'TH': {}, 'OPT': {}}
    gaps_at = []
    for n, (col, items, sw_, sh_) in enumerate(plan):
        slot_len = (sw_ if ax == 0 else sh_) * fitk
        cross = (sh_ if ax == 0 else sw_) * (fitk if ax == 1 else 1)
        off = (S - cross) / 2
        for k, kind, val, w, h in items:
            u = P.units[k]
            if kind == 'text':
                t = u.ids[0]
                f = max(int(math.ceil(P.floor[t] / 0.98)), int(val * (fitk if fitk < 1 else 1)))
                fit = measure(P.by[t], f, w if ax == 1 else 1e6)
                w2, h2 = (w if ax == 1 else fit.line_width + 2), fit.height
                v['F'][k], v['TW'][k], v['TH'][k] = f, int(w2 * M), int(h2 * M)
            else:
                sc = val * fitk
                v['S'][k] = max(1, int(sc * M))
                w2, h2 = u.sw * sc, u.sh * sc
            if ax == 0:
                x, y = pos + (slot_len - w2) / 2, off
                off += h2 + 0.08 * short
            else:
                x, y = (tW - w2) / 2, pos + off - (S - cross) / 2
                off += h2 + 0.08 * short
            v['X'][k], v['Y'][k] = int(max(0.0, min(x, tW - w2)) * M), int(max(0.0, min(y, tH - h2)) * M)
        pos += slot_len
        if n + 1 < len(plan):
            gaps_at.append((pos, gap_k))
        pos += gap_k
    # decorations and specks: packed side by side into the free stretches along the strip (the
    # gaps between slots and both ends), shrinking a step at a time until they all fit
    rest = [u for u in P.units if u.idx not in v['X']]
    taken = []
    for k in v['X']:
        u = P.units[k]
        if u.free_text:
            w_, h_ = v['TW'][k] / M, v['TH'][k] / M
        else:
            w_, h_ = u.sw * v['S'][k] / M, u.sh * v['S'][k] / M
        x_, y_ = v['X'][k] / M, v['Y'][k] / M
        taken.append((x_, x_ + w_) if ax == 0 else (y_, y_ + h_))
    taken.sort()
    free, cur = [], mar * 0.5
    for a0, a1 in taken:
        if a0 - cur > 2:
            free.append([cur, a0])
        cur = max(cur, a1)
    if L - mar * 0.5 - cur > 2:
        free.append([cur, L - mar * 0.5])
    def dims(u, lim):
        if u.free_text:
            e = P.by[u.ids[0]]
            f = int(math.ceil(P.floor[e['id']] / 0.98))
            fit = measure(e, f, 1e6)
            return ('text', f, fit.line_width + 2, fit.height)
        thin = min(min(float(P.by[i]['width']), float(P.by[i]['height'])) for i in u.ids)
        sc = max(lim / max(u.sw, u.sh), 1.1 / max(thin, 1e-3))
        return ('unit', sc, u.sw * sc, u.sh * sc)
    lim = 0.5 * short
    for _ in range(6):
        placed, ok = {}, True
        cursor = [iv[0] for iv in free]
        for u in sorted(rest, key=lambda u: -(u.sw * u.sh)):
            kind, val, w, h = dims(u, lim)
            along_len, cross_len = (w, h) if ax == 0 else (h, w)
            slot = next((j for j, iv in enumerate(free) if iv[1] - cursor[j] >= along_len + 2 and cross_len <= S - 2), None)
            if slot is None:
                ok = False
                break
            pos_al = cursor[slot] + 1
            cursor[slot] += along_len + max(2.0, 0.08 * short)
            pos_cr = (S - cross_len) / 2
            placed[u.idx] = (kind, val, w, h, pos_al, pos_cr)
        if ok or lim < 2:
            break
        lim *= 0.7
    for u in rest:
        if u.idx in placed:
            kind, val, w, h, pos_al, pos_cr = placed[u.idx]
        else:  # no room left: smallest legal size at the strip's far end
            kind, val, w, h = dims(u, 1.0)
            pos_al, pos_cr = L - mar * 0.25 - (w if ax == 0 else h), (S - (h if ax == 0 else w)) / 2
        if kind == 'text':
            v['F'][u.idx], v['TW'][u.idx], v['TH'][u.idx] = val, int(w * M), int(h * M)
        else:
            v['S'][u.idx] = max(1, int(val * M))
        x, y = (pos_al, pos_cr) if ax == 0 else (pos_cr, pos_al)
        v['X'][u.idx] = int(min(max(0.0, x), max(0.0, tW - w)) * M)
        v['Y'][u.idx] = int(min(max(0.0, y), max(0.0, tH - h)) * M)
    return _materialize(P, v, {})

