"""Training examples for the retargeter: designer resize pairs and noisy single-design tasks."""
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from engine.retarget.features import (BACKGROUND, MAX_ELEMENTS, TASK_ARRANGE, TASK_NOISY, TASK_PAIR, TEXT,
                                      N_GLOBAL, centered, encode_source, encode_target)


def load(root):
    root = Path(root)
    designs = pickle.load(open(root / 'designs.pkl', 'rb'))
    pairs = pickle.load(open(root / 'pairs.pkl', 'rb'))
    return designs, pairs


def flip(els, W):
    out = []
    for e in els:
        e = dict(e)
        e['x'] = W - e['x'] - e['w']
        e['angle'] = -(e.get('angle') or 0.0)
        if e['type'] == TEXT and e.get('align') in (1, 3):
            e['align'] = 4 - e['align']
        out.append(e)
    return out


def pair_example(designs, p, aug):
    A, B = designs[p['src']], designs[p['tgt']]
    keys = sorted(p['map'])[:MAX_ELEMENTS]
    src = [A['els'][i] for i in keys]
    tgt = [B['els'][p['map'][i]] for i in keys]
    W, H, tW, tH = A['W'], A['H'], B['W'], B['H']
    if aug:
        if random.random() < 0.5:
            src, tgt = flip(src, W), flip(tgt, tW)
        src, tgt = drop_elements(src, tgt, W, H)
        src = jitter(src, W, H)
    return src, (W, H), tgt, (tW, tH), TASK_PAIR


def drop_elements(src, tgt, W, H, p=0.15):
    """Augmentation against memorizing families: remove a random subset of non-background elements."""
    if random.random() < 0.5 or len(src) <= 3:
        return src, tgt
    keep = [i for i, e in enumerate(src) if is_cover(e, W, H) or random.random() > p]
    if len(keep) < 2:
        return src, tgt
    return [src[i] for i in keep], [tgt[i] for i in keep]


def jitter(els, W, H, pos=0.01, size=0.03):
    out = []
    for e in els:
        if is_cover(e, W, H):
            out.append(e); continue
        k = math.exp(random.gauss(0, size))
        cx = e['x'] + e['w'] / 2 + random.gauss(0, pos) * W
        cy = e['y'] + e['h'] / 2 + random.gauss(0, pos) * H
        w, h = e['w'] * k, e['h'] * k
        out.append(dict(e, x=cx - w / 2, y=cy - h / 2, w=w, h=h, font_size=(e.get('font_size') or 0) * k))
    return out


def is_cover(e, W, H):
    return e['type'] == BACKGROUND or (e['w'] >= 0.95 * W and e['h'] >= 0.95 * H)


def noisy_example(design, aspects, aug):
    """Target: a real design. Source: a distorted, jittered version of it on another aspect ratio."""
    W, H = design['W'], design['H']
    tgt = design['els'][:MAX_ELEMENTS]
    if aug and random.random() < 0.5:
        tgt = flip(tgt, W)
    r = random.choice(aspects)
    while abs(math.log(r / (W / H))) < 0.1:
        r = random.choice(aspects)
    area = W * H
    sW, sH = math.sqrt(area * r), math.sqrt(area / r)
    sx, sy = sW / W, sH / H
    mode = random.random()
    src = []
    for e in tgt:
        e = dict(e)
        if is_cover(e, W, H):
            e.update(x=e['x'] * sx, y=e['y'] * sy, w=e['w'] * sx, h=e['h'] * sy)
        else:
            cx, cy = (e['x'] + e['w'] / 2) * sx, (e['y'] + e['h'] / 2) * sy
            # Uniform scale between "fit" and "geometric mean" with per-element noise.
            s = (min(sx, sy) if mode < 0.4 else math.sqrt(sx * sy)) * math.exp(random.gauss(0, 0.2))
            cx += random.gauss(0, 0.06) * sW
            cy += random.gauss(0, 0.06) * sH
            w, h = e['w'] * s, e['h'] * s
            e.update(x=cx - w / 2, y=cy - h / 2, w=w, h=h)
            if e['type'] == TEXT:
                e['font_size'] = (e['font_size'] or 0) * s
        src.append(e)
    return src, (sW, sH), tgt, (W, H), TASK_NOISY


def arrange_example(design, aug):
    """Target: a real design. Source: its elements at their own sizes, all stacked at the centre."""
    W, H = design['W'], design['H']
    tgt = design['els'][:MAX_ELEMENTS]
    if aug and random.random() < 0.5:
        tgt = flip(tgt, W)
    return centered(tgt, W, H, W, H), (W, H), tgt, (W, H), TASK_ARRANGE


def tensorize(examples):
    n = max(len(s) for s, *_ in examples)
    B = len(examples)
    types = np.zeros((B, n), np.int64)
    feats = np.zeros((B, n, 32), np.float32)
    glob = np.zeros((B, N_GLOBAL), np.float32)
    mask = np.zeros((B, n), bool)
    bins = np.zeros((B, n, 5), np.int64)
    loss_mask = np.zeros((B, n, 5), np.float32)
    for b, (src, (W, H), tgt, (tW, tH), task) in enumerate(examples):
        t, f, g = encode_source(src, W, H, tW, tH, task)
        k = len(src)
        types[b, :k], feats[b, :k], glob[b], mask[b, :k] = t, f, g, True
        for i, e in enumerate(tgt):
            bins[b, i] = encode_target(e, src[i], W, H, tW, tH)
            loss_mask[b, i, :4] = 1.0
            loss_mask[b, i, 4] = 1.0 if (e['type'] == TEXT and src[i]['type'] == TEXT and (e.get('font_size') or 0) > 0) else 0.0
    return {k: torch.from_numpy(v) for k, v in dict(types=types, feats=feats, glob=glob, mask=mask,
                                                       bins=bins, loss_mask=loss_mask).items()}


class Sampler:
    def __init__(self, designs, pairs, split, pair_frac=0.5, arrange_frac=0.25, aug=True):
        self.designs = designs
        self.pairs = [p for p in pairs if designs[p['src']]['split'] == split]
        self.singles = [d for d in designs if d['split'] == split and len(d['els']) <= MAX_ELEMENTS]
        fam = Counter(p['family'] for p in self.pairs)
        # Families weigh 1/sqrt(size); larger aspect changes weigh more (stretch already solves small ones).
        def change(p):
            A, B = designs[p['src']], designs[p['tgt']]
            return abs(math.log((A['W'] / A['H']) / (B['W'] / B['H'])))
        w = np.array([(1.0 + 0.5 * change(p)) / math.sqrt(fam[p['family']]) for p in self.pairs])
        self.pair_p = w / w.sum()
        self.aspects = [d['W'] / d['H'] for d in designs]
        self.pair_frac = pair_frac
        self.arrange_frac = arrange_frac
        self.aug = aug

    def batch(self, size):
        ex = []
        n_pairs = sum(random.random() < self.pair_frac for _ in range(size)) if self.pairs else 0
        for i in np.random.choice(len(self.pairs), n_pairs, p=self.pair_p) if n_pairs else []:
            ex.append(pair_example(self.designs, self.pairs[i], self.aug))
        for _ in range(size - n_pairs):
            d = random.choice(self.singles)
            if random.random() < self.arrange_frac / max(1e-6, 1 - self.pair_frac):
                ex.append(arrange_example(d, self.aug))
            else:
                ex.append(noisy_example(d, self.aspects, self.aug))
        return tensorize(ex)
