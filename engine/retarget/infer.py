"""Numpy inference for the retargeting transformer (mirror of training/retarget/model.py).

Weights are a plain .npz exported by training/retarget/export.py; no torch at runtime.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import erf

from .features import N_BINS, TASK_ARRANGE, TASK_PAIR, centered, decode_target, encode_source

N_FIELDS = 5
BOS = N_FIELDS * N_BINS
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'models' / 'retarget.npz'


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _ln(x, p):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * p['weight'] + p['bias']


def _lin(x, p):
    return x @ p['weight'].T + p['bias']


class Model:
    def __init__(self, path=DEFAULT_WEIGHTS):
        z = np.load(path)
        self.cfg = json.loads(bytes(z['__config__']).decode())
        tree = {}
        for k in z.files:
            if k == '__config__':
                continue
            node = tree
            parts = k.split('.')
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = z[k].astype(np.float32)
        self.p = tree
        self.h = self.cfg['heads']

    def _attn(self, x, mem, mask, p):
        B, T, D = x.shape
        S = mem.shape[1]
        h, dh = self.h, D // self.h
        q = _lin(x, p['q']).reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        kv = _lin(mem, p['kv']).reshape(B, S, 2, h, dh).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
        s = np.where(mask, s, -1e9)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        y = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return _lin(y, p['o'])

    def _block(self, x, p, mask, mem=None, mem_mask=None):
        h = _ln(x, p['ln1'])
        x = x + self._attn(h, h, mask, p['attn'])
        if mem is not None:
            x = x + self._attn(_ln(x, p['ln2']), mem, mem_mask, p['xattn'])
        f = p['ff1'], p['ff2']
        return x + _lin(_gelu(_lin(_ln(x, p['ln3']), f[0])), f[1])

    def encode(self, types, feats, glob):
        p = self.p
        n = len(types)
        x = p['type_emb']['weight'][types] + _lin(_gelu(_lin(feats, p['feat']['0'])), p['feat']['2']) + p['rank_emb']['weight'][:n]
        g = _lin(_gelu(_lin(glob, p['glob']['0'])), p['glob']['2'])
        x = np.concatenate([g[None], x], 0)[None]
        mask = np.ones((1, 1, 1, n + 1), bool)
        for i in range(self.cfg['enc_layers']):
            x = self._block(x, p['enc'][str(i)], mask)
        return _ln(x, p['enc_ln'])[0]

    def sample(self, types, feats, glob, k=8, temperature=0.8, top_p=0.9, seed=0, return_probs=False):
        """Return k bin arrays [n, 5]; sample 0 is greedy, the rest nucleus samples.

        With return_probs, also each row's per-field distribution [k, n, 5, N_BINS] (temperature 1),
        conditioned on that row's own earlier choices: the solver's likelihood costs."""
        p = self.p
        rng = np.random.default_rng(seed)
        mem = self.encode(types, feats, glob)
        n = len(types)
        el_state = _lin(mem[1:], p['el_proj']) + p['rank_emb']['weight'][:n]
        memk = np.repeat(mem[None], k, 0)
        mem_mask = np.ones((1, 1, 1, n + 1), bool)
        bins = np.zeros((k, n, N_FIELDS), np.int64)
        prev = np.zeros((k, n, p['el_proj']['weight'].shape[0]), np.float32)
        prev[:, 0] = p['value_emb']['weight'][BOS]
        offs = np.arange(N_FIELDS) * N_BINS
        probs = np.zeros((k, n, N_FIELDS, N_BINS), np.float32) if return_probs else None
        for j in range(n):
            x = el_state[None, :j + 1] + prev[:, :j + 1]
            causal = np.tril(np.ones((j + 1, j + 1), bool))[None, None]
            for i in range(self.cfg['dec_layers']):
                x = self._block(x, p['dec'][str(i)], causal, memk, mem_mask)
            z = _ln(x[:, j], p['dec_ln'])
            for f in range(N_FIELDS):
                hp = p['heads'][str(f)]
                logits = _lin(_gelu(_lin(z, hp['0'])), hp['2'])
                if return_probs:
                    e = np.exp(logits - logits.max(-1, keepdims=True))
                    probs[:, j, f] = e / e.sum(-1, keepdims=True)
                choice = self._pick(logits, temperature, top_p, rng)
                bins[:, j, f] = choice
                if f < N_FIELDS - 1:
                    z = z + p['in_emb'][str(f)]['weight'][choice]
            if j + 1 < n:
                prev[:, j + 1] = p['value_emb']['weight'][bins[:, j] + offs].sum(1)
        return (bins, probs) if return_probs else bins

    @staticmethod
    def _pick(logits, temperature, top_p, rng):
        out = np.empty(len(logits), np.int64)
        out[0] = int(np.argmax(logits[0]))
        for r in range(1, len(logits)):
            l = logits[r] / max(temperature, 1e-4)
            pr = np.exp(l - l.max()); pr /= pr.sum()
            order = np.argsort(-pr)
            keep = order[: max(1, int(np.searchsorted(np.cumsum(pr[order]), top_p)) + 1)]
            q = pr[keep] / pr[keep].sum()
            out[r] = int(rng.choice(keep, p=q))
        return out


_MODELS = {}


def weights_path():
    return Path(os.environ.get('CHAMELEON_RETARGET_WEIGHTS') or DEFAULT_WEIGHTS)


def load(path=None):
    path = Path(path or weights_path())
    if path not in _MODELS:
        _MODELS[path] = Model(path)
    return _MODELS[path]


def predict_dist(records, W, H, tW, tH, k=4, temperature=0.8, top_p=0.9, seed=0, model=None, task=TASK_PAIR):
    """Like predict, plus per-row field distributions [k, n, 5, N_BINS] for constrained decoding."""
    model = model or load()
    if task == TASK_ARRANGE:
        records, W, H = centered(records, W, H, tW, tH), tW, tH
    types, feats, glob = encode_source(records, W, H, tW, tH, task)
    bins, probs = model.sample(types, feats, glob, k, temperature, top_p, seed, return_probs=True)
    boxes = [[decode_target(b, r, W, H, tW, tH) for b, r in zip(sample, records)] for sample in bins]
    return boxes, probs, (records, W, H)


def predict(records, W, H, tW, tH, k=8, temperature=0.8, top_p=0.9, seed=0, model=None, task=TASK_PAIR):
    """records: element records (see features.py) in stacking order. Returns k lists of boxes.

    task=TASK_PAIR conditions on the source arrangement (sample 0 greedy); TASK_ARRANGE gives the
    model only the elements and their sizes, for fresh arrangements on very different aspects."""
    model = model or load()
    if task == TASK_ARRANGE:
        records, W, H = centered(records, W, H, tW, tH), tW, tH
    types, feats, glob = encode_source(records, W, H, tW, tH, task)
    bins = model.sample(types, feats, glob, k, temperature, top_p, seed)
    return [[decode_target(b, r, W, H, tW, tH) for b, r in zip(sample, records)] for sample in bins]
