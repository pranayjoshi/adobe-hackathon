"""Encoder-decoder transformer that retargets a layout to a new canvas.

Encoder: one token per source element (type, geometry, text attributes) plus a global
token (source aspect, target aspect, element count, task). Decoder: one causal token per
element in source stacking order, tied to that element's encoder state so identity never
drifts; it sees every earlier element's placed box. A small chained head then emits five
discretized fields, (w, h, x, y, font size), each conditioned on the earlier ones. Written with plain linear layers so engine/retarget/infer.py can mirror it in numpy.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.retarget.features import MAX_ELEMENTS, N_BINS, N_FEATURES, N_GLOBAL, N_TYPES

N_FIELDS = 5
BOS = N_FIELDS * N_BINS


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mem, mask):
        B, T, D = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, cross, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.cross = cross
        if cross:
            self.ln2 = nn.LayerNorm(d)
            self.xattn = Attention(d, heads)
        self.ln3 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, self_mask, mem=None, mem_mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, self_mask))
        if self.cross:
            x = x + self.drop(self.xattn(self.ln2(x), mem, mem_mask))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.ln3(x)))))
        return x


class Retargeter(nn.Module):
    def __init__(self, d=256, heads=8, enc_layers=4, dec_layers=6, dropout=0.1):
        super().__init__()
        self.config = dict(d=d, heads=heads, enc_layers=enc_layers, dec_layers=dec_layers)
        self.drop = nn.Dropout(dropout)
        self.type_emb = nn.Embedding(N_TYPES, d)
        self.feat = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(N_GLOBAL, d), nn.GELU(), nn.Linear(d, d))
        self.rank_emb = nn.Embedding(MAX_ELEMENTS, d)
        self.enc = nn.ModuleList([Block(d, heads, False, dropout) for _ in range(enc_layers)])
        self.enc_ln = nn.LayerNorm(d)
        self.el_proj = nn.Linear(d, d)
        self.value_emb = nn.Embedding(N_FIELDS * N_BINS + 1, d)
        self.dec = nn.ModuleList([Block(d, heads, True, dropout) for _ in range(dec_layers)])
        self.dec_ln = nn.LayerNorm(d)
        # Per-field heads chained inside an element: field f sees the element state plus fields < f.
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_BINS)) for _ in range(N_FIELDS)])
        self.in_emb = nn.ModuleList([nn.Embedding(N_BINS, d) for _ in range(N_FIELDS - 1)])

    def encode(self, types, feats, glob, el_mask):
        """types [B,n], feats [B,n,F], glob [B,N_GLOBAL], el_mask [B,n] bool (True = real element)."""
        B, n = types.shape
        ranks = torch.arange(n, device=types.device).expand(B, n)
        x = self.drop(self.type_emb(types) + self.feat(feats) + self.rank_emb(ranks))
        x = torch.cat([self.glob(glob)[:, None], x], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=types.device), el_mask], 1)
        mask = valid[:, None, None, :]
        for blk in self.enc:
            x = blk(x, mask)
        return self.enc_ln(x), mask

    def decoder_inputs(self, mem, prev_bins):
        """prev_bins [B, n, 5]: previous element's bins at each position (first row ignored, BOS used)."""
        B, n, _ = prev_bins.shape
        offs = torch.arange(N_FIELDS, device=mem.device) * N_BINS
        prev = self.value_emb(prev_bins + offs).sum(2)
        prev = torch.cat([self.value_emb.weight[BOS].expand(B, 1, -1), prev[:, 1:]], 1)
        ranks = torch.arange(n, device=mem.device)
        return self.el_proj(mem[:, 1:n + 1]) + prev + self.rank_emb(ranks)[None]

    def decode_hidden(self, mem, mem_mask, prev_bins):
        x = self.decoder_inputs(mem, prev_bins)
        n = x.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None]
        for blk in self.dec:
            x = blk(x, causal, mem, mem_mask)
        return self.dec_ln(x)

    def field_logits(self, z, f):
        h = self.heads[f]
        return h[2](F.gelu(h[0](z)))

    def heads_teacher(self, z, bins):
        """All 5 field logits, each conditioned on the true earlier fields of the same element."""
        out = []
        for f in range(N_FIELDS):
            out.append(self.field_logits(z, f))
            if f < N_FIELDS - 1:
                z = z + self.in_emb[f](bins[..., f])
        return torch.stack(out, 2)  # [B, n, 5, N_BINS]

    def forward(self, types, feats, glob, el_mask, bins):
        mem, mem_mask = self.encode(types, feats, glob, el_mask)
        prev = torch.cat([bins[:, :1], bins[:, :-1]], 1)
        z = self.decode_hidden(mem, mem_mask, prev)
        return self.heads_teacher(z, bins)


def param_count(m):
    return sum(p.numel() for p in m.parameters())
