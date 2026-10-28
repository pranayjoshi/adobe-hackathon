"""Strip adapter: low-rank (LoRA) corrections to the frozen retargeter, switched on by the target's
aspect ratio, for extreme strips (4:1 and beyond, wide or tall).

The base model is frozen and the adapter's gate is exactly 0 below 3:1, so every other format
keeps today's predictions bit for bit. Training data: real designer pairs whose target is 3:1 or
beyond (Crello has 1600x400 and 728x90 families, no tall strips) plus synthetic strip layouts that
passed every engine check (tools/postproc/strips/synth.py). A KL term keeps the adapted
distribution near the base model's, and early stopping uses held-out strips.

    .venv/bin/python -m training.retarget.adapter --out work/retarget-runs/strip-a1
    .venv/bin/python -m training.retarget.adapter --export work/retarget-runs/strip-a1/best.pt engine/models/retarget_strip.npz
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.retarget.features import MAX_ELEMENTS, TASK_PAIR
from engine.retarget.infer import strip_gate
from training.retarget.data import drop_elements, flip, jitter, load, pair_example, tensorize
from training.retarget.model import Retargeter
from training.retarget.train import soft_targets

TARGETS = ['dec.{i}.attn.q', 'dec.{i}.attn.o', 'dec.{i}.xattn.q', 'dec.{i}.xattn.o', 'dec.{i}.ff2']
HEADS = ['heads.{f}.0']


class Gate:
    """Per-example gate shared by every adapted layer; set before each forward pass."""
    value = None


class Adapted(nn.Module):
    def __init__(self, base, rank, gate):
        super().__init__()
        self.base = base
        self.gate = gate
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        y = self.base(x)
        g = self.gate.value
        if g is None:
            return y
        g = g.view(-1, *([1] * (x.dim() - 1)))
        return y + g * ((x @ self.A.T) @ self.B.T)


def target_names(model):
    names = [t.format(i=i) for i in range(len(model.dec)) for t in TARGETS]
    return names + [h.format(f=f) for f in range(len(model.heads)) for h in HEADS]


def add_adapters(model, rank):
    gate = Gate()
    for name in target_names(model):
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        leaf = name.rsplit('.', 1)[1]
        setattr(parent, leaf, Adapted(getattr(parent, leaf), rank, gate))
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for name in target_names(model):
        m = model.get_submodule(name)
        m.A.requires_grad_(True); m.B.requires_grad_(True)
        params += [m.A, m.B]
    return gate, params


def adapter_state(model):
    return {name: {'A': model.get_submodule(name).A.detach().cpu(), 'B': model.get_submodule(name).B.detach().cpu()}
            for name in target_names(model)}


def load_adapter_state(model, state):
    for name, ab in state.items():
        m = model.get_submodule(name)
        m.A.data.copy_(ab['A']); m.B.data.copy_(ab['B'])


# ------------------------------------------------------------------------------------------ data
def _held_out(design_id):
    return int(hashlib.md5(str(design_id).encode()).hexdigest(), 16) % 10 == 0


def strip_data(designs, pairs, synthetic_files):
    """(train, val) example sources: ('pair', pair) for real pairs at 3:1 or beyond, ('syn', row) for
    synthetic strips. Synthetic rows come from train-split designs; a tenth (by design) is held out."""
    def ratio(d):
        a = d['W'] / d['H']
        return max(a, 1 / a)
    real = {'train': [], 'val': []}
    for p in pairs:
        sp = designs[p['src']]['split']
        if sp in real and ratio(designs[p['tgt']]) >= 3:
            real[sp].append(('pair', p))
    syn = {'train': [], 'val': []}
    for f in synthetic_files:
        for line in open(f):
            r = json.loads(line)
            if not r.get('kept') or not r.get('target'):
                continue
            d = designs[r['src']]
            if d['id'] != r.get('src_id', d['id']) or len(r['target']['els']) != len(d['els']):
                continue
            syn['val' if _held_out(d['id']) else 'train'].append(('syn', r))
    return real, syn


def syn_example(designs, r, aug):
    A = designs[r['src']]
    t = r['target']
    src = A['els'][:MAX_ELEMENTS]
    tgt = t['els'][:MAX_ELEMENTS]
    W, H, tW, tH = A['W'], A['H'], t['W'], t['H']
    if aug:
        if random.random() < 0.5:
            src, tgt = flip(src, W), flip(tgt, tW)
        src, tgt = drop_elements(src, tgt, W, H)
        src = jitter(src, W, H)
    return src, (W, H), tgt, (tW, tH), TASK_PAIR


def example(designs, item, aug):
    kind, x = item
    return pair_example(designs, x, aug) if kind == 'pair' else syn_example(designs, x, aug)


def batch(designs, items, aug, device):
    exs = [example(designs, it, aug) for it in items]
    b = {k: v.to(device) for k, v in tensorize(exs).items()}
    gate = torch.tensor([strip_gate(tw, th) for *_, (tw, th), _t in exs], dtype=torch.float32, device=device)
    return b, gate


def losses(model, gate_holder, b, gate, kl_weight, sigma=1.5):
    gate_holder.value = None
    with torch.no_grad():
        base = model(b['types'], b['feats'], b['glob'], b['mask'], b['bins']).float()
    gate_holder.value = gate
    logits = model(b['types'], b['feats'], b['glob'], b['mask'], b['bins']).float()
    lm = b['loss_mask']
    nll = (-(soft_targets(b['bins'], sigma) * F.log_softmax(logits, -1)).sum(-1) * lm).sum() / lm.sum().clamp(min=1)
    pb = F.softmax(base, -1)
    kl = ((pb * (F.log_softmax(base, -1) - F.log_softmax(logits, -1))).sum(-1) * lm).sum() / lm.sum().clamp(min=1)
    base_nll = (-(soft_targets(b['bins'], sigma) * F.log_softmax(base, -1)).sum(-1) * lm).sum() / lm.sum().clamp(min=1)
    return nll, kl, base_nll


def evaluate(model, gate_holder, designs, items, device):
    model.eval()
    tot = {'nll': 0.0, 'base_nll': 0.0, 'kl': 0.0}
    n = 0
    with torch.no_grad():
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            b, g = batch(designs, chunk, False, device)
            nll, kl, base_nll = losses(model, gate_holder, b, g, 0.0, sigma=1.0)
            for k, v in (('nll', nll), ('base_nll', base_nll), ('kl', kl)):
                tot[k] += v.item() * len(chunk)
            n += len(chunk)
    return {k: round(v / max(n, 1), 4) for k, v in tot.items()}


# ----------------------------------------------------------------------------------------- train
def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    designs, pairs = load(a.data)
    ck = torch.load(a.base, map_location='cpu')
    model = Retargeter(**{k: ck['config'][k] for k in ('d', 'heads', 'enc_layers', 'dec_layers')}, dropout=a.dropout)
    model.load_state_dict(ck['state'])
    gate_holder, params = add_adapters(model, a.rank)
    # dropout off: the frozen base must give the same distribution in both passes of the KL term
    model.to(device).eval()
    real, syn = strip_data(designs, pairs, a.synthetic)
    val_real = list(real['val'])
    random.Random(1).shuffle(val_real)
    val_real = val_real[:600]
    val_syn = syn['val']
    print(json.dumps({'adapter_params': sum(p.numel() for p in params), 'train_real': len(real['train']), 'train_syn': len(syn['train']),
                      'val_real': len(val_real), 'val_syn': len(val_syn), 'device': device}), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    best, bad, log, t0, run = float('inf'), 0, [], time.time(), None
    for step in range(1, a.steps + 1):
        n_syn = sum(random.random() < a.syn_frac for _ in range(a.batch))
        items = random.choices(syn['train'], k=n_syn) + random.choices(real['train'], k=a.batch - n_syn)
        b, g = batch(designs, items, True, device)
        nll, kl, _ = losses(model, gate_holder, b, g, a.kl)
        loss = nll + a.kl * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            vr = evaluate(model, gate_holder, designs, val_real, device)
            vs = evaluate(model, gate_holder, designs, val_syn, device)
            # selection: the mean of held-out real and synthetic strips, so neither source dominates
            score = (vr['nll'] + vs['nll']) / 2
            rec = {'step': step, 'train': round(run, 4), 'val_real': vr, 'val_syn': vs, 'score': round(score, 4),
                   'min': round((time.time() - t0) / 60, 1)}
            log.append(rec); print(json.dumps(rec), flush=True)
            state = {'config': dict(ck['config'], rank=a.rank, targets=target_names(model)), 'adapter': adapter_state(model),
                     'step': step, 'score': score, 'base': str(a.base)}
            torch.save(state, out / 'last.pt')
            if score < best - 1e-4:
                best, bad = score, 0
                torch.save(state, out / 'best.pt')
            else:
                bad += 1
                if bad >= a.patience:
                    print(json.dumps({'early_stop': step}), flush=True)
                    break
    (out / 'log.json').write_text(json.dumps(log, indent=1))


# ---------------------------------------------------------------------------------------- export
def export(ckpt, out, base_npz, data):
    """Adapter weights as a small .npz next to the base weights; checks numpy parity with torch."""
    from engine.retarget.infer import Model
    ck = torch.load(ckpt, map_location='cpu')
    arrays = {}
    for name, ab in ck['adapter'].items():
        arrays[name + '.lora_A'] = ab['A'].numpy().astype(np.float32)
        arrays[name + '.lora_B'] = ab['B'].numpy().astype(np.float32)
    cfg = {'rank': ck['config']['rank'], 'step': ck['step'], 'score': ck['score'], 'gate': 'aspect 3:1 -> 4:1 ramp'}
    arrays['__config__'] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    np.savez_compressed(out, **arrays)
    # parity: torch adapted greedy decode vs numpy on held-out real strips
    base = torch.load(ck['base'], map_location='cpu')
    model = Retargeter(**{k: base['config'][k] for k in ('d', 'heads', 'enc_layers', 'dec_layers')})
    model.load_state_dict(base['state'])
    gate_holder, _ = add_adapters(model, ck['config']['rank'])
    load_adapter_state(model, ck['adapter'])
    model.eval()
    npm = Model(base_npz, adapter=out)
    designs, pairs = load(data)
    test = [p for p in pairs if designs[p['src']]['split'] == 'test'
            and max(designs[p['tgt']]['W'] / designs[p['tgt']]['H'], designs[p['tgt']]['H'] / designs[p['tgt']]['W']) >= 4][:5]
    agree = total = 0
    for p in test:
        ex = pair_example(designs, p, False)
        tW, tH = ex[3]
        b = tensorize([ex])
        gate_holder.value = torch.tensor([strip_gate(tW, tH)])
        with torch.no_grad():
            mem, mm = model.encode(b['types'], b['feats'], b['glob'], b['mask'])
            n = b['types'].shape[1]
            bins = torch.zeros(1, n, 5, dtype=torch.long)
            for j in range(n):
                prev = torch.cat([bins[:, :1], bins[:, :-1]], 1)
                z = model.decode_hidden(mem, mm, prev)[:, j]
                for f in range(5):
                    bins[:, j, f] = model.field_logits(z, f).argmax(-1)
                    if f < 4:
                        z = z + model.in_emb[f](bins[:, j, f])
        nb = npm.sample(b['types'][0].numpy(), b['feats'][0].numpy(), b['glob'][0].numpy(), k=1, gate=strip_gate(tW, tH))[0]
        agree += int((np.abs(nb - bins[0].numpy()) <= 1).sum()); total += nb.size
    print(json.dumps({'out': out, 'parity_within_1_bin': round(agree / max(total, 1), 4), 'config': cfg}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='work/retarget-data')
    ap.add_argument('--base', default='work/retarget-runs/v2/best.pt')
    ap.add_argument('--synthetic', nargs='+', default=['work/postproc/strips/synthetic.jsonl'])
    ap.add_argument('--out', default='work/retarget-runs/strip-a1')
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--kl', type=float, default=0.1)
    ap.add_argument('--syn-frac', type=float, default=0.5)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--export', nargs=2, metavar=('CKPT', 'OUT'))
    ap.add_argument('--base-npz', default='engine/models/retarget.npz')
    a = ap.parse_args()
    if a.export:
        export(a.export[0], a.export[1], a.base_npz, a.data)
    else:
        train(a)


if __name__ == '__main__':
    main()
