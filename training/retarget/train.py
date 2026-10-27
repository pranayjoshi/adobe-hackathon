"""Train the retargeting transformer on Crello (noisy single-design tasks + designer resize pairs).

    .venv/bin/python -m training.retarget.train --data work/retarget-data --out work/retarget-runs/v1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.retarget.data import Sampler, load, pair_example, tensorize
from training.retarget.model import Retargeter, param_count
from engine.retarget.features import N_BINS


def soft_targets(bins, sigma):
    k = torch.arange(N_BINS, device=bins.device, dtype=torch.float32)
    d = (k[None, None] - bins[..., None].float()) ** 2
    p = torch.exp(-d / (2 * sigma ** 2))
    return p / p.sum(-1, keepdim=True)


def loss_fn(model, batch, device, sigma=1.5):
    b = {k: v.to(device) for k, v in batch.items()}
    logits = model(b['types'], b['feats'], b['glob'], b['mask'], b['bins'])
    target = soft_targets(b['bins'], sigma)
    lm = b['loss_mask']
    nll = -(target * F.log_softmax(logits.float(), -1)).sum(-1)
    return (nll * lm).sum() / lm.sum().clamp(min=1)


def evaluate(model, designs, pairs, device, split='val', limit=600):
    model.eval()
    ps = [p for p in pairs if designs[p['src']]['split'] == split]
    random.Random(0).shuffle(ps)
    ps = ps[:limit]
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ps), 64):
            batch = tensorize([pair_example(designs, p, False) for p in ps[i:i + 64]])
            l = loss_fn(model, batch, device, sigma=1.0)
            tot += l.item() * len(ps[i:i + 64]); cnt += len(ps[i:i + 64])
    model.train()
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='work/retarget-data')
    ap.add_argument('--out', default='work/retarget-runs/v1')
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--pair-frac', type=float, default=0.5)
    ap.add_argument('--arrange-frac', type=float, default=0.25)
    ap.add_argument('--finetune-steps', type=int, default=8000, help='final steps with pair_frac 0.9')
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--enc', type=int, default=4)
    ap.add_argument('--dec', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--eval-every', type=int, default=500)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    designs, pairs = load(a.data)
    train = Sampler(designs, pairs, 'train', a.pair_frac, a.arrange_frac)
    model = Retargeter(a.d, 8, a.enc, a.dec, a.dropout).to(device)
    print(json.dumps({'params': param_count(model), 'train_pairs': len(train.pairs), 'train_singles': len(train.singles),
                      'device': device}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    best = float('inf'); log = []; t0 = time.time(); run = 0.0
    for step in range(1, a.steps + 1):
        if step == a.steps - a.finetune_steps:
            train.pair_frac, train.arrange_frac = 0.85, 0.1
        loss = loss_fn(model, train.batch(a.batch), device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            val = evaluate(model, designs, pairs, device)
            rec = {'step': step, 'train': round(run, 4), 'val_pair_nll': round(val, 4), 'lr': sched.get_last_lr()[0],
                   'min': round((time.time() - t0) / 60, 1)}
            log.append(rec); print(json.dumps(rec), flush=True)
            state = {'config': model.config, 'state': model.state_dict(), 'step': step, 'val': val}
            torch.save(state, out / 'last.pt')
            if val < best:
                best = val
                torch.save(state, out / 'best.pt')
    (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
