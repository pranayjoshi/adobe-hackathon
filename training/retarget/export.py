"""Export a trained checkpoint to the numpy weights file the engine loads, and check parity.

    .venv/bin/python -m training.retarget.export work/retarget-runs/v1/best.pt engine/models/retarget.npz
"""
import argparse
import json

import numpy as np
import torch

from engine.retarget.infer import Model
from training.retarget.data import Sampler, load, pair_example, tensorize
from training.retarget.model import Retargeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--data', default='work/retarget-data')
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu')
    arrays = {k: v.numpy().astype(np.float16 if v.ndim > 1 else np.float32) for k, v in ck['state'].items()}
    cfg = dict(ck['config'], step=ck['step'], val=ck['val'])
    arrays['__config__'] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    np.savez_compressed(a.out, **arrays)
    # Parity: teacher-forced logits from torch vs greedy decode in numpy on a few held-out pairs.
    model = Retargeter(**{k: cfg[k] for k in ('d', 'heads', 'enc_layers', 'dec_layers')})
    model.load_state_dict(ck['state']); model.eval()
    npm = Model(a.out)
    designs, pairs = load(a.data)
    test = [p for p in pairs if designs[p['src']]['split'] == 'test'][:5]
    agree = total = 0
    for p in test:
        ex = pair_example(designs, p, False)
        b = tensorize([ex])
        with torch.no_grad():
            mem, mm = model.encode(b['types'], b['feats'], b['glob'], b['mask'])
            # Torch greedy decode for comparison.
            n = b['types'].shape[1]
            bins = torch.zeros(1, n, 5, dtype=torch.long)
            for j in range(n):
                prev = torch.cat([bins[:, :1], bins[:, :-1]], 1)
                z = model.decode_hidden(mem, mm, prev)[:, j]
                for f in range(5):
                    bins[:, j, f] = model.field_logits(z, f).argmax(-1)
                    if f < 4:
                        z = z + model.in_emb[f](bins[:, j, f])
        nb = npm.sample(b['types'][0].numpy(), b['feats'][0].numpy(), b['glob'][0].numpy(), k=1)[0]
        agree += int((np.abs(nb - bins[0].numpy()) <= 1).sum()); total += nb.size
    print(json.dumps({'out': a.out, 'parity_within_1_bin': round(agree / total, 4), 'config': cfg}))


if __name__ == '__main__':
    main()
