"""Gate summary for harness runs: python work/postproc/summarize.py runs/a.jsonl [runs/b.jsonl ...]"""
import collections
import json
import statistics as S
import sys


def load(path):
    return [json.loads(l) for l in open(path)]


def summary(rows):
    ok = [r for r in rows if 'error' not in r and not r.get('failed')]
    n = len(ok)
    out = {'n': n, 'errors': len(rows) - n}
    if not n:
        return out
    out['eligible'] = round(sum(r['eligible'] for r in ok) / n, 4)
    m = [r['m'] for r in ok if r.get('m')]
    if m:
        for k in ('iou', 'font_err', 'size_err', 'center_err', 'order_agree'):
            v = [x[k] for x in m if x.get(k) is not None]
            if v:
                out[k] = round(S.mean(v), 4)
    c = [r['chk'] for r in ok]
    def rate(f):
        return round(sum(1 for x in c if f(x)) / n, 4)
    out['align_loss'] = rate(lambda x: x.get('align_pairs', 0) >= 2 and x.get('align_kept', 0) / max(x['align_pairs'], 1) < 0.5)
    out['coverage_med'] = round(S.median(x.get('coverage', 0) for x in c), 4)
    out['overflow'] = rate(lambda x: bool(x.get('text_overflow')))
    out['below_floor'] = rate(lambda x: bool(x.get('text_below_floor')))
    out['collision'] = rate(lambda x: x.get('coll_n', 0) > 0)
    out['occluded'] = rate(lambda x: bool(x.get('occluded')))
    out['unsupported'] = rate(lambda x: bool(x.get('unsupported_text')))
    out['contrast'] = rate(lambda x: bool(x.get('contrast_regress')))
    out['hidden'] = rate(lambda x: bool(x.get('hidden')))
    out['hier_flip'] = rate(lambda x: x.get('hier_flips', 0) > 0)
    out['content_share_ratio_med'] = round(S.median(x['content_share_ratio'] for x in c), 4)
    if any('appeal' in r for r in ok):
        out['appeal'] = round(S.mean(r['appeal'] for r in ok if 'appeal' in r), 4)
    out['t_med'] = round(S.median(r['t'] for r in ok), 3)
    out['veto'] = collections.Counter(v.split(':')[0] + (':' + v.split(':')[1][:24] if ':' in v else '') for r in ok for v in r['veto']).most_common(8)
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, json.dumps(summary(load(p)), indent=None))
