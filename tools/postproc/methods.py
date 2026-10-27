"""Post-processing methods under evaluation. Each: (source, target, A, recs) -> (design, info)."""


def current(src, target, A, recs):
    """Today's committed post-processing: greedy sample, materialize + deoverlap."""
    from engine.retarget.propose import propose
    d = dict(propose(src, target, A, k=1))
    return d['0'], {}


_PRIORS = None


def _priors():
    global _PRIORS
    if _PRIORS is None:
        import json
        from pathlib import Path
        _PRIORS = json.load(open(Path(__file__).resolve().parents[1] / 'retarget-scale/designer_priors.json'))
    return _PRIORS


def solver(src, target, A, recs, weights=None, time_limit=0.3):
    """Constrained decoding of the model's greedy distribution (P2/P3 solver)."""
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    W, H = float(src['canvas']['width']), float(src['canvas']['height'])
    boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1)
    P = Problem(src, target, A, recs, _priors())
    sol = solve(P, boxes[0], probs[0], weights=weights, time_limit=time_limit)
    if sol is None:
        d, info = current(src, target, A, recs)
        return d, {'fallback': True}
    return sol.design, {'status': sol.status, 'parts': sol.parts, 'notes': sol.notes}


def solver_gated(src, target, A, recs, weights=None, time_limit=2.0, workers=8):
    """Solver with today's pipeline as warm start and as the fallback: whichever passes the engine
    wins, the solver's layout when both do."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import eligible
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.propose import propose
    W, H = float(src['canvas']['width']), float(src['canvas']['height'])
    cur = dict(propose(src, target, A, k=1))['0']
    boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1)
    P = Problem(src, target, A, recs, _priors())
    sol = solve(P, boxes[0], probs[0], weights=weights, time_limit=time_limit, hint=cur, workers=workers)
    if sol is None or sol.design is None:
        return cur, {'pick': 'current', 'why': 'solver_none'}
    s_ok = eligible(src, target, sol.design, A)[0]
    if s_ok:
        return sol.design, {'pick': 'solver', 'notes': sol.notes, 'parts': sol.parts}
    c_ok = eligible(src, target, cur, A)[0]
    return (cur, {'pick': 'current', 'why': 'solver_ineligible' if c_ok else 'both_ineligible', 'notes': sol.notes})


def solver_gated_w1(src, target, A, recs):
    """solver_gated for parallel evaluation: one solver thread per process."""
    return solver_gated(src, target, A, recs, time_limit=1.5, workers=1)


STRIPS = {'800x90': (800, 90), '970x90': (970, 90), '728x90': (728, 90), '90x800': (90, 800),
          '120x600': (120, 600), '160x600': (160, 600), '300x1050': (300, 1050)}


def solver_strip(src, target, A, recs, weights=None, time_limit=2.0, workers=8):
    """Gated solver; extreme strips start from the reading-order composer instead of today's layout."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import eligible
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.propose import propose
    from engine.retarget.strip import compose
    W, H = float(src['canvas']['width']), float(src['canvas']['height'])
    cur = dict(propose(src, target, A, k=1))['0']
    boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1)
    P = Problem(src, target, A, recs, _priors())
    hint = compose(P) if P.strip else cur
    sol = solve(P, boxes[0], probs[0], weights=weights, time_limit=time_limit, hint=hint, workers=workers)
    info = {'strip': P.strip}
    if sol is not None and sol.design is not None and eligible(src, target, sol.design, A)[0]:
        return sol.design, dict(info, pick='solver', notes=sol.notes)
    if P.strip and eligible(src, target, hint, A)[0]:
        return hint, dict(info, pick='composer')
    if eligible(src, target, cur, A)[0]:
        return cur, dict(info, pick='current')
    return (sol.design if sol is not None and sol.design is not None else cur), dict(info, pick='none_eligible')


def solver_strip_w1(src, target, A, recs):
    return solver_strip(src, target, A, recs, time_limit=1.5, workers=1)


def composer_only(src, target, A, recs):
    from engine.retarget.solver import Problem
    from engine.retarget.strip import compose
    return compose(Problem(src, target, A, recs, _priors())), {}


PROFILES = {'balanced': {}, 'bold': {'headline': 0.6, 'margin': 0.25, 'balance': 0.1}}


def candidates_all(src, target, A, recs, time_limit=1.5, workers=8):
    """Every candidate the new post-processing considers, as (tag, design)."""
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.propose import propose
    from engine.retarget.strip import compose
    W, H = float(src['canvas']['width']), float(src['canvas']['height'])
    today = dict(propose(src, target, A, k=1))
    out = [('current', today['0'])] + ([('bound', today['0_bound'])] if '0_bound' in today else [])
    boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1)
    P = Problem(src, target, A, recs, _priors())
    hint = today['0']
    if P.strip:
        hint = compose(P)
        out.append(('composer', hint))
    for name, wts in PROFILES.items():
        sol = solve(P, boxes[0], probs[0], weights=wts, time_limit=time_limit, hint=hint, workers=workers)
        if sol is not None and sol.design is not None:
            out.append(('solver_' + name, sol.design))
    return out


def final(src, target, A, recs, time_limit=1.5, workers=8):
    """Appeal-ranked selection among valid candidates (plan section 3.5)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.retarget import appeal as AP
    si = AP.source_info(src, A)
    scored = []
    for tag, d in candidates_all(src, target, A, recs, time_limit, workers):
        ok = AP.eligible(src, target, d, A)[0]
        scored.append((ok, AP.score(src, d, A, si), tag, d))
    ok = [s for s in scored if s[0]]
    best = max(ok or scored, key=lambda s: s[1])
    return best[3], {'pick': best[2], 'eligible': bool(ok), 'scores': {s[2]: [round(s[1], 3), s[0]] for s in scored}}


def final_w1(src, target, A, recs):
    return final(src, target, A, recs, time_limit=1.2, workers=1)
