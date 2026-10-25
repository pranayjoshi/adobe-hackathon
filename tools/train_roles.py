"""Train portable role trees from JSONL {source, roles:{element_id:role}, family}.

Input labels must be reviewed or explicitly identified as weak. No source IDs are features.
"""
import argparse,json
from pathlib import Path
from engine.learning import features,FEATURE_NAMES

def train(path,out):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import accuracy_score,classification_report
    xs=[];ys=[];groups=[]
    for line in path.read_text().splitlines():
        row=json.loads(line);f=features(row['source'])
        for eid,role in row['roles'].items():
            if eid in f:xs.append(f[eid]);ys.append(role);groups.append(row['family'])
    if len(set(groups))<5:raise ValueError('At least five independently labeled source families required')
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=7).split(xs,ys,groups))
    model=RandomForestClassifier(n_estimators=48,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=7,n_jobs=1)
    x=np.array(xs);y=np.array(ys);model.fit(x[tr],y[tr]);pred=model.predict(x[va]);report={'held_out_accuracy':accuracy_score(y[va],pred),'classes':classification_report(y[va],pred,output_dict=True,zero_division=0),'training_families':len(set(groups[i] for i in tr)),'validation_families':len(set(groups[i] for i in va))}
    trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'values':t.value[:,0,:].tolist()})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'features':FEATURE_NAMES,'classes':model.classes_.tolist(),'trees':trees,'report':report}));print(json.dumps(report,indent=2))



# ---------------------------------------------------------------------------
# v2: calibrated, family-disjoint, versioned artifact (engine/models/roles.v2.json)
#
#   .venv/bin/python -m tools.train_roles_v2 is not a module; run:
#   .venv/bin/python tools/train_roles.py LABELS --v2 --output engine/models/roles.v2.json --report OUT.json
#
# Splits by source family into train / calibration / test (60/20/20). Compares
# Random Forest and gradient boosting on the calibration split, fits a softmax
# temperature on calibration, then picks class-specific override thresholds
# that reach the target precision on calibration. Test metrics are reported as
# weak-label agreement (labels are machine-distilled, not designer judgment).
# ---------------------------------------------------------------------------
SUITE_SCHEMA = Path(__file__).resolve().parents[1] / 'training/role_schema.json'


def _family_key(source):
    import hashlib, re
    textkey = '|'.join(sorted(re.sub(r'\s+', ' ', str(e.get('content', ''))).strip().lower() for e in source['elements'] if e.get('content')))
    return hashlib.sha256(textkey.encode()).hexdigest()[:20] if textkey else None


def _suite_exclusions(root):
    schema = json.loads(SUITE_SCHEMA.read_text())
    ex = schema['excluded_from_training']
    keys = set()
    for cid in ex['catalog_ids']:
        p = root / 'output/crello/designs' / f'{cid}.json'
        if p.exists():
            keys.add(_family_key(json.loads(p.read_text())['source']))
    for rel in ex['local_inputs']:
        p = root / rel
        if p.exists():
            keys.add(_family_key(json.loads(p.read_text())['source']))
    return set(ex['source_ids']), {k for k in keys if k}


def _tree_json(t, gb=False):
    vals = t.value[:, 0, :].tolist() if not gb else [[v] for v in t.value[:, 0, 0].tolist()]
    return {'left': t.children_left.tolist(), 'right': t.children_right.tolist(), 'feature': t.feature.tolist(),
            'threshold': t.threshold.tolist(), 'values': vals}


def _ece(probs, y, classes, bins=10):
    import numpy as np
    conf = probs.max(1); pred = np.array(classes)[probs.argmax(1)]; correct = (pred == y).astype(float)
    e = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def train_v2(path, out, report_path=None, target_precision=0.8, min_support=30, seed=7):
    import hashlib
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from engine.learning import (FEATURE_NAMES_V2, FEATURE_SCHEMA_HASH, ROLE_ARTIFACT_VERSION, TEXT_ROLES, IMAGE_ROLES,
                                 _sha, features_v2)
    root = Path(__file__).resolve().parents[1]
    ex_ids, ex_fams = _suite_exclusions(root)
    ontology = json.loads(SUITE_SCHEMA.read_text())
    xs, ys, groups, excluded = [], [], [], 0
    usable = set(TEXT_ROLES) | set(IMAGE_ROLES)
    raw = path.read_bytes()
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    from training.families import build_split_manifest,resolve_family
    family_manifest=build_split_manifest([(resolve_family(row),row) for row in rows])
    def canonical_family(row):
        family=resolve_family(row)
        while family in family_manifest['aliases']:family=family_manifest['aliases'][family]
        return family
    for row in rows:
        if row.get('source_id') in ex_ids or _family_key(row['source']) in ex_fams:
            excluded += 1
            continue
        f = features_v2(row['source'])
        types = {e['id']: e['type'] for e in row['source']['elements']}
        for eid, role in row['roles'].items():
            # Only roles the runtime may override; type-consistent labels only.
            if eid in f and role in usable and ((types[eid] == 'text') == (role in TEXT_ROLES)):
                xs.append(f[eid]); ys.append(role); groups.append(canonical_family(row))
    if len(set(groups))<5 or len(set(ys))<2:
        raise ValueError('At least five independent source families and two roles are required')
    X, Y, G = np.array(xs, dtype=float), np.array(ys), np.array(groups)
    assert np.isfinite(X).all(), 'non-finite features'
    a, rest = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed).split(X, Y, G))
    b, c = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(X[rest], Y[rest], G[rest]))
    tr, ca, te = a, rest[b], rest[c]
    assert not (set(G[tr]) & set(G[ca]) or set(G[tr]) & set(G[te]) or set(G[ca]) & set(G[te])), 'family leak'
    if not (set(Y[ca])|set(Y[te]))<=set(Y[tr]):
        raise ValueError('Calibration/test contain roles absent from training; collect more independent families')
    cands = {
        'rf': RandomForestClassifier(n_estimators=64, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=1),
        'gb': GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=.1, subsample=.8, random_state=seed),
    }
    if len(set(Y[tr]))<3:
        cands.pop('gb')  # binary sklearn logits have a different export shape
    from sklearn.utils.class_weight import compute_sample_weight
    scores = {}
    for name, model in cands.items():
        if name == 'gb':
            model.fit(X[tr], Y[tr], sample_weight=compute_sample_weight('balanced', Y[tr]))
        else:
            model.fit(X[tr], Y[tr])
        scores[name] = f1_score(Y[ca], model.predict(X[ca]), average='macro')
    best = max(scores, key=scores.get); model = cands[best]; classes = model.classes_.tolist()

    def logits(Xs):
        if best == 'rf':
            return np.log(np.clip(model.predict_proba(Xs), 1e-9, 1))
        return model.decision_function(Xs) if len(classes) > 2 else None

    def soft(z, T):
        z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    Lc = logits(X[ca]); yc = np.array([classes.index(v) for v in Y[ca]])
    Ts = np.exp(np.linspace(np.log(.25), np.log(8), 80))
    nll = [-np.log(soft(Lc, T)[np.arange(len(yc)), yc] + 1e-12).mean() for T in Ts]
    T = float(Ts[int(np.argmin(nll))])
    Pc = soft(Lc, T); pred_c = Pc.argmax(1)
    thresholds = {}
    for k, cls in enumerate(classes):
        sel = pred_c == k
        chosen = None
        for t in np.linspace(.4, .99, 60):
            m = sel & (Pc[:, k] >= t)
            n_, p_ = int(m.sum()), float((yc[m] == k).mean()) if m.any() else 0.0
            z = 1.645  # one-sided 95% Wilson lower bound: a small lucky slice must not unlock overrides
            lower = (p_ + z * z / (2 * n_) - z * ((p_ * (1 - p_) + z * z / (4 * n_)) / n_) ** .5) / (1 + z * z / n_) if n_ else 0.0
            if n_ >= min_support and lower >= target_precision:
                chosen = round(float(t), 3); break
        if chosen is not None:
            thresholds[cls] = chosen
    Lt = logits(X[te]); Pt = soft(Lt, T); raw_t = soft(Lt, 1.0); pred_t = np.array(classes)[Pt.argmax(1)]
    yt_idx = np.array([classes.index(v) for v in Y[te]])
    override = {}
    for cls, t in thresholds.items():
        k = classes.index(cls); m = (Pt.argmax(1) == k) & (Pt[:, k] >= t)
        override[cls] = {'threshold': t, 'fires': int(m.sum()), 'precision': float((yt_idx[m] == k).mean()) if m.any() else None}
    report = {
        'label_kind': 'weak (machine-distilled DesignAsCode themes mapped by keyword); agreement is not designer judgment',
        'excluded_suite_rows': excluded, 'rows': len(Y),
        'family_manifest_hash':_sha(family_manifest),
        'split_policy':'exact immutable/provenance union, then disjoint 60/20/20 source-family groups',
        'families': {'train': len(set(G[tr])), 'calibration': len(set(G[ca])), 'test': len(set(G[te]))},
        'candidates_macro_f1_on_calibration': scores, 'selected': best, 'temperature': T,
        'test_macro_f1': float(f1_score(Y[te], pred_t, average='macro')),
        'test_accuracy': float((pred_t == Y[te]).mean()),
        'test_ece_uncalibrated': _ece(raw_t, Y[te], classes), 'test_ece_calibrated': _ece(Pt, Y[te], classes),
        'test_per_role': classification_report(Y[te], pred_t, output_dict=True, zero_division=0),
        'override_thresholds_on_test': override, 'target_precision': target_precision,
    }
    if best == 'rf':
        trees = [_tree_json(e.tree_) for e in model.estimators_]; extra = {}
    else:
        trees = [[_tree_json(e.tree_, gb=True) for e in stage] for stage in model.estimators_]
        init = model._raw_predict_init(X[:1])[0].tolist()
        extra = {'init': init, 'learning_rate': model.learning_rate}
    calibration = {'method': 'temperature', 'temperature': T, 'thresholds': thresholds, 'target_precision': target_precision,
                   'split': 'calibration families disjoint from train/test'}
    art = {'version': ROLE_ARTIFACT_VERSION, 'model_type': best, 'classes': classes, 'features': FEATURE_NAMES_V2,
           'feature_schema_hash': FEATURE_SCHEMA_HASH, 'ontology': ontology['roles'], 'ontology_hash': _sha(ontology['roles']),
           'calibration': calibration, 'calibration_hash': _sha(calibration), 'trees': trees, **extra,
           'training': {'labels_sha256': hashlib.sha256(raw).hexdigest(), 'seed': seed, 'report': {k: v for k, v in report.items() if k != 'test_per_role'}}}
    # Portable-export self-check: runtime evaluation must match sklearn.
    from engine.learning import predict_proba
    chk = np.array([predict_proba(art, list(x)) for x in X[te][:200]])
    assert np.allclose(chk, Pt[:200], atol=1e-6), f'portable export mismatch ({best}): {np.abs(chk-Pt[:200]).max()}'
    art['artifact_hash'] = _sha({k: v for k, v in art.items() if k != 'training'})[:16]
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(art))
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'test_per_role'}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('labels', type=Path); p.add_argument('--output', type=Path, default=None)
    p.add_argument('--v2', action='store_true'); p.add_argument('--report', default=None); a = p.parse_args()
    if a.v2:
        train_v2(a.labels, a.output or Path('engine/models/roles.v2.json'), a.report)
    else:
        train(a.labels, a.output or Path('engine/models/roles.json'))
