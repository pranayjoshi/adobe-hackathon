"""Paired descriptive reports. AI rubric deltas are not human preference wins."""
import argparse,json,statistics
from pathlib import Path
from evaluation.visual_suite import atomic_json

def compare(baseline, challenger):
    baseline,challenger=Path(baseline),Path(challenger)
    a=json.loads((baseline/'results.json').read_text());b=json.loads((challenger/'results.json').read_text())
    am=json.loads((baseline/'manifest.json').read_text());bm=json.loads((challenger/'manifest.json').read_text())
    for key in ('suite_sha256','resources'):
        if am[key]!=bm[key]:raise ValueError('incomparable frozen inputs/resources: '+key)
    aa={r['case']:r for r in a['cases']};bb={r['case']:r for r in b['cases']}
    if set(aa)!=set(bb):raise ValueError('paired case sets differ')
    rows=[]
    for key,old in aa.items():
        new=bb[key];both=old.get('score') is not None and new.get('score') is not None
        rows.append({'case':key,'family':old['family'],'baseline_status':old['status'],'challenger_status':new['status'],
                     'baseline_valid':bool(old.get('valid')),'challenger_valid':bool(new.get('valid')),'baseline_rendered':bool(old.get('rendered')),'challenger_rendered':bool(new.get('rendered')),'validity_delta':int(bool(new.get('valid')))-int(bool(old.get('valid'))),
                     'render_delta':int(bool(new.get('rendered')))-int(bool(old.get('rendered'))),
                     'rubric_delta':round(new['score']-old['score'],3) if both else None,
                     'baseline_defects':old.get('defects'), 'challenger_defects':new.get('defects'),
                     'seconds_delta':round(new.get('seconds',0)-old.get('seconds',0),3)})
    evaluation_paths={'engine/rubric.py','engine/metrics.py','engine/judging.py','engine/quality.py','engine/invariants.py'}
    def evaluator(manifest): return {r['path']:r['sha256'] for r in manifest.get('code',{}).get('files',[]) if r['path'] in evaluation_paths}
    left,right=evaluator(am),evaluator(bm)
    same_evaluator=bool(left and right and left==right)
    deltas=[r['rubric_delta'] for r in rows if r['rubric_delta'] is not None]
    result={'baseline':str(baseline),'challenger':str(challenger),'summary':{'cases':len(rows),'paired_scored':len(deltas),'failed_baseline':a['summary']['failed'],'failed_challenger':b['summary']['failed'],
                       'rubric_mean_delta':statistics.mean(deltas) if deltas else None,
                       'new_no_valid_rejections_of_invalid_baseline':sum(r['baseline_rendered'] and not r['baseline_valid'] and not r['challenger_rendered'] for r in rows),'new_validity_failures':sum(r['validity_delta']<0 for r in rows), 'new_render_failures':sum(r['render_delta']<0 for r in rows),
                       'identical_evaluator':same_evaluator,'rubric_interpretation':'paired internal rubric' if same_evaluator else 'evaluator changed or unidentified; raw score deltas do not establish output improvement','human_win_tie_loss':None,'generalization_evidence':False,'note':'Fixed four-family development panel; failed slots remain in denominator.'},'cases':rows}
    atomic_json(challenger/'comparison.json',result)
    atomic_json(challenger/'comparisons'/('summary-vs-'+baseline.name+'.json'),result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('baseline',type=Path);p.add_argument('challenger',type=Path);a=p.parse_args();print(json.dumps(compare(a.baseline,a.challenger),indent=2))
