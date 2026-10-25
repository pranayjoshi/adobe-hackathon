"""Source-relative quality gates, separate from structural preservation."""
import math
from .blocks import FLOORS

def candidate_assessment(result, verification=None):
    """Versioned assessment sidecar; preferences cannot replace hard gates."""
    rubric=result.get('rubric') or {}
    verification=verification or result.get('verification') or {'verified':False,'status':'unverified'}
    structural=bool(result.get('valid',False) and rubric.get('gate',False))
    tier=rubric.get('tier2') or {}
    return {'version':'assessment-v1','structural_gate':structural,
            'blocking_defects':tier.get('defects',[]),'measured_dimensions':tier.get('scores',{}),
            'verification':{k:v for k,v in verification.items() if k!='image'},
            'eligible':structural and bool(verification.get('verified')),
            'uncertainty':{'preference_calibrated':False}}

def judge_quality(source,design,A,diagnostics):
    original={e['id']:e for e in source['elements']};area=math.sqrt(design['canvas']['width']*design['canvas']['height']/(A.sw*A.sh))
    unreadable=[]
    for e in design['elements']:
        if e['type']!='text' or not e.get('content','').strip():continue
        role=A.roles.get(e['id'],'body');src=original[e['id']]
        floor=min(FLOORS.get(role,9),src['font_size']*max(.5,min(1,area)))
        if e['font_size']<floor*.98:unreadable.append(e['id'])
    defects={k:list(diagnostics[k]) for k in ('overflow','collisions','contrast_regressions')}
    # Crello text boxes can already be shorter than the rendered line box.
    # Penalize new/worsened overflow, rather than punishing a preserved source.
    from .typography import measure
    by={e['id']:e for e in design['elements']}
    worsened=[]
    for eid in defects['overflow']:
        e,src=by[eid],original[eid]
        before=measure(src,int(src['font_size']),max(1,src['width']))
        after=measure(e,int(e['font_size']),max(1,e['width']))
        if after.height/max(1,e['height'])>max(1,before.height/max(1,src['height']))+.02 or after.line_width/max(1,e['width'])>max(1,before.line_width/max(1,src['width']))+.02:worsened.append(eid)
    defects['overflow']=worsened
    defects['unreadable']=unreadable
    from .scene import scene_defects
    defects.update(scene_defects(source,design))
    count=sum(len(v) for v in defects.values())
    return {'passed':count==0,'defect_count':count,'defects':defects}
