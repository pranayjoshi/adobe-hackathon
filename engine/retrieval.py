"""Geometry-only training library with optimal role/shape slot assignment.

CHAMELEON_RETRIEVAL_VERSION=legacy (default) keeps the original aspect-ratio
lookup over engine/models/layouts.json. `fit_aware` (plan 05) uses the
versioned layouts.v2.json: structural pre-filter -> block-to-slot assignment
with hard-forbidden impossible semantics and complete source coverage ->
measured materialization -> rerank by actual text fit, recording why each
retrieved arrangement was rejected. An invalid or incompatible library raises
LibraryError (a ValueError), which the pipeline logs as "invalid library
ignored" while the templates carry on.
"""
import copy,json,math,os,time
from pathlib import Path
from .canonical import ROOT, backdrop_patch
from .quality import units
from .repair import contain
from .typography import fit_text
LIBRARY=ROOT/'engine/models/layouts.json'
LIBRARY_V2=ROOT/'engine/models/layouts.v2.json'
VERSIONS=('legacy','fit_aware')


def retrieval_version(value=None):
    """Explicit keyword first, then CHAMELEON_RETRIEVAL_VERSION; unknown -> legacy."""
    v=value if value is not None else os.environ.get('CHAMELEON_RETRIEVAL_VERSION','legacy')
    return v if v in VERSIONS else 'legacy'


def retrieve(source,target,A,*,version=None):
    return retrieve_with_report(source,target,A,version=version)[0]


def retrieve_with_report(source,target,A,*,version=None):
    """(designs, report). The report lists the pool, selections and rejection reasons."""
    if retrieval_version(version)=='legacy':
        return _retrieve_legacy(source,target,A),{'version':'legacy'}
    from .providers.retrieval_fit import retrieve_fit_aware
    return retrieve_fit_aware(source,target,A)


def _retrieve_legacy(source,target,A):
    path=Path(os.environ.get('CHAMELEON_LAYOUT_LIBRARY',str(LIBRARY)))
    if not path.is_file():return []
    library=json.loads(path.read_text());out=[];blocks=A.all_blocks()
    for item in sorted(library,key=lambda x:abs(math.log((target['width']/target['height'])/x['ratio'])))[:6]:
        slots=item['slots']
        if not slots or not blocks:continue
        def cost(b,s):return (0 if b.role==s['role'] else 2)+abs(math.log(max(.001,b.src_box.w/b.src_box.h)/max(.001,s['w']/s['h'])))
        try:
            from scipy.optimize import linear_sum_assignment
            rows,cols=linear_sum_assignment([[cost(b,s) for s in slots] for b in blocks])
        except ImportError:continue
        if len(rows)!=len(blocks):continue # no unmatched content discarded
        design=copy.deepcopy(source);design['canvas']=dict(target);by={e['id']:e for e in design['elements']}
        # Background layers follow the target; other non-content layers are uniformly contained.
        sx,sy=target['width']/A.sw,target['height']/A.sh
        bg={e['id']:e for e in A.background}
        for e in design['elements']:
            e.update(x=e['x']*sx,y=e['y']*sy,width=e['width']*min(sx,sy),height=e['height']*min(sx,sy))
            if e['id'] in bg:e.update(backdrop_patch(bg[e['id']],target['width'],target['height']))
        for i,j in zip(rows,cols):
            b,s=blocks[i],slots[j];r=b.src_box;w,h=s['w']*target['width'],s['h']*target['height'];k=min(w/max(1,r.w),h/max(1,r.h))
            for original in b.members:
                e=by[original['id']];e.update(x=s['x']*target['width']+(original['x']-r.x)*k,y=s['y']*target['height']+(original['y']-r.y)*k,width=original['width']*k,height=original['height']*k)
                if e['type']=='text':e['font_size']=fit_text(e,e['width'],e['height'],max(1,round(original['font_size']*k))).font_size
        out.append(contain(design,A))
    return out

def export_layout(source,A):
    return {'ratio':A.sw/A.sh,'slots':[{'role':b.role,'x':b.src_box.x/A.sw,'y':b.src_box.y/A.sh,'w':b.src_box.w/A.sw,'h':b.src_box.h/A.sh} for b in A.all_blocks()]}
