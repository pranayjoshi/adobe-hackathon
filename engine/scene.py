"""Photographic text-support relationships, independent of rigid grouping."""
from .canonical import element_box,rotated_extent
from .structure import flat_asset
from functools import lru_cache
import json


@lru_cache(maxsize=1024)
def low_detail_asset(asset_id):
    """Gradients and soft textures: almost no strong edges at thumbnail scale.

    Measured on the brand kit and Crello backgrounds: the Solstice gradient
    with bokeh dots scores ~0.044, paper/marble/watercolour textures 0.01-0.06,
    while photographs with a subject start around 0.09 (a dark gym photo) and
    are mostly 0.12-0.45. Those textures have no subject that copy was placed
    to avoid, so they have no quiet region to preserve.
    """
    from PIL import Image,ImageFilter
    from .canonical import ASSETS_DIR
    path=(ASSETS_DIR/str(asset_id or '')).resolve()
    if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():return False
    with Image.open(path) as im:
        im=im.convert('RGBA');im.thumbnail((128,128))
        edges=im.convert('L').filter(ImageFilter.FIND_EDGES).tobytes();alpha=im.getchannel('A').tobytes()
    vals=[v for v,a in zip(edges,alpha) if a>240]
    return bool(vals) and sum(v>24 for v in vals)/len(vals)<.06

def scene_defects(source,design):
    from .photo_analysis import source_photo_supports,painted_coverage
    by={e['id']:e for e in design['elements']};original={e['id']:e for e in source['elements']}
    supports=source_photo_supports(source)
    unsupported=[];displaced=[]
    for text in source['elements']:
        if text['type']!='text' or not text.get('content','').strip():continue
        box=rotated_extent(text)
        if text['id'] not in supports:continue
        host=original[supports[text['id']]];old=element_box(host);new=element_box(by[host['id']]);out=rotated_extent(by[text['id']])
        if painted_coverage(by[host['id']],out,design)<.9:unsupported.append(text['id'])
        # Displacement only means something over a photograph with subjects.
        # Over a gradient or soft texture (e.g. the Solstice background shared
        # by every sample) relocating copy loses no negative-space relationship,
        # and contrast changes are judged separately by contrast_regressions.
        if low_detail_asset(host.get('asset_id')):continue
        # The image has embedded subjects. Moving type across that photograph
        # discards its original negative-space relationship even if it fits.
        dx=abs((box.cx-old.x)/old.w-(out.cx-new.x)/max(1,new.w))
        dy=0 if by[host['id']].get('fit')=='cover' else abs((box.cy-old.y)/old.h-(out.cy-new.y)/max(1,new.h))
        if max(dx,dy)>.14:displaced.append(text['id'])
    return {'unsupported_text':unsupported,'scene_displacement':displaced}


def background_cover_allowed(source,before,after):
    """Only a canvas background may switch to the schema's standard crop fit."""
    if before.get('type')!='image' or after.get('fit')!='cover' or before.get('group_id') or before.get('rotation',0):return False
    from .canonical import Box
    canvas=Box(0,0,source['canvas']['width'],source['canvas']['height'])
    bottom=min(e.get('z_index',0) for e in source['elements'])
    return before.get('z_index',0)<=bottom+1 and element_box(before).intersection_area(canvas)>=.85*canvas.area


def propose_photo_scene(source,target,A,*,mode=None,status=None,budget=None):
    """Photo-backed proposals. `heuristic` (default) is the legacy copy column;
    `subject_aware` (CHAMELEON_PHOTO_ANALYSIS) adds support-preserving scenes
    placed on quiet, subject-free regions of the original photograph."""
    from .photo_analysis import photo_mode
    out=_legacy_photo_scene(source,target,A)
    if photo_mode(mode)!='subject_aware':return out
    try:
        extra,info=support_scenes(source,target,A,mode='subject_aware',budget=budget)
    except (ValueError,KeyError,ZeroDivisionError,OSError) as exc:
        extra,info=[],{'status':'failed:'+type(exc).__name__}
    if status is not None:status.update(info)
    return out+extra


def _legacy_photo_scene(source,target,A):
    """Reflow copy in its source horizontal quiet region over a cover photo."""
    import copy,math
    from .canonical import union_all
    from .typography import measure,tight_width
    photos=[e for e in A.background if e['type']=='image' and background_cover_allowed(source,e,{**e,'fit':'cover'}) and not flat_asset(e.get('asset_id'))]
    texts=[e for e in source['elements'] if e['type']=='text']
    if len(photos)!=1 or not texts or any(e.get('rotation',0) or e.get('group_id') for e in texts):return []
    host=photos[0];hb=element_box(host);bounds=union_all(element_box(e) for e in texts)
    if hb.intersection_area(bounds)<.95*bounds.area:return []
    # A broad mixed composition is not one copy column.
    if bounds.w/hb.w>.65:return []
    cw,ch=target['width'],target['height'];sw,sh=A.sw,A.sh
    # Map the source copy's horizontal region into the destination. Cover
    # crops on the vertical axis for banners; portrait candidates are judged
    # independently and can lose to the original-arrangement fallback.
    left=max(.04,min(.8,(bounds.x-hb.x)/hb.w-.025))*cw
    right=min(.96,max(left/cw+.12,(bounds.x1-hb.x)/hb.w+.025))*cw
    width=max(1,right-left);margin=.18*ch;gap=.025*min(cw,ch)
    ordered=sorted(texts,key=lambda e:(e['y'],e['x']))
    out=copy.deepcopy(source);out['canvas']=dict(target);by={e['id']:e for e in out['elements']};k=min(cw/sw,ch/sh)
    for original in source['elements']:
        e=by[original['id']];e.update(x=original['x']/sw*cw,y=original['y']/sh*ch,width=original['width']*k,height=original['height']*k)
        e['x']=max(0,min(cw-e['width'],e['x']));e['y']=max(0,min(ch-e['height'],e['y']))
        if original['id']!=host['id'] and original['type']=='image' and original['height']<.1*sh and original['y']<.3*sh:
            e['y']=max(0,.05*ch-e['height']/2)
    by[host['id']].update(x=0,y=0,width=cw,height=ch,fit='cover')
    lo,hi=0.,min(2,math.sqrt(cw*ch/(sw*sh)))
    for _ in range(18):
        scale=(lo+hi)/2;fits=[measure(e,max(1,int(e['font_size']*scale)),width) for e in ordered]
        if all(not f.overflow for f in fits) and sum(f.height for f in fits)+gap*(len(fits)-1)<=ch-2*margin:lo=scale
        else:hi=scale
    fits=[measure(e,max(1,int(e['font_size']*lo)),width) for e in ordered]
    total=sum(f.height for f in fits)+gap*(len(fits)-1);y=(ch-total)/2
    for e,f in zip(ordered,fits):
        w=min(width,tight_width(e,f));by[e['id']].update(x=left+(width-w)/2,y=y,width=w,height=f.height,font_size=f.font_size);y+=f.height+gap
    return [out]


# ----------------------------------------------------------------------------
# Subject-aware, support-preserving photo scenes (guide 08)
# ----------------------------------------------------------------------------
def _support_hosts(source,host):
    """Secondary panels (large non-flat images above the photo) and what sits on them."""
    from .photo_analysis import painted_coverage
    area=source['canvas']['width']*source['canvas']['height']
    panels={}
    for p in source['elements']:
        if p['type']!='image' or p['id']==host['id'] or p.get('z_index',0)<=host.get('z_index',0):continue
        if element_box(p).area<.25*area or flat_asset(p.get('asset_id')) or p.get('rotation'):continue
        pb=element_box(p)
        members=[e for e in source['elements'] if e.get('z_index',0)>p.get('z_index',0) and e['type'] in ('text','image','shape')
                 and pb.intersection_area(rotated_extent(e))>=.95*max(1e-6,rotated_extent(e).area) and element_box(e).area<.5*pb.area
                 and painted_coverage(p,rotated_extent(e),source)>=.9]
        if any(m['type']=='text' for m in members):panels[p['id']]=(p,members)
    # An element on several panels belongs to the top-most one (scene_defects rule).
    owner={}
    for pid,(p,members) in sorted(panels.items(),key=lambda t:t[1][0].get('z_index',0)):
        for m in members:owner[m['id']]=pid
    return {pid:(p,[m for m in members if owner.get(m['id'])==pid]) for pid,(p,members) in panels.items()}


def _clusters(elements,gap):
    """Union of source elements whose boxes are within `gap` of each other."""
    parent={e['id']:e['id'] for e in elements}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for n,a in enumerate(elements):
        for b in elements[n+1:]:
            x,y=rotated_extent(a),rotated_extent(b)
            dx=max(0,max(x.x,y.x)-min(x.x1,y.x1));dy=max(0,max(x.y,y.y)-min(x.y1,y.y1))
            if max(dx,dy)<=gap:parent[find(a['id'])]=find(b['id'])
    groups={}
    for e in elements:groups.setdefault(find(e['id']),[]).append(e)
    return sorted(groups.values(),key=lambda g:min(e['y'] for e in g))


def _edges(box,sw,sh,tol):
    return {'left':box.x<=tol,'right':box.x1>=sw-tol,'top':box.y<=tol,'bottom':box.y1>=sh-tol}


def _place_rigid(out,members,frame,k,x,y):
    """Scale source `members` about their source `frame` by k and put the frame at (x, y)."""
    by={e['id']:e for e in out['elements']}
    for m in members:
        e=by[m['id']]
        e.update(x=x+(m['x']-frame.x)*k,y=y+(m['y']-frame.y)*k,width=m['width']*k,height=m['height']*k)
        if m['type']=='text':e['font_size']=max(1,int(m['font_size']*k))


def _anchor(box,sw,sh,cw,ch,w,h,tol):
    ed=_edges(box,sw,sh,tol)
    x=0 if ed['left'] else cw-w if ed['right'] else box.cx/sw*cw-w/2
    y=0 if ed['top'] else ch-h if ed['bottom'] else box.cy/sh*ch-h/2
    return max(0,min(cw-w,x)),max(0,min(ch-h,y))


def _split(cluster,level):
    """Split a cluster at its `level` largest vertical gaps between horizontal bands."""
    if level<=0 or len(cluster)<2:return [cluster]
    items=sorted(cluster,key=lambda e:(rotated_extent(e).y,e['id']))
    bands=[[items[0]]];bottom=rotated_extent(items[0]).y1
    for e in items[1:]:
        b=rotated_extent(e)
        if b.y<bottom-.5:bands[-1].append(e);bottom=max(bottom,b.y1)
        else:bands.append([e]);bottom=b.y1
    if len(bands)<2:return [cluster]
    gaps=sorted(range(len(bands)-1),key=lambda i:-(min(rotated_extent(e).y for e in bands[i+1])-max(rotated_extent(e).y1 for e in bands[i])))[:level]
    parts=[[]]
    for i,band in enumerate(bands):
        parts[-1]+=band
        if i in gaps:parts.append([])
    parts=[p for p in parts if p]
    # A part with no copy (a rule, an icon) stays with the copy next to it.
    merged=[]
    for n,p in enumerate(parts):
        if any(e['type']=='text' for e in p) or len(parts)==1:merged.append(p)
        elif n+1<len(parts):parts[n+1]=p+parts[n+1]
        elif merged:merged[-1]+=p
        else:merged.append(p)
    return merged


def _flow(items,k,region_w,gap):
    """Arrange rigid clusters (scaled by k) in reading order: one column, else one row.

    Returns [(cluster, frame, x, y)] relative to (0, 0) and the content size.
    """
    from .canonical import union_all
    frames=[(c,union_all(rotated_extent(e) for e in c)) for c in items]
    col_w=max(f.w for _,f in frames)*k
    lefts=[f.x for _,f in frames];x0=min(lefts)
    out=[];y=0.
    for c,f in frames:
        out.append((c,f,(f.x-x0)*k,y));y+=f.h*k+gap
    column=(out,max((f.x-x0)*k+f.w*k for _,f in frames),y-gap)
    out=[];x=0.
    for c,f in frames:
        out.append((c,f,x,0.));x+=f.w*k+gap
    row=(out,x-gap,max(f.h*k for _,f in frames))
    return column,row


_SRC_CONTRAST={}


def _cheap_contrast(source,design,host,panels,field):
    """Conservative per-color, per-layer composited contrast estimates.

    This shares the assessment definition, including minority rich-text colors
    and styled alpha. It is not official rendered verification.
    """
    from .quality import text_contrast,composite,painted_prefix
    from .photo_analysis import asset_hash
    key=(id(source),tuple((e.get('asset_id'),asset_hash(e.get('asset_id'))) for e in source['elements'] if e['type']=='image'))
    if key not in _SRC_CONTRAST:
        _SRC_CONTRAST.clear();_SRC_CONTRAST[key]=(source,text_contrast(source)[0])
    src=_SRC_CONTRAST[key][1]
    images={t['id']:composite(painted_prefix(design,t['id']),size=128)[0]
            for t in design['elements'] if t['type']=='text' and str(t.get('content','')).strip()}
    ratios=text_contrast(design,images)[0]
    return [eid for eid,r in ratios.items() if r<3 and r<.6*src.get(eid,21)]


def support_scenes(source,target,A,*,mode='subject_aware',max_out=3,budget=None,max_configs=180):
    """Cover photo + rigid support clusters on quiet regions; (designs, status).

    * the background photo fills the canvas through the schema's cover crop;
    * copy is clustered by source proximity, so units such as a headline and
      its figure never split; clusters keep their source arrangement and scale;
    * clusters whose copy sat on a panel (e.g. a band behind the offer) flow
      inside that panel, which scales (aspect kept) to contain them;
    * clusters on the bare photo go to low-saliency regions away from the
      detected subject, near the same point of the photograph as in the source;
    * other decorations stay attached to the canvas edges they touched.
    No pixels are generated and no field beyond allowlisted geometry changes.
    """
    import copy,math
    from .edit_operators import _stop
    from .invariants import passes,validate
    if max_out<=0 or max_configs<=0:return [],{'status':'config_budget','configs':[]}
    from .canonical import Box,union_all
    from .photo_analysis import analyze_asset,canvas_field,color_luminance,fit_mapping,quiet_regions,source_intent_boxes,region_cost,text_on_subject
    from .blocks import FLOORS
    photos=[e for e in A.background if e['type']=='image' and background_cover_allowed(source,e,{**e,'fit':'cover'}) and not flat_asset(e.get('asset_id'))]
    if len(photos)!=1:return [],{'status':'not_applicable','reason':'needs exactly one photographic background'}
    host=photos[0]
    texts=[e for e in source['elements'] if e['type']=='text' and str(e.get('content','')).strip()]
    if not texts or any(abs(float(e.get('rotation') or 0))>.5 for e in source['elements'] if e['id']!=host['id']):
        return [],{'status':'not_applicable','reason':'no copy, or rotated layers'}
    maps=analyze_asset(host.get('asset_id'),mode=mode,avoid=source_intent_boxes(source,host))
    if maps is None:return [],{'status':'fallback','reason':'photo asset unreadable'}
    cw,ch=float(target['width']),float(target['height']);sw,sh=float(source['canvas']['width']),float(source['canvas']['height'])
    tol=.015*max(sw,sh);margin=max(6.,.03*min(cw,ch));gap=max(3.,.02*min(cw,ch))
    panels=_support_hosts(source,host)
    hosted={m['id']:pid for pid,(p,ms) in panels.items() for m in ms if m['type']=='text'}
    background={e['id'] for e in A.background}
    loose=[e for e in source['elements'] if e['id'] not in background and e['id'] not in panels]
    small=lambda e:e['type']=='text' or element_box(e).area<.06*sw*sh
    copy_items=[e for e in loose if small(e) and (e['type']=='text' or A.roles.get(e['id'])!='decoration' or any(m['id']==e['id'] for _,(p,ms) in panels.items() for m in ms))]
    decor=[e for e in loose if e not in copy_items]
    clusters=_clusters(copy_items,.04*max(sw,sh))
    def assign(level):
        attached={pid:[] for pid in panels};free=[]
        for c in (p for c in clusters for p in _split(c,level)):
            votes=[hosted[e['id']] for e in c if e['id'] in hosted]
            if votes:attached[max(sorted(set(votes)),key=votes.count)].append(c)
            else:free.append(c)
        return attached,free
    area=math.sqrt(cw*ch/(sw*sh))
    def unreadable(design):
        by={e['id']:e for e in design['elements']};bad=0
        for m in texts:
            role=A.roles.get(m['id'],'body');floor=min(FLOORS.get(role,9),m['font_size']*max(.5,min(1,area)))
            bad+=by[m['id']]['font_size']<floor*.98
        return bad
    hb=copy.deepcopy(host);hb.update(x=0,y=0,width=cw,height=ch,fit='cover')
    s0=fit_mapping(host,maps.original_size);s1=fit_mapping(hb,maps.original_size)
    def photo_point(px,py):
        u,v=(px-s0[2])/s0[0],(py-s0[3])/s0[1]
        return u*s1[0]+s1[2],v*s1[1]+s1[3]
    field=canvas_field(hb,maps,cw,ch)
    if field is None:return [],{'status':'fallback','reason':'photo not analysable'}
    base=min(1.,area);designs=[];log=[];seen=set();attempted=0;stopped=None
    import itertools
    for level,factor,arrangement,anchor,inset in itertools.product((0,1,2),(1.25,1.,.82,.66,.5),('column','row'),('edges','intent'),('source','top','bottom')):
                stopped=_stop(budget)
                if stopped or attempted>=max_configs:
                    stopped=stopped or 'config_budget';break
                attempted+=1
                attached,free=assign(level)
                if arrangement=='row' and not any(len(v)>1 for v in attached.values()):continue
                k=base*factor
                out=copy.deepcopy(source);out['canvas']={'width':int(cw),'height':int(ch)}
                by={e['id']:e for e in out['elements']}
                by[host['id']].update(x=0.,y=0.,width=cw,height=ch,fit='cover')
                placed=[];fail=None
                for pid,(p,members) in sorted(panels.items()):
                    pb=element_box(p);items=attached[pid]
                    if not items:continue
                    column,row=_flow(items,k,cw,gap)
                    lay,w,h=column if arrangement=='column' else row
                    frame0=union_all(rotated_extent(e) for c in items for e in c)
                    # Panel keeps the source inset of its copy, scaled to contain the flowed copy.
                    lx,ly=max(0.,frame0.x-pb.x)/pb.w,max(0.,frame0.y-pb.y)/pb.h
                    kp=max(w/max(1e-6,pb.w*(1-lx)*.98),h/max(1e-6,pb.h*(1-ly)*.98),k)
                    if pb.w*kp>cw+.5 or pb.h*kp>ch+.5:
                        kp=min(cw/pb.w,ch/pb.h)
                        if w>pb.w*kp*(1-lx)+.5 or h>pb.h*kp*(1-ly)+.5:fail='panel_too_small';break
                    pw,ph=pb.w*kp,pb.h*kp
                    if anchor=='edges':px,py=_anchor(pb,sw,sh,cw,ch,pw,ph,tol)
                    else:
                        fx,fy=photo_point(frame0.cx,frame0.cy)
                        px=max(0,min(cw-pw,fx-lx*pw-w/2));py=max(0,min(ch-ph,fy-ly*ph-h/2))
                    by[pid].update(x=px,y=py,width=pw,height=ph)
                    # Copy keeps its source offset on the panel (it may hang over
                    # the panel's edge as it did in the source), or hugs its top/bottom.
                    ox=px+lx*pw
                    oy=py+(frame0.y-pb.y)/pb.h*ph if inset=='source' else py+ly*ph if inset=='top' else py+ph-h-ly*ph*.25
                    ox=max(0.,min(ox,cw-w));oy=max(0.,min(oy,ch-h))
                    for c,f,dx,dy in lay:_place_rigid(out,c,f,k,ox+dx,oy+dy)
                    placed.append(Box(px,py,pw,ph))
                if fail:log.append({'inset':inset,'level':level,'factor':factor,'arrangement':arrangement,'anchor':anchor,'status':fail});continue
                panel_boxes=list(placed);content=[]
                for c in sorted(free,key=lambda c:-union_all(rotated_extent(e) for e in c).area):
                    frame=union_all(rotated_extent(e) for e in c)
                    kc=min(k,(ch-2*margin)/frame.h,(cw-2*margin)/frame.w);w,h=frame.w*kc,frame.h*kc
                    fx,fy=photo_point(frame.cx,frame.cy)
                    lead=max((e for e in c if e['type']=='text'),key=lambda e:e['font_size'],default=None)
                    lum=color_luminance(lead.get('color')) if lead else None
                    pad=[Box(b.x-gap,b.y-gap,b.w+2*gap,b.h+2*gap) for b in content]
                    # Panels are backdrops: free copy may sit over them but not over their copy.
                    pad+=[Box(b.x-gap,b.y-gap,b.w+2*gap,b.h+2*gap) for b in (union_all(rotated_extent(by[e['id']]) for e in cc) for pid in attached for cc in attached[pid])]
                    regions=quiet_regions(field,w,h,text_lum=lum,within=Box(margin,margin,cw-2*margin,ch-2*margin),avoid=pad,prefer=(fx,fy),max_results=1)
                    if regions:q=regions[0].box
                    else:
                        q=Box(max(0,min(cw-w,fx-w/2)),max(0,min(ch-h,fy-h/2)),w,h)
                        if any(b.intersection_area(q)>0 for b in pad):fail='no_quiet_region';break
                    _place_rigid(out,c,frame,kc,q.x,q.y);content.append(q)
                if fail:log.append({'inset':inset,'level':level,'factor':factor,'arrangement':arrangement,'anchor':anchor,'status':fail});continue
                for d in decor:
                    db=element_box(d);kd=min(k,cw/db.w,ch/db.h)
                    w,h=db.w*kd,db.h*kd;x,y=_anchor(db,sw,sh,cw,ch,w,h,tol)
                    by[d['id']].update(x=x,y=y,width=w,height=h)
                for e in out['elements']:
                    # Never shrink a layer to an invisible sliver: grow it uniformly about its centre.
                    if min(e['width'],e['height'])<1.2:
                        g=1.2/max(1e-6,min(e['width'],e['height']));cx,cy=e['x']+e['width']/2,e['y']+e['height']/2
                        e['width']*=g;e['height']*=g;e['x']=max(0,min(cw-e['width'],cx-e['width']/2));e['y']=max(0,min(ch-e['height'],cy-e['height']/2))
                    e['x']=max(0.,min(cw-e['width'],e['x']));e['y']=max(0.,min(ch-e['height'],e['y']))
                key=json.dumps([[round(e['x']),round(e['y']),round(e['width'])] for e in out['elements']])
                if key in seen:continue
                seen.add(key)
                sd=scene_defects(source,out);ur=unreadable(out)
                low=_cheap_contrast(source,out,host,panels,field)
                from .quality import collisions as _collisions
                hits=len(_collisions(source,out,A))
                subj=sum(text_on_subject(source,out,mode=mode).values())
                cost=10*(ur+len(sd['unsupported_text'])+len(sd['scene_displacement'])+len(low)+hits)+4*subj-2*factor+.5*level
                designs.append((cost,len(designs),out,factor))
                log.append({'inset':inset,'level':level,'factor':factor,'arrangement':arrangement,'anchor':anchor,'unreadable':ur,'unsupported':len(sd['unsupported_text']),
                            'displaced':len(sd['scene_displacement']),'low_contrast':len(low),'collisions':hits,'subject':round(subj,3),'cost':round(cost,2)})
    designs.sort(key=lambda t:(t[0],t[1]))
    # Keep the two best of each scale so verification sees size trade-offs.
    per={};diverse=[]
    for t in designs:
        if per.get(t[3],0)<2:diverse.append(t);per[t[3]]=per.get(t[3],0)+1
    designs=diverse
    # Second stage: the same repair + quality gates the pipeline applies
    # (collisions, contrast on the composited estimate, floors, support).
    from .repair import repair
    from .quality import diagnose
    from .judging import judge_quality
    final=[]
    for cost,n,d,_ in designs[:6]:
        stopped=_stop(budget) or stopped
        if _stop(budget):break
        fixed,_=repair(source,d,A)
        if not passes(validate(source,fixed)):continue
        q=judge_quality(source,fixed,A,diagnose(source,fixed,A))
        subj=sum(text_on_subject(source,fixed,mode=mode).values())
        final.append((10*q['defect_count']+4*subj+.1*cost,n,fixed,q['defect_count']))
    final.sort(key=lambda t:(t[0],t[1]))
    return [d for _,_,d,_ in final[:max_out]],{'status':'partial:'+stopped if stopped else 'ok' if final else 'no_feasible_scene','maps':maps.to_dict(),'configs':log,
                                             'attempted_configs':attempted,'measured_defects':[h for _,_,_,h in final[:max_out]]}
