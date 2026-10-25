"""Bounded group-level collision repair and renderer-compatible text fitting."""
import copy
from .canonical import rotated_extent,element_box
from .quality import units,collisions
from .typography import fit_text,measure

def contain(design,A,locked=()):
    locked=set(locked)
    by={e['id']:e for e in design['elements']};cw,ch=design['canvas']['width'],design['canvas']['height']
    for block,box in units(A,design):
        if locked.intersection(block.ids):continue
        k=min(1,cw/max(1,box.w),ch/max(1,box.h))
        for eid in block.ids:
            e=by[eid]
            e['x']=(e['x']-box.x)*k+box.x;e['y']=(e['y']-box.y)*k+box.y
            e['width']=max(.01,e['width']*k);e['height']=max(.01,e['height']*k)
            if e['type']=='text':e['font_size']=max(1,int(e['font_size']*k))
        b=next(r for u,r in units(A,design) if u.id==block.id)
        dx=max(0,-b.x)-max(0,b.x1-cw);dy=max(0,-b.y)-max(0,b.y1-ch)
        for eid in block.ids:by[eid]['x']+=dx;by[eid]['y']+=dy
    return design

def move(design,block,dx,dy):
    ids=set(block.ids)
    for e in design['elements']:
        if e['id'] in ids:e['x']+=dx;e['y']+=dy

def repair(source,design,A,locked=()):
    out=copy.deepcopy(design);trace=[]
    if isinstance(locked,dict):
        by={e['id']:e for e in out['elements']}
        for eid,rect in locked.items():
            if eid in by:by[eid].update({k:v for k,v in rect.items() if k in ('x','y','width','height')})
    locked=set(locked)
    out=contain(out,A,locked)
    for _ in range(4):
        pairs=collisions(source,out,A)
        if not pairs:break
        best=None;cost=len(pairs)+sum(p['fraction'] for p in pairs)
        blocks={b.id:(b,r) for b,r in units(A,out)}
        for p in pairs[:6]:
            a,x=blocks[p['a']];b,y=blocks[p['b']]
            for moving,other,mb,ob in ((a,b,x,y),(b,a,y,x)):
                if locked.intersection(moving.ids):continue
                for dx,dy in ((ob.x1-mb.x+2,0),(ob.x-mb.x1-2,0),(0,ob.y1-mb.y+2),(0,ob.y-mb.y1-2)):
                    trial=copy.deepcopy(out);move(trial,moving,dx,dy);contain(trial,A,locked)
                    ps=collisions(source,trial,A);c=len(ps)+sum(v['fraction'] for v in ps)
                    if c<cost-1e-6:best=(trial,{'group':moving.id,'dx':round(dx,2),'dy':round(dy,2)});cost=c
        if best is None:break
        out,step=best;trace.append(step)
    for e in out['elements']:
        if e['type']=='text' and e['id'] not in locked:
            e['font_size']=fit_text(e,e['width'],e['height'],int(e['font_size'])).font_size
    # Flexible backdrops grow around the text they originally contained.
    by={e['id']:e for e in out['elements']}
    from .canonical import union_all
    for panel in A.panels:
        if panel['id'] in locked:continue
        children=[by[i] for i in panel['children'] if i in by]
        if not children:continue
        bounds=union_all(rotated_extent(e) for e in children);pad=max(2,.02*min(out['canvas'].values()))
        x=max(0,bounds.x-pad);y=max(0,bounds.y-pad)
        width=min(out['canvas']['width']-x,bounds.x1-x+pad);height=min(out['canvas']['height']-y,bounds.y1-y+pad)
        original=next(e for e in source['elements'] if e['id']==panel['id'])
        if original['type']=='image' and original.get('fit')!='cover':
            k=max(width/original['width'],height/original['height'])
            width,height=original['width']*k,original['height']*k
            if width>out['canvas']['width'] or height>out['canvas']['height']:continue
            x=min(x,out['canvas']['width']-width);y=min(y,out['canvas']['height']-height)
        by[panel['id']].update(x=x,y=y,width=width,height=height)
    return out,trace
