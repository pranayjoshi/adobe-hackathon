"""Independent visible-geometry and composited contrast diagnostics."""
from __future__ import annotations
import copy, hashlib, json, math, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
from PIL import Image, ImageColor, ImageDraw, ImageOps, ImageStat
from .canonical import ASSETS_DIR, FONTS_DIR, ROOT, Box, element_box, rotated_extent, union_all
from .typography import measure
from . import render_pool
_RENDER_LOCK = threading.RLock()
_CACHE: dict[str, Image.Image] = {}
_CACHE_MAX_BYTES=128*1024*1024
_ESTIMATES: dict[str, Image.Image] = {}
_RESOURCE_BYTES = {}
VERIFICATION_PROFILE = 'official-exact-v1'

def _file_hash(path):
    """Byte identities cached only while file identity/mtime/ctime remain unchanged."""
    stat=path.stat(); signature=(str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if signature not in _RESOURCE_BYTES:
        if len(_RESOURCE_BYTES)>4096:_RESOURCE_BYTES.clear()
        _RESOURCE_BYTES[signature]=hashlib.sha256(path.read_bytes()).hexdigest()
    return _RESOURCE_BYTES[signature]

def resource_fingerprint(design):
    """Include actual images, all available font faces and renderer implementation."""
    paths=[]
    for e in design.get('elements',[]):
        if e.get('type')!='image':continue
        path=(ASSETS_DIR/str(e.get('asset_id',''))).resolve()
        if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe image asset: '+str(e.get('asset_id')))
        paths.append(path)
    paths += [p for p in FONTS_DIR.rglob('*') if p.is_file() and p.suffix.lower() in ('.ttf','.otf','.json')]
    paths += list((ROOT/'renderer'/'converter').glob('*.py'))
    paths += [ROOT/'engine'/'render_worker.py']
    values=[(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),_file_hash(p)) for p in sorted(set(paths))]
    # Installed native renderer package is also part of the rendering profile.
    from importlib.metadata import version, PackageNotFoundError
    try: native=version('cr-renderer')
    except PackageNotFoundError:native='unavailable'
    return hashlib.sha256(json.dumps([VERIFICATION_PROFILE,native,values],sort_keys=True).encode()).hexdigest()

def verify_design(design, output_path=None, profile=VERIFICATION_PROFILE, timeout=12, max_size=None):
    """Render exact serialized design in isolation; failures never become success.

    The caller must remove ``image`` before persisting the returned sidecar.
    Full canvas pixels are returned by default; diagnostic callers may downscale.
    """
    started=time.monotonic()
    result={'status':'failed','verified':False,'image':None,'profile':profile,'cache_hit':False,'warnings':[]}
    try:
        payload=json.dumps(design,sort_keys=True,allow_nan=False,separators=(',',':'))
        result['design_hash']=hashlib.sha256(payload.encode()).hexdigest()
        w,h=(design['canvas'][k] for k in ('width','height'))
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 or int(v)!=v for v in (w,h)):
            raise ValueError('Canvas must use positive integer dimensions')
        if max(w,h)>16384 or w*h>32_000_000:raise ValueError('Canvas exceeds isolated render resource limit')
        if os.environ.get('CHAMELEON_RENDER','1')=='0':
            result.update(status='disabled',error='Rendering disabled');return result
        result['resource_hash']=resource_fingerprint(design)
        key=hashlib.sha256(json.dumps([result['design_hash'],result['resource_hash'],profile,max_size]).encode()).hexdigest()
        with _RENDER_LOCK:
            if key in _CACHE:
                rendered=_CACHE[key].copy();result['cache_hit']=True
            else:
                with tempfile.TemporaryDirectory(prefix='chameleon-verify-') as folder:
                    task=Path(folder)/'design.json';png=Path(folder)/'render.png';meta=Path(folder)/'meta.json'
                    task.write_text(payload)
                    process=render_pool.run([str(task),str(png),str(max_size or int(max(w,h))),str(meta)],timeout=max(.01,float(timeout)))
                    if process.returncode or not png.is_file():
                        detail=json.loads(meta.read_text()).get('error') if meta.is_file() else None
                        raise RuntimeError(detail or 'Official renderer failed')
                    result['warnings']=json.loads(meta.read_text()).get('warnings',[]) if meta.is_file() else []
                    if result['warnings']:raise RuntimeError('Official renderer emitted resource or conversion warnings: '+'; '.join(result['warnings']))
                    with Image.open(png) as im:im.load();rendered=im.convert('RGB').copy()
                    if max_size is None and rendered.size!=(int(w),int(h)):raise RuntimeError('Renderer dimensions do not match exported canvas')
            # Resource files must remain stable for this request. A concurrent
            # edit while waiting for the lock, rendering, or copying a cached
            # image cannot attach old identities to newly read pixels.
            if resource_fingerprint(design)!=result['resource_hash']:
                raise RuntimeError('Resources changed during verification; retry with stable asset/font files')
            if not result['cache_hit']:
                image_bytes=rendered.width*rendered.height*3
                while _CACHE and (len(_CACHE)>=64 or sum(im.width*im.height*3 for im in _CACHE.values())+image_bytes>_CACHE_MAX_BYTES):
                    _CACHE.pop(next(iter(_CACHE)))
                if image_bytes<=_CACHE_MAX_BYTES:_CACHE[key]=rendered.copy()
        if output_path is not None:
            destination=Path(output_path);destination.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.png',delete=False) as handle:pending=Path(handle.name)
            try:rendered.save(pending,format='PNG');os.replace(pending,destination)
            finally:pending.unlink(missing_ok=True)
        result.update(status='verified',verified=True,image=rendered,dimensions=list(rendered.size))
    except subprocess.TimeoutExpired:result['error']='Official renderer timed out'
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:result['error']=str(exc)
    finally:result['duration_ms']=round((time.monotonic()-started)*1000,3)
    return result

def ink(e):
    b = rotated_extent(e)
    if e['type'] == 'text' and not e.get('rotation'):
        f = measure(e, int(e.get('font_size', 1)), max(1, e['width']))
        width=max(0,min(b.w,f.line_width))
        align=e.get('text_align','left');offset=(b.w-width)/2 if align=='center' else b.w-width if align=='right' else 0
        return Box(b.x+offset, b.y, width, min(b.h, f.height))
    return b

def units(A, design):
    by = {e['id']: e for e in design['elements']}
    return [(b, union_all(rotated_extent(by[i]) for i in b.ids if i in by)) for b in A.all_blocks() if any(i in by for i in b.ids)]

def collisions(source, design, A):
    old = {b.id: r for b,r in units(A,source)}
    blocks = units(A,design); out=[]
    for n,(a,x) in enumerate(blocks):
        for b,y in blocks[n+1:]:
            before=old[a.id].intersection_area(old[b.id])/max(1,min(old[a.id].area,old[b.id].area))
            after=x.intersection_area(y)/max(1,min(x.area,y.area))
            if after > before + .025: out.append({'a':a.id,'b':b.id,'fraction':round(after-before,4),'ids':a.ids+b.ids})
    return out

def color(value, opacity=1):
    try:
        if str(value).startswith(('rgb(', 'rgba(')):
            from renderer.converter.colorutil import parse_rgba
            r,g,b,alpha=parse_rgba(value);a=round(alpha*255)
        else:r,g,b,a=ImageColor.getcolor(value or '#000000','RGBA')
    except (ValueError,TypeError): r,g,b,a=0,0,0,255
    return r,g,b,round(a*max(0,min(1,float(opacity))))

def composite(design, size=640):
    """Coarse non-text composite; diagnostic only, original assets untouched."""
    cw,ch=design['canvas']['width'],design['canvas']['height']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in (cw,ch,size)):
        raise ValueError('Diagnostic canvas and size must be positive finite numbers')
    scale=min(1,size/max(cw,ch))
    # Size and bytes participate: images may be edited without changing asset IDs.
    resources=[]
    for e in design['elements']:
        if e['type']=='image':
            p=(ASSETS_DIR/str(e.get('asset_id',''))).resolve()
            resources.append((str(p),_file_hash(p) if p.is_relative_to(ASSETS_DIR.resolve()) and p.is_file() else 'missing'))
    key=json.dumps({'canvas':design['canvas'],'elements':[e for e in design['elements'] if e['type']!='text'],'size':size,'resources':resources},sort_keys=True)
    if key in _ESTIMATES:return _ESTIMATES[key].copy(),scale
    canvas=Image.new('RGBA',(max(1,round(cw*scale)),max(1,round(ch*scale))),'white')
    for e in sorted(design['elements'],key=lambda x:x.get('z_index',0)):
        if e['type']=='text':continue
        scaled=[e[k]*scale for k in ('width','height')]
        if any(not math.isfinite(v) or v<=0 or v>16384 for v in scaled) or scaled[0]*scaled[1]>32_000_000:
            raise ValueError('Diagnostic layer exceeds allocation limit: '+str(e.get('id')))
        w,h=max(1,round(e['width']*scale)),max(1,round(e['height']*scale))
        if e['type']=='image':
            path=(ASSETS_DIR/str(e.get('asset_id',''))).resolve()
            if path.is_relative_to(ASSETS_DIR.resolve()) and path.is_file():
                with Image.open(path) as im:
                    if max(im.size)>16384 or im.width*im.height>32_000_000:
                        raise ValueError('Original image exceeds diagnostic decode limit: '+str(e.get('id')))
                    im=im.convert('RGBA')
                    if e.get('fit')=='cover':layer=ImageOps.fit(im,(w,h))
                    elif e.get('fit')=='fill':layer=im.resize((w,h))
                    else:
                        layer=Image.new('RGBA',(w,h)); fitted=ImageOps.contain(im,(w,h));layer.alpha_composite(fitted,((w-fitted.width)//2,(h-fitted.height)//2))
                layer.putalpha(layer.getchannel('A').point(lambda a:round(a*float(e.get('opacity',1)))))
            else:layer=Image.new('RGBA',(w,h),(180,180,180,255))
        else:
            layer=Image.new('RGBA',(w,h));draw=ImageDraw.Draw(layer)
            fill=color(e.get('fill_color') or e.get('fill') or '#000000',e.get('opacity',1))
            if e.get('shape_type') in ('ellipse','circle'):draw.ellipse((0,0,w-1,h-1),fill=fill)
            else:draw.rectangle((0,0,w-1,h-1),fill=fill)
        x,y=e['x']*scale,e['y']*scale
        if e.get('rotation'):
            original=layer.size;layer=layer.rotate(-float(e['rotation']),expand=True,resample=Image.Resampling.BICUBIC)
            x-=(layer.width-original[0])/2;y-=(layer.height-original[1])/2
        canvas.alpha_composite(layer,(round(x),round(y)))
    result=canvas.convert('RGB')
    if len(_ESTIMATES)>=32:_ESTIMATES.pop(next(iter(_ESTIMATES)))
    _ESTIMATES[key]=result
    return result.copy(),scale

def painted_prefix(design, text_id):
    """Exact stable paint order, including lower text and equal-z lower layers."""
    ordered=sorted(enumerate(design['elements']),key=lambda item:(item[1].get('z_index',0),item[0]))
    prefix=[]
    for _,element in ordered:
        if element['id']==text_id:return {**design,'elements':prefix}
        prefix.append(element)
    raise KeyError(text_id)

def rendered_background(design, text_id=None, timeout=8):
    """Isolated official renderer: native crashes/timeouts cannot kill resize."""
    clean=painted_prefix(design,text_id) if text_id is not None else {**design,'elements':[e for e in design['elements'] if e['type']!='text']}
    result=verify_design(clean,timeout=timeout,max_size=640)
    if result['verified']:return result['image'],'cached' if result['cache_hit'] else 'official'
    return None,'disabled' if result['status']=='disabled' else 'renderer timed out' if 'timed out' in result.get('error','') else 'renderer failed'

def luminance(rgb):
    v=[x/255 for x in rgb[:3]];v=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in v]
    return .2126*v[0]+.7152*v[1]+.0722*v[2]

def text_contrast(design, image=None):
    """Conservative box-region contrast for each actually used foreground color.

    Each color gets its own spatial lower-quartile estimate; the weakest color
    controls the result. Pooling colors first can hide a poorly contrasting
    minority rich-text run. These are box samples, not positioned glyph masks;
    a color can therefore be conservatively assessed outside its actual run.
    """
    ratios={};busy={}
    for e in design['elements']:
        if e['type']!='text' or not str(e.get('content','')).strip():continue
        # Only layers below this text contribute to its background.
        supplied=image.get(e['id']) if isinstance(image,dict) else image
        im,sc=composite(painted_prefix(design,e['id'])) if supplied is None else (supplied,supplied.width/design['canvas']['width'])
        b=ink(e)
        # Match the current converter's inclusive ranges. The schema/corpus
        # endpoint discrepancy is documented in guide10; don't reinterpret it.
        text=str(e['content']);foreground=[e.get('color','#000000')]*len(text)
        for run in e.get('style_runs',[]):
            if run.get('color'):
                for index in range(max(0,int(run.get('start',0))),min(len(text),int(run.get('end',len(text)-1))+1)):
                    foreground[index]=run['color']
        fgs=list(dict.fromkeys(color(value,e.get('opacity',1)) for char,value in zip(text,foreground) if not char.isspace()))
        pixels={f:[] for f in fgs}
        for yy in range(4):
            for xx in range(12):
                x=min(im.width-1,max(0,round((b.x+b.w*(xx+.5)/12)*sc)));y=min(im.height-1,max(0,round((b.y+b.h*(yy+.5)/4)*sc)))
                bg=im.getpixel((x,y));lb=luminance(bg)
                for f in fgs:
                    alpha=f[3]/255;lf=luminance(tuple(alpha*f[i]+(1-alpha)*bg[i] for i in range(3)))
                    pixels[f].append((max(lf,lb)+.05)/(min(lf,lb)+.05))
        quantiles=[]
        for samples in pixels.values():
            vals=sorted(samples);quantiles.append(vals[len(vals)//4])
        ratios[e['id']]=round(min(quantiles),3)
        left,top=max(0,min(im.width,int(b.x*sc))),max(0,min(im.height,int(b.y*sc)))
        right,bottom=max(0,min(im.width,int(b.x1*sc))),max(0,min(im.height,int(b.y1*sc)))
        region=im.crop((left,top,right,bottom)) if right>left and bottom>top else None
        busy[e['id']]=round(ImageStat.Stat(region.convert('L')).stddev[0]/128,3) if region and region.width and region.height else 0
    return ratios,busy

def diagnose(source,design,A,pixels=False,budget=None):
    overflow=[];small=[]
    for e in design['elements']:
        if e['type']!='text':continue
        f=measure(e,int(e['font_size']),max(1,e['width']))
        if f.overflow or f.height>e['height']+.5:overflow.append(e['id'])
        if e['font_size']<7:small.append(e['id'])
    images={};source_images={};statuses=[];deadline=time.monotonic()+8;incomplete=[];attempts=0;charged=0
    if budget is not None:deadline=min(deadline,budget.started+budget.seconds)
    if pixels:
        # A shared per-assessment deadline bounds all prefix workers together.
        for scope,doc,dest in (('design',design,images),('source',source,source_images)):
            for e in doc['elements']:
                if e['type']!='text' or not e.get('content'):continue
                remaining=deadline-time.monotonic()
                reason=None
                if budget is not None and budget.cancellation and budget.cancellation():reason='cancelled'
                elif remaining<=0:reason='deadline'
                elif budget is not None and budget.renders>=budget.max_renders:reason='render_budget'
                if reason:
                    statuses.append('renderer timed out' if reason=='deadline' else reason)
                    incomplete.append({'scope':scope,'id':e['id'],'reason':reason});continue
                # Reserve before work so no prefix call can exceed the cap.
                # Confirmed cache hits refund their reservation. Failed workers
                # conservatively consume one attempt; no success is fabricated.
                attempts+=1
                if budget is not None:budget.renders+=1
                im,status=rendered_background(doc,e['id'],timeout=remaining);statuses.append(status)
                if status=='cached':
                    if budget is not None:budget.renders-=1
                else:charged+=1
                if im is not None:dest[e['id']]=im
                else:incomplete.append({'scope':scope,'id':e['id'],'reason':status})
    failures=[s for s in statuses if s not in ('official','cached')]
    status=failures[0] if failures else ('official' if 'official' in statuses else 'cached') if pixels else 'estimate'
    ratios,busy=text_contrast(design,images);src,_=text_contrast(source,source_images)
    low=[i for i,v in ratios.items() if v<3 and v<.6*src.get(i,21)]
    return {'overflow':overflow,'small_text':small,'collisions':collisions(source,design,A),'contrast':ratios,'contrast_regressions':low,'busyness':busy,'render_mode':status,'contrast_verified':pixels and not failures,'verification_profile':VERIFICATION_PROFILE if pixels else 'estimate',
            'verification_incomplete':incomplete,'pixel_render_attempts':attempts,'pixel_renders_charged':charged,'render_accounting':'conservative_uncached_prefix_attempts',
            'contrast_sampling':'minimum_per_used_color_spatial_quartile; conservative_box_regions_not_glyph_masks'}
