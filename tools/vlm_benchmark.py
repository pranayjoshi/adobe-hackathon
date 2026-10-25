"""Order-reversed visual judgment probe; records failures instead of inventing scores."""
import argparse,json,tempfile,time
from pathlib import Path
from PIL import Image
from engine.canonical import ASSETS_DIR
from engine.vlm import describe,status
from renderer.converter.convert import render_design_to_png
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path,default=Path('output/vlm-probe.json'));a=p.parse_args();records=[]
    if not status()['enabled'] or not status()['weights_present']:raise SystemExit('Enable a permitted local model and projector first; no model download is automatic.')
    schema={'type':'object','required':['winner'],'additionalProperties':False,'properties':{'winner':{'type':'string','enum':['A','B','tie']}}}
    with tempfile.TemporaryDirectory() as tmp:
        ims=[]
        for i,file in enumerate((a.a,a.b)):
            d=json.loads(file.read_text());d=d.get('design',d);png=Path(tmp)/f'{i}.png';render_design_to_png(d,ASSETS_DIR,png,max_size=672)
            with Image.open(png) as im:ims.append(im.convert('RGB').copy())
        for order in ((0,1),(1,0)):
            # Composite probes intentionally reveal positional bias; this is a
            # capability diagnostic, not a deployed aesthetic evaluator.
            sheet=Image.new('RGB',(1344,672),'white')
            for j,index in enumerate(order):
                im=ims[index].copy();im.thumbnail((672,672));sheet.paste(im,(j*672,0))
            start=time.monotonic();reply=describe(sheet,'Compare design A on the left and B on the right. Which preserves readable, intentional composition? Return A, B, or tie.',schema)
            records.append({'order':order,'reply':reply,'seconds':time.monotonic()-start})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'probes':records,'notes':'Reversed-order composite probe; no ground-truth accuracy claimed.'},indent=2))
