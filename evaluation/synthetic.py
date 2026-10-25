"""Seeded adversarial designs; synthetic labels are never called human preferences."""
import argparse,json,random
from pathlib import Path

def designs(n=12,seed=7):
    rng=random.Random(seed)
    for i in range(n):
        w,h=rng.choice([(1080,1350),(1600,400),(1080,1080),(600,900)])
        els=[{'id':'background','type':'shape','shape_type':'rectangle','fill_color':'#182331','x':0,'y':0,'width':w,'height':h,'z_index':0}]
        count=rng.randint(2,8)
        for j in range(count):
            text=('UNBREAKABLE'+str(i)*60) if i%7==0 and j==0 else ('A headline worth reading' if j==0 else 'Supporting details '+str(j)+' with meaningful copy.')
            els.append({'id':f'text_{j}','type':'text','content':text,'font_family':'Solstice Sans','font_size':48 if j==0 else 20,'color':'#ffffff','x':w*.1,'y':h*.08+j*(h*.8/count),'width':w*.8,'height':max(32,h*.7/count),'z_index':j+1})
        yield {'source':{'canvas':{'width':w,'height':h},'elements':els},'family':f'synthetic-{seed}-{i}','provenance':'team-generated adversarial fixture','split':'test'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=12);p.add_argument('--seed',type=int,default=7);p.add_argument('--output',type=Path,default=Path('output/synthetic'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for i,d in enumerate(designs(a.count,a.seed)):(a.output/f'synthetic_{i:03}.json').write_text(json.dumps(d,indent=2))
