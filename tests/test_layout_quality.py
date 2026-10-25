import copy,unittest
from engine.analysis import analyze
from engine.pipeline import generate
from engine.judging import judge_quality
from engine.quality import diagnose

def text(i,x,y,w=100,h=40,fs=20):return dict(id=i,type='text',x=x,y=y,width=w,height=h,font_family='Solstice Sans',font_size=fs,content='Some text',color='rgba(0,0,0,1)',z_index=2)
def shape(i,x,y,w,h):return dict(id=i,type='shape',shape_type='rectangle',fill_color='rgba(255,255,255,1)',x=x,y=y,width=w,height=h,z_index=0)
class LayoutQualityTests(unittest.TestCase):
 def test_panel_overlap_cannot_swallow_paragraphs(self):
  s={'canvas':{'width':600,'height':800},'elements':[shape('panel',10,10,400,700),text('a',20,30),text('b',20,120),text('c',20,220)]}
  a=analyze(s);self.assertEqual(len(a.blocks),3);self.assertTrue(all(len(b.ids)==1 for b in a.blocks));self.assertTrue(a.panels)
 def test_overlapping_text_stays_independent(self):
  s={'canvas':{'width':600,'height':800},'elements':[text('a',20,20),text('b',20,30)]};self.assertEqual(len(analyze(s).blocks),2)
 def test_explicit_groups_stay_together(self):
  els=[text('a',20,20),text('b',20,70)]
  for e in els:e['group_id']='group'
  a=analyze({'canvas':{'width':600,'height':800},'elements':els});self.assertEqual(len(a.blocks),1);self.assertEqual(len(a.blocks[0].ids),2)
 def test_tiny_type_is_quality_failure_even_with_all_ids(self):
  s={'canvas':{'width':600,'height':800},'elements':[text('a',20,20)]};d=copy.deepcopy(s);d['elements'][0]['font_size']=3;a=analyze(s)
  q=judge_quality(s,d,a,diagnose(s,d,a));self.assertFalse(q['passed']);self.assertEqual(q['defects']['unreadable'],['a'])
 def test_columns_and_safety_candidate_preserve_originals(self):
  s={'canvas':{'width':600,'height':800},'elements':[text(str(i),20,20+i*100) for i in range(5)]};before=copy.deepcopy(s)
  out=generate(s,{'width':1600,'height':400},options={'render':False,'solver':False,'retrieval':False},all_candidates=True)
  self.assertTrue(any(c['template'].startswith('columns_') for c in out));self.assertTrue(any(c['template']=='source_frame' for c in out));self.assertEqual(s,before)
  for c in out:self.assertTrue(c['valid'],c['checks']);self.assertIn('quality',c)
if __name__=='__main__':unittest.main()
