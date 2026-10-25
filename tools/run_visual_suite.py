"""Run the exact frozen 16-case panel; see --help for reproducibility switches."""
import argparse, json
from pathlib import Path
from evaluation.visual_suite import ARMS, run

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--suite',type=Path,default=Path('docs/individual-update-plans/visual-suite.v1.json'))
    p.add_argument('--arm',choices=ARMS,default='legacy_batch');p.add_argument('--experiment',required=True)
    p.add_argument('--profile',choices=('offline','local_judge'),default='offline');p.add_argument('--config',type=Path)
    p.add_argument('--code-root',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--unverified-level3',action='store_true',help='separate four-case adaptive diagnostic only; never a successful verified arm');p.add_argument('--case-timeout',type=float,default=180)
    a=p.parse_args();print(run(a.suite,a.experiment,a.arm,a.profile,json.loads(a.config.read_text()) if a.config else None,a.code_root,a.baseline,a.resume,a.case_timeout,unverified_level3=a.unverified_level3))
if __name__=='__main__':main()
