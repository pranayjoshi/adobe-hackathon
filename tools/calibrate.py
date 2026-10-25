"""Fit pairwise preferences with source-family holdout; no automatic deployment.

Kept for the README command; the implementation lives in training/train_ranker.py.
"""
import argparse
from pathlib import Path
from training.train_ranker import load, train

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',type=Path);p.add_argument('--synthetic',type=Path);p.add_argument('--output',type=Path,default=Path('output/calibration/ranker.json'));a=p.parse_args()
    train(load(a.labels,'human'),load(a.synthetic,'synthetic') if a.synthetic else [],a.output)
