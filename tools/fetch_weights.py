"""Explicit build-time weight download with required SHA256; no runtime network."""
import argparse,hashlib,os,urllib.request
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('sha256');p.add_argument('output',type=Path);a=p.parse_args()
    if not a.url.startswith('https://') or len(a.sha256)!=64:p.error('HTTPS URL and full SHA256 required')
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.partial');digest=hashlib.sha256()
    try:
        with urllib.request.urlopen(a.url,timeout=60) as src,temp.open('wb') as dst:
            while data:=src.read(1024*1024):dst.write(data);digest.update(data)
        if digest.hexdigest()!=a.sha256.lower():raise ValueError('SHA256 mismatch; weight file not installed')
        os.replace(temp,a.output);print('Verified',a.output)
    finally:
        if temp.exists():temp.unlink()
