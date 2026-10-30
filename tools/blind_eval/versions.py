"""Fingerprints shared by the build and the engine workers."""
import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code_fingerprint():
    """Fingerprint of the merged engine's code and switches; every merged run in a study must share it."""
    files = sorted((ROOT / 'merged').glob('*.py')) + sorted((ROOT / 'merged').glob('*.json'))
    switches = ','.join(f"{k}={os.environ.get(k, '1')}" for k in ('MERGED_GROWTH', 'MERGED_SCALE_SPACING', 'MERGED_FLOORPLAN_LEEWAY', 'MERGED_PREFS'))
    from merged.gaurav_bridge import root
    gaurav = root()
    switches += f",gaurav={git_version(gaurav) if gaurav else 'off'}"
    return hashlib.sha256(b''.join(f.name.encode() + f.read_bytes() for f in files) + switches.encode()).hexdigest()[:16]


def git_version(root):
    """Commit of an engine checkout (Gaurav's), so a pull makes its old runs stale."""
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip() or None
