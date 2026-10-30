"""Fingerprints shared by the build and the engine workers."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code_fingerprint():
    """Fingerprint of the merged engine's code and switches; every merged run in a study must share it."""
    files = sorted((ROOT / 'merged').glob('*.py'))
    switches = ','.join(f"{k}={os.environ.get(k, '1')}" for k in ('MERGED_GROWTH', 'MERGED_SCALE_SPACING'))
    return hashlib.sha256(b''.join(f.name.encode() + f.read_bytes() for f in files) + switches.encode()).hexdigest()[:16]
