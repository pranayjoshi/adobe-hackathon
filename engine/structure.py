"""Bounded visual tests for raster panels; no asset pixels are changed."""
from functools import lru_cache
from PIL import Image
from .canonical import ASSETS_DIR

@lru_cache(maxsize=1024)
def flat_asset(asset_id):
    path=(ASSETS_DIR/str(asset_id or '')).resolve()
    if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():return False
    with Image.open(path) as im:
        im=im.convert('RGBA');im.thumbnail((64,64))
        raw=im.tobytes()
        colors=[tuple(raw[j:j+3]) for j in range(0,len(raw),4) if raw[j+3]>240]
    if not colors:return False
    return all(max(c[i] for c in colors)-min(c[i] for c in colors)<=4 for i in range(3))
