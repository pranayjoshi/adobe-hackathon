"""Short-lived renderer process used to contain native library failures."""
import json,sys
from pathlib import Path
from .canonical import ASSETS_DIR
from renderer.converter.convert import render_design_to_png
if __name__=='__main__':
    # Native allocations are bounded in this short-lived worker, never in the API.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(20,20))
        if sys.platform!='darwin':resource.setrlimit(resource.RLIMIT_AS,(2_000_000_000,2_000_000_000))
    except (ImportError,ValueError,OSError):pass
    design=json.loads(Path(sys.argv[1]).read_text())
    try:
        warnings=render_design_to_png(design,ASSETS_DIR,sys.argv[2],max_size=int(sys.argv[3]) if len(sys.argv)>3 else 640,strict_fonts=True)
        if len(sys.argv)>4:Path(sys.argv[4]).write_text(json.dumps({'warnings':warnings}))
    except Exception as exc:
        if len(sys.argv)>4:Path(sys.argv[4]).write_text(json.dumps({'error':str(exc),'warnings':[]}))
        raise
