"""Renderer process used to contain native library failures: one render per process, or warm (--serve)."""
import json,sys
from pathlib import Path
from .canonical import ASSETS_DIR
from renderer.converter.convert import render_design_to_png
def render(args):
    """One render: args are (design.json, out.png[, max_size[, meta.json]])."""
    design=json.loads(Path(args[0]).read_text())
    try:
        warnings=render_design_to_png(design,ASSETS_DIR,args[1],max_size=int(args[2]) if len(args)>2 else 640,strict_fonts=True)
        if len(args)>3:Path(args[3]).write_text(json.dumps({'warnings':warnings}))
    except Exception as exc:
        if len(args)>3:Path(args[3]).write_text(json.dumps({'error':str(exc),'warnings':[]}))
        raise


if __name__=='__main__':
    serve=sys.argv[1:]==['--serve']
    # Native allocations are bounded in this worker process, never in the API. A warm worker
    # (--serve, engine/render_pool.py) renders up to 40 designs before it is replaced.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(400,400) if serve else (20,20))
        if sys.platform!='darwin':resource.setrlimit(resource.RLIMIT_AS,(2_000_000_000,2_000_000_000))
    except (ImportError,ValueError,OSError):pass
    if not serve:
        render(sys.argv[1:])
    else:
        # Replies go on a private copy of stdout; the renderer's own prints (font fallbacks)
        # are sent to stderr so they cannot corrupt the protocol.
        import os
        reply=os.fdopen(os.dup(1),'w');os.dup2(2,1);sys.stdout=sys.stderr
        for line in sys.stdin:
            try:render(json.loads(line));code=0
            except Exception:code=1
            reply.write(f'{code}\n');reply.flush()
