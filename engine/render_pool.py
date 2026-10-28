"""Warm official-renderer processes for isolated verification renders.

Importing the renderer costs about 1.3 s, which dominated every verification when each render
started a fresh process. Workers here stay alive between renders. A worker is replaced after a
crash, a timeout, or MAX_TASKS renders, so native failures still never reach the API process.
CHAMELEON_RENDER_POOL=0 falls back to one process per render.
"""
import atexit, json, os, select, subprocess, sys, threading
from types import SimpleNamespace
from .canonical import ROOT

MAX_TASKS=40
_idle=[];_lock=threading.Lock()
_RUN=subprocess.run  # a substituted subprocess.run (the tests' renderer seam) is honoured


class _Worker:
    def __init__(self):
        self.proc=subprocess.Popen([sys.executable,'-m','engine.render_worker','--serve'],cwd=ROOT,
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        self.tasks=0

    def ask(self,args,timeout):
        self.proc.stdin.write((json.dumps(args)+'\n').encode());self.proc.stdin.flush()
        ready,_,_=select.select([self.proc.stdout],[],[],timeout)
        if not ready:raise subprocess.TimeoutExpired('render_worker',timeout)
        line=self.proc.stdout.readline()
        if not line:raise OSError('render worker exited')
        self.tasks+=1
        return int(line)

    def close(self):
        try:self.proc.kill();self.proc.wait(timeout=2)
        except Exception:pass
        for f in (self.proc.stdin,self.proc.stdout):
            try:f.close()
            except Exception:pass


@atexit.register
def _shutdown():
    with _lock:
        while _idle:_idle.pop().close()


def run(args,timeout):
    """Render like `python -m engine.render_worker *args`; returns an object with `returncode`."""
    if os.environ.get('CHAMELEON_RENDER_POOL','1')=='0' or os.name!='posix' or subprocess.run is not _RUN:
        return subprocess.run([sys.executable,'-m','engine.render_worker',*args],cwd=ROOT,stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,timeout=timeout)
    with _lock:
        worker=_idle.pop() if _idle else None
    if worker is not None and worker.proc.poll() is not None:
        worker.close();worker=None
    if worker is None:
        worker=_Worker()
    try:
        code=worker.ask(list(args),timeout)
    except subprocess.TimeoutExpired:
        worker.close();raise
    except (OSError,ValueError):
        # The worker died mid-render (a native crash): report it as a failed render.
        worker.close();return SimpleNamespace(returncode=1)
    if worker.tasks>=MAX_TASKS:worker.close()
    else:
        with _lock:_idle.append(worker)
    return SimpleNamespace(returncode=code)
