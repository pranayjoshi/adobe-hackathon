"""Disposable durable job process; native failures cannot kill the API.

Invoked only by the local supervisor, with a stored request and claim token.
"""
from pathlib import Path
import sys


def main():
    rid,owner,root=sys.argv[1:]
    from service import workspace
    workspace.STORE=Path(root)
    from service.app import GenerateRequest,_perform_job
    repo=workspace.repository()
    current=repo.job(rid)
    if not current or current.get('owner')!=owner or current['state']!='running':
        return 2
    try:
        request=GenerateRequest.model_validate(repo.job_request(rid))
        _perform_job(request,rid,owner,repo)
        return 0
    except Exception as exc:
        message=type(exc).__name__+': '+str(exc)
        repo.append_event(rid,{'type':'job_failed','error':message},owner)
        repo.update_job(rid,{'state':'failed','stage':'failed','error':message},owner)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
