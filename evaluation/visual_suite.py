"""Frozen, resumable visual evidence. Workers import a named code snapshot only."""
from __future__ import annotations
import hashlib, html, json, os, platform, shutil, signal, subprocess, sys, time
from pathlib import Path

ARMS = ('legacy_batch', 'single', 'single_restart', 'small_beam')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str)); tmp.replace(path)

def code_manifest(root):
    root = Path(root)
    files = []
    for folder in ('engine', 'renderer', 'evaluation', 'training'):
        for p in sorted((root / folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts:
                files.append({'path': str(p.relative_to(root)), 'sha256': digest(p)})
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files': files}

def dependency_versions():
    from importlib.metadata import version, PackageNotFoundError
    result = {}
    for package in ('cr-renderer','skia-python','Pillow','ortools','scipy','numpy','httpx'):
        try: result[package] = version(package)
        except PackageNotFoundError: result[package] = 'not-installed'
    return result


def resource_manifest(root):
    root = Path(root) / 'brand_kit'
    files = [{'path': str(p.relative_to(root)), 'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()]
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files': files}

def case_matrix(suite):
    cases = [{'id': s['id'] + '__' + f['id'], 'source': s, 'format': f} for s in suite['sources'] for f in suite['formats']]
    if len(cases) != suite['expected_cases_per_arm'] or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('suite case count or duplicate identity mismatch')
    return cases

def load_source(spec, root):
    path = Path(root) / spec['path']
    if digest(path) != spec['sha256']:
        raise ValueError('source hash mismatch: ' + spec['id'])
    raw = json.loads(path.read_text()); source = raw[spec.get('source_field', 'source')]
    if source['canvas'] != spec['source_canvas'] or len(source['elements']) != spec['element_count']:
        raise ValueError('source geometry mismatch: ' + spec['id'])
    # Provenance is in the task sidecar, never in the editable design envelope.
    return {'canvas': source['canvas'], 'elements': source['elements']}

def complete_slot(folder):
    folder = Path(folder)
    status = folder / 'status.json'
    if not status.exists(): return False
    state = json.loads(status.read_text())
    if state['status'] == 'failed': return (folder / 'failure.json').exists()
    required = ('final.json', 'final.png', 'assessment.json', 'trace.jsonl')
    return all((folder / p).is_file() for p in required) and all(digest(folder / p) == h for p, h in state.get('artifacts', {}).items())

def gallery(folder, rows, baseline=None, arm='legacy_batch'):
    parts = ['<!doctype html><meta charset="utf-8"><title>Frozen design evidence</title><style>body{font:15px system-ui;background:#111b1e;color:#eef;padding:24px}article{border:1px solid #567;padding:16px;margin:16px 0}img{max-width:46%;max-height:430px;object-fit:contain;background:white}pre{white-space:pre-wrap}a{color:#adf}</style><h1>Frozen visual evidence</h1><p>Development regression panel; no human preference claim. Full-resolution images are linked.</p>']
    for r in rows:
        slot = Path('arms') / arm / r['source'] / r['format']
        parts.append('<article><h2>' + html.escape(r['case']) + '</h2><p><a href="inputs/' + r['source'] + '/source.json">Source JSON</a> · <a href="inputs/' + r['source'] + '/source.png">Source reconstruction</a> · <a href="inputs/' + r['source'] + '/reference.jpg">Original reference if available</a></p>')
        if baseline:
            old = next(iter(sorted((Path(baseline) / 'arms').glob('*/' + r['source'] + '/' + r['format'] + '/final.png'))), None)
            if old: parts.append('<p>Baseline</p><img src="' + html.escape(os.path.relpath(old, folder)) + '">')
        if not r.get('rendered') and (Path(folder) / slot / 'diagnostic-adaptive.png').exists(): parts.append('<p>UNVERIFIED adaptive engine diagnostic · missing original</p><img src="' + str(slot / 'diagnostic-adaptive.png') + '">')
        if not r.get('rendered') and (Path(folder) / slot / 'diagnostic-unverified.png').exists(): parts.append('<p>UNVERIFIED MISSING ORIGINAL · diagnostic only</p><img src="' + str(slot / 'diagnostic-unverified.png') + '">')
        if r.get('rendered'): parts.append('<p>Current · <a href="' + str(slot / 'final.json') + '">editable JSON</a></p><a href="' + str(slot / 'final.png') + '"><img src="' + str(slot / 'final.png') + '"></a>')
        parts.append('<pre>' + html.escape(json.dumps(r, indent=2)) + '</pre></article>')
    (Path(folder) / 'index.html').write_text(''.join(parts))

def comparison_sheets(folder, rows, baseline, arm):
    """Display sheets only: final native images remain the scoring artifacts."""
    if not baseline: return
    from PIL import Image, ImageDraw
    folder, baseline = Path(folder), Path(baseline)
    for row in rows:
        source, fmt = row['source'], row['format']
        old = next(iter(sorted((baseline / 'arms').glob('*/' + source + '/' + fmt))), None)
        new = folder / 'arms' / arm / source / fmt
        sheet = Image.new('RGB',(1440,450),'#e9edef'); draw = ImageDraw.Draw(sheet)
        draw.text((16,12),row['case'] + ' | source / baseline / challenger; display only',fill='#17252b')
        items = [('Source reconstruction',folder/'inputs'/source/'source.png',16,280),
                 ('Baseline',old/'final.png' if old else None,320,540),('Challenger',new/'final.png',880,540)]
        scale=min(520/row['target']['width'],340/row['target']['height'])
        for label,path,x,width in items:
            draw.text((x,40),label,fill='#17252b')
            if path and path.exists():
                with Image.open(path) as image:
                    im=image.convert('RGB')
                    if label=='Source reconstruction':im.thumbnail((width,340))
                    else:im=im.resize((max(1,round(row['target']['width']*scale)),max(1,round(row['target']['height']*scale))))
                    sheet.paste(im,(x,75))
            else:draw.text((x,75),'UNAVAILABLE / FAILED',fill='#a01d1d')
        path=folder/'comparisons'/source/(fmt+'.png');path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)


def run(suite_path, experiment, arm='legacy_batch', profile='offline', config=None, code_root=None,
        baseline=None, resume=False, timeout=180, repository_root=None, unverified_level3=False):
    repo = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    code = Path(code_root or repo).resolve(); suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text()); cases = case_matrix(suite)
    if unverified_level3: cases = [c for c in cases if c['source']['id']=='level3']
    folder = repo / suite.get('artifact_root', 'output/agent-benefits') / experiment
    options = {'render': True, 'roles': False, 'vlm': False, 'controller_mode': arm, 'visual_judge': 'off'}
    options.update((config or {}).get('options', {})); options['controller_mode'] = arm
    if unverified_level3:
        if arm!='legacy_batch' or profile!='offline': raise ValueError('unverified Level3 diagnostics use the offline legacy engine only')
        options={'render':False,'roles':False,'vlm':False,'refine_seconds':4,'controller_mode':'legacy_batch','visual_judge':'off'}
    if profile == 'offline' and (options.get('visual_judge', 'off') != 'off' or options.get('vlm') or options.get('learned_edit')):
        raise ValueError('offline profile forbids HTTP judges and VLM calls')
    resolved = {'schema_version': 1, 'suite_sha256': digest(suite_path), 'suite_id': suite['suite_id'], 'suite':suite, 'reference_sha256':{item['id']:digest(repo/item['reference_path']) if item.get('reference_path') and (repo/item['reference_path']).is_file() else None for item in suite['sources']},
                'arm': arm, 'profile': profile, 'configuration': config or {}, 'runtime_environment': {k:os.environ[k] for k in ('CHAMELEON_RENDER','CHAMELEON_LAYOUT_LIBRARY','CHAMELEON_LAYOUT_LIBRARY_V2','CHAMELEON_ROLE_MODEL','CHAMELEON_ROLE_WEIGHTS','CHAMELEON_ROLE_WEIGHTS_V2','CHAMELEON_RANKER_WEIGHTS','CHAMELEON_GROUP_POLICIES','CHAMELEON_GROUP_POLICY_FEATURES','CHAMELEON_PROPOSAL_PORTFOLIO','CHAMELEON_RETRIEVAL_VERSION','CHAMELEON_REFINEMENT_MODE','CHAMELEON_PHOTO_ANALYSIS','CHAMELEON_OPENJEV_URL','CHAMELEON_OPENJEV_MODEL','CHAMELEON_VLM','CHAMELEON_VLM_BACKEND','CHAMELEON_VLM_BUNDLE','CHAMELEON_VLM_MODEL','CHAMELEON_VLM_MMPROJ') if k in os.environ}, 'options': options, 'timeout_seconds': timeout, 'code_root': str(code),
                'code': code_manifest(code), 'resources': resource_manifest(code), 'python': sys.version,
                'platform': platform.platform(), 'dependencies': dependency_versions(), 'executable': sys.executable, 'network_policy': 'Python socket guard; native subprocess network isolation not certified',
                'source_snapshot': json.loads((code / 'snapshot.json').read_text()) if (code / 'snapshot.json').exists() else None,
                'unverified_level3_only':unverified_level3, 'harness_sha256': digest(__file__), 'expected_missing_resources': ['level3:solstice_holiday_badge_2025.png (intentional FAQ.md missing-asset case; never substitute)'], 'expected_cases': len(cases), 'training_allowed': False, 'baseline': str(baseline) if baseline else None}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        if not resume: raise FileExistsError('experiment exists; use --resume or a new name')
        if json.loads((folder / 'manifest.json').read_text()) != resolved: raise ValueError('resume configuration/code/resource identity mismatch')
    else: atomic_json(folder / 'manifest.json', resolved)
    harness = folder / 'worker.py'
    if not harness.exists(): shutil.copyfile(Path(__file__), harness)
    if digest(harness) != resolved['harness_sha256']: raise ValueError('frozen worker harness mismatch')
    rows = []
    for c in cases:
        spec = c['source']; fmt = c['format']; slot = folder / 'arms' / arm / spec['id'] / fmt['id']
        slot.mkdir(parents=True, exist_ok=True)
        if not (resume and complete_slot(slot)):
            try:
                source = load_source(spec, repo)
                inp = folder / 'inputs' / spec['id']; inp.mkdir(parents=True, exist_ok=True)
                atomic_json(inp / 'source.json', source)
                if spec.get('reference_path'):
                    ref = repo / spec['reference_path']
                    if not ref.is_file(): raise FileNotFoundError('missing reference ' + str(ref))
                    shutil.copyfile(ref, inp / 'reference.jpg')
                task = {'source': source, 'target': {'width': fmt['width'], 'height': fmt['height']}, 'options': options,
                        'profile': profile, 'code_root': str(code), 'slot': str(slot), 'input_folder': str(inp), 'unverified_level3':unverified_level3}
                atomic_json(slot / 'task.json', task)
                env = dict(os.environ); env['PYTHONPATH'] = str(code); env['PYTHONHASHSEED'] = '0'; env['OMP_NUM_THREADS'] = '1'
                # No shell; snapshot cwd + worker bootstrap prevent imports from the mutable checkout.
                cmd = [sys.executable, str(harness.resolve()), '--worker', str(slot / 'task.json')]
                with (slot / 'worker.log').open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try: returncode = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        raise TimeoutError(f'case exceeded {timeout}s; worker process group terminated')
                if returncode or not complete_slot(slot): raise RuntimeError(f'worker failed with exit code {returncode}; see worker.log')
            except Exception as exc:
                if not (slot / 'failure.json').exists(): atomic_json(slot / 'failure.json', {'stage': 'case_execution', 'error': f'{type(exc).__name__}: {exc}'})
                prior_status = json.loads((slot / 'status.json').read_text()) if (slot / 'status.json').is_file() else {}
                atomic_json(slot / 'status.json', {**(prior_status if prior_status.get('status')=='failed' else {}), 'status': 'failed', 'rendered': False, 'valid': False})
        state = json.loads((slot / 'status.json').read_text())
        if state['status']=='failed' and (slot / 'failure.json').exists(): state['failure']=json.loads((slot / 'failure.json').read_text())
        row = {'case': c['id'], 'source': spec['id'], 'family': spec.get('source_id', spec['id']), 'format': fmt['id'], 'target': {'width': fmt['width'], 'height': fmt['height']}, **state}
        rows.append(row)
        summary = {'cases': len(cases), 'finished': len(rows), 'rendered': sum(r.get('rendered', False) for r in rows), 'valid': sum(r.get('valid', False) for r in rows), 'failed': sum(r['status'] == 'failed' for r in rows)}
        atomic_json(folder / 'results.json', {'summary': summary, 'cases': rows}); gallery(folder, rows, baseline, arm)
        print(row['case'], row['status'], row.get('seconds', ''), flush=True)
    comparison_sheets(folder, rows, baseline, arm)
    current_code, current_resources = code_manifest(code), resource_manifest(code)
    integrity = {'code_unchanged': current_code == resolved['code'], 'resources_unchanged': current_resources == resolved['resources']}
    atomic_json(folder / 'integrity.json', integrity)
    if not all(integrity.values()): raise RuntimeError('code or resources changed during experiment; results are not a frozen run')
    (folder / 'handoff.md').write_text(f"# {experiment}\n\n{json.dumps(summary)}\n\nArm: `{arm}`. All final PNGs use the official renderer at native target size.\n\nSource reconstruction PNGs and available Crello references are retained for visual review. No human preference votes were collected. Resource accounting reports case wall time and worker peak RSS; native renderer subprocess memory is not certified.\n")
    return folder


def worker(task_path):
    task = json.loads(Path(task_path).read_text()); code = Path(task['code_root']); sys.path.insert(0, str(code))
    slot = Path(task['slot']); started = time.monotonic()
    import contextlib, inspect, resource, socket
    from engine.pipeline import generate
    from renderer.converter.convert import render_design_to_png, _normalize_weight, _is_italic, _weight_style_to_font_style
    from renderer.converter.brand_fonts import build_font_dict
    from cr_renderer.fonts import normalize_family
    from engine.canonical import ASSETS_DIR, FONTS_DIR
    source = task['source']; target = task['target']; network = []; renders = [0]
    original_connect = socket.socket.connect
    if task['profile'] == 'offline':
        def refuse(sock, address):
            network.append(str(address)); raise OSError('network prohibited during offline visual suite')
        socket.socket.connect = refuse
    try:
        fonts = build_font_dict(str(FONTS_DIR)); missing = []
        for e in source['elements']:
            if e['type'] == 'image' and not (ASSETS_DIR / e['asset_id']).is_file(): missing.append('asset:' + e['asset_id'])
            if e['type'] == 'text':
                weight = _normalize_weight(e.get('font_weight')); style = _weight_style_to_font_style(weight in ('bold', 'semibold', 'extrabold', 'black'), _is_italic(e.get('font_style') or e.get('italic')))
                available = fonts.get(normalize_family(e['font_family']), [])
                if not any(f['fontWeight'] == weight and f['fontStyle'] == style for f in available): missing.append('font:' + e['font_family'] + ':' + weight + ':' + style)
        if missing:
            # Official diagnostic only. Its missing-asset placeholder is never promoted to final or success.
            diagnostic = slot / 'diagnostic-unverified.png'
            try:
                from engine.pipeline import uniform_stretch
                preview = uniform_stretch(source, target)
                atomic_json(slot / 'diagnostic-unverified.json', preview)
                warnings = render_design_to_png(preview, ASSETS_DIR, diagnostic, max_size=max(target.values()))
                atomic_json(slot / 'diagnostic-unverified.meta.json', {'verification':'UNVERIFIED_MISSING_ORIGINAL','method':'uniform_stretch (not adaptive)','missing':missing,'warnings':warnings,'counts_as_success':False})
            except Exception as diagnostic_exc:
                atomic_json(slot / 'diagnostic-unverified.meta.json', {'verification':'UNVERIFIED_MISSING_ORIGINAL','error':str(diagnostic_exc),'counts_as_success':False})
            if task.get('unverified_level3'):
                diagnostic_options = {'render':False,'roles':False,'vlm':False,'refine_seconds':4,'controller_mode':'legacy_batch'}
                candidate = generate(source,target,top_k=1,options=diagnostic_options)[0]
                design=candidate['design']; atomic_json(slot/'diagnostic-adaptive.json',design)
                warnings=render_design_to_png(design,ASSETS_DIR,slot/'diagnostic-adaptive.png',max_size=max(target.values()))
                atomic_json(slot/'diagnostic-adaptive.meta.json',{'verification':'UNVERIFIED_MISSING_ORIGINAL','method':'legacy adaptive engine diagnostic only','missing':missing,'warnings':warnings,'counts_as_success':False,'options':diagnostic_options})
                atomic_json(slot/'diagnostic-adaptive.assessment.json',{k:v for k,v in candidate.items() if k!='design'})
            raise ValueError('missing exact resources: ' + ', '.join(sorted(set(missing))))
        source_png = Path(task['input_folder']) / 'source.png'
        source_render_calls = int(not source_png.exists())
        if not source_png.exists():
            warnings = render_design_to_png(source, ASSETS_DIR, source_png, max_size=max(source['canvas'].values()))
            atomic_json(source_png.with_suffix('.render.json'), {'warnings': warnings, 'renderer': 'official', 'reconstruction_review': 'pending_visual_review'})
        trace = (slot / 'trace.jsonl').open('w')
        def event(ev):
            safe = {k:v for k,v in ev.items() if k not in ('image','design')}
            trace.write(json.dumps(safe, default=str) + '\n'); trace.flush()
        def artifact(ev):
            iterations = slot / 'iterations'; iterations.mkdir(exist_ok=True)
            stem = iterations / f'{renders[0]:04d}'; renders[0] += 1
            if ev.get('design'): atomic_json(stem.with_suffix('.json'), ev['design'])
            im = ev.get('image')
            if im is not None: im.save(stem.with_suffix('.png'))
            atomic_json(stem.with_suffix('.meta.json'), {k:v for k,v in ev.items() if k not in ('image','design')}); event(ev)
        # Observe the official native worker at its process boundary, including
        # full revisions and per-prefix diagnostics. Cached images make no call.
        original_run = subprocess.run
        def archive_native(command, *args, **kwargs):
            response = original_run(command, *args, **kwargs)
            if isinstance(command,(list,tuple)) and 'engine.render_worker' in command:
                index = command.index('engine.render_worker')
                task_file, png = Path(command[index+1]), Path(command[index+2])
                archive = slot / 'native-renders'; archive.mkdir(exist_ok=True)
                stem = archive / f'{len(list(archive.glob("*.meta.json"))):04d}'
                if task_file.is_file(): shutil.copyfile(task_file,stem.with_suffix('.json'))
                if png.is_file(): shutil.copyfile(png,stem.with_suffix('.png'))
                atomic_json(stem.with_suffix('.meta.json'), {'kind':'official_native_call','returncode':response.returncode,'rendered':png.is_file(),'counts_as_iteration':False})
            return response
        subprocess.run = archive_native
        params = inspect.signature(generate).parameters
        kwargs = {'top_k': 1, 'options': task['options']}
        if 'progress_callback' in params: kwargs['progress_callback'] = event
        if 'artifact_callback' in params: kwargs['artifact_callback'] = artifact
        results = generate(source, target, **kwargs)
        if not results: raise ValueError('no valid design returned')
        result = results[0]; design = result['design']
        if set(design) != {'canvas','elements'}: raise ValueError('noncanonical design envelope')
        atomic_json(slot / 'final.json', design)
        warnings = render_design_to_png(design, ASSETS_DIR, slot / 'final.png', max_size=max(target.values()))
        from PIL import Image
        with Image.open(slot / 'final.png') as im:
            if im.size != (target['width'], target['height']): raise ValueError('renderer output not native target resolution')
        if not renders[0]:
            artifact({'type':'revision','design':design,'rendered':True,'accepted':True,'origin':'legacy_final','assessment':{},'image':Image.open(slot / 'final.png')})
        assessment = {k:v for k,v in result.items() if k != 'design'}
        assessment['render'] = {'renderer':'official','warnings':warnings,'native_resolution':True}; assessment['network_attempts'] = network
        atomic_json(slot / 'assessment.json', assessment); trace.close()
        required = ('final.json','final.png','assessment.json','trace.jsonl')
        atomic_json(slot / 'status.json', {'status':'complete','rendered':True,'valid':bool(result.get('valid') and result.get('rubric',{}).get('gate',True)),
                    'quality_passed':bool(result.get('rubric',{}).get('passed',False)), 'score':result.get('rubric',{}).get('tier2',{}).get('overall'),
                    'defects':result.get('rubric',{}).get('tier2',{}).get('defects',[]), 'seconds':round(time.monotonic()-started,3),
                    'iteration_artifacts':renders[0], 'render_accounting':{'source_direct_calls':source_render_calls,'final_direct_calls':1,'native_worker_calls':len(list((slot / 'native-renders').glob('*.meta.json'))),'callback_artifacts':renders[0]}, 'worker_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'worker_peak_rss_unit':'bytes' if sys.platform == 'darwin' else 'KiB', 'network_attempts':network,
                    'usage':result.get('controller',{}).get('usage'), 'stop_reason':result.get('controller',{}).get('stop_reason'),
                    'artifacts':{p:digest(slot/p) for p in required}})
    except Exception as exc:
        atomic_json(slot / 'failure.json', {'stage':'worker','error':f'{type(exc).__name__}: {exc}','controller_report':getattr(exc,'report',None)})
        if getattr(exc,'report',None) is not None: atomic_json(slot / 'assessment.json',{'controller':exc.report,'verified':False,'exportable':False})
        atomic_json(slot / 'status.json', {'status':'failed','rendered':False,'valid':False,'seconds':round(time.monotonic()-started,3)})
        raise
    finally: socket.socket.connect = original_connect

if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--worker': worker(sys.argv[2])
