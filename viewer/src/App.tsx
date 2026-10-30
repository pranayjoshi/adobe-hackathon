import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ratioLabel } from "./api";
import { ActiveDesign } from "./components/ActiveDesign";
import { CandidateCard } from "./components/CandidateCard";
import { DetailModal } from "./components/DetailModal";
import { Preview } from "./components/Preview";
import { Workspace } from "./components/Workspace";
import { CrelloBrowser, type CrelloDesign } from "./components/CrelloBrowser";
import { Sidebar } from "./components/Sidebar";
import { Icon, Logo, Toggle, roleColor } from "./components/ui";
import type { ActivePreview, Analysis, AnalysisOverrides, BlockInfo, Candidate, ControllerMode, Design, GenerateResponse, Preset, Sample, Target } from "./types";

export default function App() {
  const sourceRequest=useRef(0);
  const generationAbort=useRef<AbortController|null>(null);
  const generationRevision=useRef(0);
  const currentJob=useRef('');
  const [controllerMode,setControllerMode]=useState<ControllerMode>('legacy_batch');
  const [solverMode,setSolverMode]=useState('legacy');
  const [stage,setStage]=useState('');
  const [activePreview,setActivePreview]=useState<ActivePreview|null>(null);
  const [crelloOpen,setCrelloOpen]=useState(false);
  const [crelloSource,setCrelloSource]=useState<CrelloDesign|null>(null);
  const [jobId,setJobId]=useState("");
  const [progress,setProgress]=useState(0);
  const [workspace,setWorkspace] = useState(false);
  const [allCandidates,setAllCandidates] = useState(false);
  const [locks,setLocks] = useState<Record<string,Record<string,number>>>({});
  const [lockId,setLockId] = useState("");
  const [useSolver,setUseSolver] = useState(true);
  const [useRefine,setUseRefine] = useState(true);
  const [useRetrieval,setUseRetrieval] = useState(true);
  const [learnedLayouts,setLearnedLayouts] = useState(true);
  const [semanticLayout,setSemanticLayout] = useState(false);
  const [analysisOverrides,setAnalysisOverrides] = useState<AnalysisOverrides>({roles:{},group_modes:{}});
  const [appliedIntent,setAppliedIntent] = useState(intentKey({roles:{},group_modes:{}}));
  const [samples, setSamples] = useState<Sample[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [health, setHealth] = useState<{ renderer: boolean; text_metrics: boolean; learned_layouts?: boolean } | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState("design");
  const [source, setSource] = useState<Design | null>(null);
  const [sourceRender, setSourceRender] = useState<string | null>(null);
  const [sourceWarnings, setSourceWarnings] = useState<string[]>([]);
  const [sourcePreviewVerified, setSourcePreviewVerified] = useState<boolean | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [targets, setTargets] = useState<Target[]>([]);
  const [emphasis, setEmphasis] = useState("balanced");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<GenerateResponse | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [baselines, setBaselines] = useState(false);
  const [open, setOpen] = useState<{ t: number; c: number } | null>(null);

  const loadSource = useCallback(async (d: Design, id: string | null, name: string, task?: { width: number; height: number }) => {
    ++sourceRequest.current;
    generationAbort.current?.abort();
    if(currentJob.current)void api.cancel(currentJob.current).catch(()=>{});
    setBusy(false);
    setActivePreview(null);setOpen(null);
    setSource(d);
    setCrelloSource(null);
    setLocks({});
    setAnalysisOverrides({roles:{},group_modes:{}});setAppliedIntent(intentKey({roles:{},group_modes:{}}));
    setSourceId(id);
    setSourceName(name);
    setResp(null);
    setSourceRender(null);
    setSourceWarnings([]);
    setSourcePreviewVerified(null);
    setAnalysis(null);
    setError(null);
    if (task) {
      setTargets((prev) => {
        const keep = prev.filter((t) => !t.id.startsWith("task"));
        const exists = keep.some((t) => t.width === task.width && t.height === task.height);
        return exists ? keep : [{ id: `task-${task.width}x${task.height}`, label: "Task target", ...task }, ...keep];
      });
    }

  }, []);

  useEffect(()=>{
    if(!source)return;
    const abort=new AbortController();
    const revision=sourceRequest.current;
    setAnalysis(null);setSourceWarnings([]);setSourcePreviewVerified(null);
    api.analyze(source,semanticLayout?'hierarchical':'legacy',abort.signal).then(result=>{
      if(abort.signal.aborted||revision!==sourceRequest.current)return;
      setAnalysis(result.analysis);setSourceRender(result.render);setSourceWarnings(result.warnings??[]);
      setSourcePreviewVerified(result.preview_verified??result.render!=null);
    }).catch(error=>{if(!abort.signal.aborted&&revision===sourceRequest.current)setError(String(error));});
    return()=>abort.abort();
  },[source,semanticLayout]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.presets().then(setPresets).catch(() => {});
    api
      .samples()
      .then((s) => {
        setSamples(s);
        const first = s.find((x) => x.id.includes("level2")) ?? s[0];
        if (first) {
          setTargets([]);
          loadSource(first.source, first.id, first.id.replace("chameleon_sample_", ""), first.target_canvas);
        }
      })
      .catch((e) => setError(`Engine service unreachable — start it with uvicorn. (${e.message})`));
  }, [loadSource]);

  useEffect(()=>{generationRevision.current+=1;},[source,targets,emphasis,locks,controllerMode,solverMode,useSolver,useRefine,useRetrieval,learnedLayouts,semanticLayout,allCandidates,analysisOverrides]);
  useEffect(()=>()=>{generationAbort.current?.abort();},[]);

  const generate = async () => {
    if (!source) return;
    const revision=generationRevision.current;
    const sourceRevision=sourceRequest.current;
    const effectiveIntent={roles:{...analysisOverrides.roles},group_modes:semanticLayout?{...analysisOverrides.group_modes}:{}};
    generationAbort.current?.abort();
    const abort=new AbortController();generationAbort.current=abort;
    setBusy(true);setProgress(0);setJobId('');setStage('queued');setActivePreview(null);setError(null);
    try {
      const result=await api.generate(source,targets,emphasis,locks,allCandidates,
        {solver:useSolver,solver_mode:solverMode,controller_mode:controllerMode,
         refine:useRefine,retrieval:useRetrieval,learned_retarget:learnedLayouts&&health?.learned_layouts!==false,
         ...(semanticLayout?{group_policies:'hierarchical',proposal_portfolio:'diverse',retrieval_version:'fit_aware',
             refinement_mode:'beam_lns',photo_mode:'subject_aware',background_mode:'safe_cover',role_mode:'shadow'}:{})},
        (id,p,job)=>{currentJob.current=id;if(revision===generationRevision.current&&sourceRevision===sourceRequest.current){setJobId(id);setProgress(p);setStage(job.stage??job.state);if(job.active_preview)setActivePreview(job.active_preview);}},abort.signal,effectiveIntent);
      if(revision!==generationRevision.current||sourceRevision!==sourceRequest.current){
        if(sourceRevision===sourceRequest.current)setError('Inputs changed while generating. The completed run is saved in history.');
        return;
      }
      setAppliedIntent(intentKey(effectiveIntent));
      setResp(result);
    } catch (e) {
      if(!abort.signal.aborted&&sourceRevision===sourceRequest.current)setError(e instanceof Error?e.message:String(e));
    } finally {
      if(generationAbort.current===abort){setBusy(false);currentJob.current='';}
    }
  };

  const updateCandidate=(targetIndex:number,candidate:Candidate,expectedRun?:string)=>setResp(current=>current&&current.run_id===expectedRun?{
    ...current,results:current.results.map((result,index)=>index===targetIndex?{...result,candidates:[candidate,...result.candidates.slice(1)]}:result)
  }:current);


  // id -> role / label lookups for overlays (block ids and element ids).
  const { roleOf, labelOf } = useMemo(() => {
    const a = resp?.analysis ?? analysis;
    const roles: Record<string, string> = { ...(a?.roles ?? {}) };
    const labels: Record<string, string> = {};
    const walk = (b: BlockInfo) => {
      roles[b.id] = b.role;
      labels[b.id] = b.label;
      b.children?.forEach(walk);
    };
    a?.blocks.forEach(walk);
    a?.stickers.forEach(walk);
    a?.accents.forEach(walk);
    return { roleOf: (id: string) => roles[id], labelOf: (id: string) => labels[id] };
  }, [resp, analysis]);

  const legend = useMemo(() => {
    const a = resp?.analysis;
    if (!a) return [];
    const seen = new Map<string, string>();
    [...a.blocks, ...a.stickers, ...a.accents].forEach((b) => seen.set(b.role, b.label));
    return [...seen.entries()];
  }, [resp]);

  return (
    <div className="flex h-full flex-col">
      {crelloOpen && <CrelloBrowser onClose={()=>setCrelloOpen(false)} onSelect={(d,item)=>{loadSource(d,"crello-"+item.id,item.title);setCrelloSource(item);setCrelloOpen(false);}}/>}
      <header className="flex items-center gap-4 border-b border-white/[0.06] px-5 py-3.5">
        <Logo />
        <button className="chip" disabled={busy} onClick={()=>setCrelloOpen(true)}>Browse Crello</button>
        <button className="chip" onClick={()=>setWorkspace(!workspace)}>{workspace ? "Design studio" : "Review workspace"}</button>
        <div className="ml-4 hidden text-sm text-zinc-500 md:block">
          One design in, every format out — <span className="text-zinc-300">same elements, same words, rearranged with intent.</span>
        </div>
        <div className="ml-auto flex items-center gap-2 text-[11px]">
          <StatusDot ok={!!health?.renderer} label="Official renderer" />
          <StatusDot ok={!!health?.text_metrics} label="Glyph metrics" />
        </div>
      </header>

      {workspace ? <Workspace onLoad={(r,d,request)=>{++sourceRequest.current;generationAbort.current?.abort();if(currentJob.current)void api.cancel(currentJob.current).catch(()=>{});currentJob.current='';setBusy(false);setActivePreview(null);setOpen(null);setCrelloSource(null);setSource(d);setSourceId(null);setSourceName("Saved design");setAnalysis(r.analysis);setSourceRender(null);setLocks(request?.locks??{});setAnalysisOverrides(request?.analysis_overrides??{roles:{},group_modes:{}});setAppliedIntent(intentKey(request?.analysis_overrides??{roles:{},group_modes:{}}));setEmphasis(request?.emphasis??'balanced');setAllCandidates(request?.all_candidates??false);setTargets((request?.targets??r.results.map(result=>result.target)).map((target,index)=>({...target,id:target.id??`saved-${index}`,label:target.label??'Saved target'})));const options=request?.options??{};const mode=options.controller_mode??r.results[0]?.candidates[0]?.controller?.mode??'legacy_batch';if(['legacy_batch','single','single_restart','small_beam'].includes(String(mode)))setControllerMode(mode as ControllerMode);setSolverMode(options.solver_mode==='joint'?'joint':'legacy');setUseSolver(typeof options.solver==='boolean'?options.solver:true);setUseRefine(typeof options.refine==='boolean'?options.refine:true);setUseRetrieval(typeof options.retrieval==='boolean'?options.retrieval:true);setLearnedLayouts(options.learned_retarget!==false);setSemanticLayout(options.group_policies==='hierarchical');setResp(r);setWorkspace(false);}}/> : <>
      <div className="border-b border-white/10 px-5 py-3 flex flex-wrap items-center gap-4 text-xs">
        <label title={health?.learned_layouts===false?'Model weights not installed (engine/models/retarget.npz)':'Adds layouts from a model trained on how designers resized 1,400 Crello templates'}><input type="checkbox" disabled={health?.learned_layouts===false} checked={learnedLayouts&&health?.learned_layouts!==false} onChange={e=>setLearnedLayouts(e.target.checked)}/> Designer-trained layouts</label>
        <label title="Keep related copy, panels and illustrations together; use measured alternatives and subject-aware edits. Experimental."><input type="checkbox" checked={semanticLayout} onChange={e=>setSemanticLayout(e.target.checked)}/> Semantic layout (experimental)</label>
        <select aria-label="Element to keep fixed" className="bg-zinc-900 p-2" value={lockId} onChange={e=>setLockId(e.target.value)}><option value="">Keep an element fixed…</option>{source?.elements.map(e=><option key={e.id} value={e.id}>{e.content?.slice(0,30) || e.id}</option>)}</select>
        <button className="chip" disabled={!lockId} onClick={()=>{const e=(resp?.results[0]?.candidates[0]?.design ?? source)?.elements.find(e=>e.id===lockId);if(e)setLocks({...locks,[lockId]:{x:e.x,y:e.y,width:e.width,height:e.height}});}}>Lock current placement</button>
        {Object.keys(locks).length>0&&<button className="chip" onClick={()=>setLocks({})}>Clear {Object.keys(locks).length} {Object.keys(locks).length===1?'lock':'locks'}</button>}
        <details className="relative">
          <summary className="chip cursor-pointer select-none list-none">Advanced</summary>
          <div className="absolute right-0 top-full z-20 mt-2 flex w-80 flex-col gap-3 rounded-lg border border-white/10 bg-zinc-950 p-4 shadow-xl">
            <label className="flex flex-col gap-1">Workflow <select aria-label="Design workflow" className="bg-zinc-900 p-2" value={controllerMode} onChange={e=>setControllerMode(e.target.value as ControllerMode)}><option value="legacy_batch">Compare generated layouts</option><option value="single">Improve one design</option><option value="single_restart">Improve with composition recovery</option><option value="small_beam">Explore a small shortlist</option></select></label>
            <label className="flex flex-col gap-1">Solver <select aria-label="Geometry solver" className="bg-zinc-900 p-2" value={solverMode} onChange={e=>setSolverMode(e.target.value)}><option value="legacy">Placement</option><option value="joint">Joint text sizing and placement</option></select></label>
            <label><input type="checkbox" checked={allCandidates} onChange={e=>setAllCandidates(e.target.checked)}/> Inspect all options</label>
            <label><input type="checkbox" checked={useSolver} onChange={e=>setUseSolver(e.target.checked)}/> Optimize placement</label>
            <label title="Evaluate-and-fix loop: improves the best three candidates against the design rubric"><input type="checkbox" checked={useRefine} onChange={e=>setUseRefine(e.target.checked)}/> Refine best 3</label>
            <label><input type="checkbox" checked={useRetrieval} onChange={e=>setUseRetrieval(e.target.checked)}/> Saved arrangements</label>
          </div>
        </details>
        {busy && <span role="status">{stage.replaceAll('_',' ')} · {Math.round(progress*100)}% of sizes completed {jobId && <button className="chip" onClick={()=>api.cancel(jobId).catch(e=>setError(String(e)))}>Cancel generation</button>}</span>}
      </div>
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div className="h-[38vh] min-h-[180px] shrink-0 border-b border-white/[0.06] pt-4 lg:h-auto lg:min-h-0 lg:border-b-0 lg:border-r">
          <Sidebar
            onBrowseCrello={()=>setCrelloOpen(true)}
            crelloReference={crelloSource?.reference}
            sourceName={sourceName}
            samples={samples}
            presets={presets}
            sourceId={sourceId}
            source={source}
            sourceRender={sourceRender}
            sourceWarnings={sourceWarnings}
            sourcePreviewVerified={sourcePreviewVerified}
            analysis={analysis}
            analysisOverrides={analysisOverrides}
            onAnalysisOverrides={setAnalysisOverrides}
            semanticLayout={semanticLayout}
            intentPending={intentKey({roles:analysisOverrides.roles,group_modes:semanticLayout?analysisOverrides.group_modes:{}})!==appliedIntent}

            onPickSample={(s) => loadSource(s.source, s.id, s.id.replace("chameleon_sample_", ""), s.target_canvas)}
            onUpload={(d, name, task) => loadSource(d, null, name, task)}
            targets={targets}
            setTargets={setTargets}
            emphasis={emphasis}
            setEmphasis={setEmphasis}
            onGenerate={generate}
            busy={busy}
            error={error}
          />
        </div>

        <main className="min-h-0 flex-1 overflow-y-auto">
          {busy&&activePreview&&<div className="panel m-6 space-y-3 p-5"><p role="status" className="text-sm text-emerald-300">Improving size {activePreview.target_index+1} · {stage.replaceAll('_',' ')} · {activePreview.operation}</p><Preview src={activePreview.render} width={activePreview.width} height={activePreview.height} maxHeight={440} className="w-full"/><p className="text-xs text-zinc-500">Latest accepted, rendered checkpoint. Rejected edits cannot replace it.</p></div>}
          {!resp && !busy && <EmptyState source={source} targets={targets} />}
          {busy && !resp && !activePreview && <LoadingState targets={targets} />}
          {resp && (
            <div className={`space-y-10 px-5 py-6 sm:px-8 ${busy ? "opacity-50" : ""}`}>
              <div className="flex flex-wrap items-center gap-3">
                <div>
                  <h1 className="font-display text-3xl text-white">Layouts</h1>
                  <p className="text-xs text-zinc-500">
                    {resp.results.length} size{resp.results.length > 1 ? "s" : ""} · {resp.results.reduce((n, r) => n + r.candidates.length, 0)} options ·
                    solved and rendered in {(resp.elapsed_ms / 1000).toFixed(1)}s
                  </p>
                </div>
                <div className="ml-auto flex items-center gap-1">
                  <Toggle on={overlay} onChange={setOverlay} label="Group overlay" />
                  <Toggle on={baselines} onChange={setBaselines} label="Show naive stretch" />
                </div>
              </div>
              {overlay && (
                <div className="-mt-6 flex flex-wrap gap-1.5">
                  {legend.map(([role, label]) => (
                    <span key={role} className="chip !text-[11px]">
                      <span className="h-2 w-2 rounded-sm" style={{ background: roleColor(role) }} /> {label}
                    </span>
                  ))}
                </div>
              )}
              {resp.results.map((r, ti) => {
                const { width, height } = r.target;
                const wide = width / height >= 2.2;
                return (
                  <section key={`${width}x${height}-${ti}`}>
                    <div className="mb-3 flex items-baseline gap-3">
                      <h2 className="text-base font-semibold text-white">{r.target.label ?? "Custom"}</h2>
                      <span className="font-mono text-xs text-zinc-500">
                        {width}×{height} · {ratioLabel(width, height)}
                      </span>
                      <span className="ml-auto text-[11px] text-zinc-600">{r.elapsed_ms} ms</span>
                    </div>
                    {baselines && (
                      <div className="mb-3 flex items-center gap-4 rounded-xl border border-amber-300/10 bg-amber-300/[0.03] p-3">
                        <div className="w-40 shrink-0 text-xs text-amber-200/80">
                          <div className="font-semibold">Naive stretch</div>
                          <div className="mt-0.5 text-amber-200/50">What the starter baseline produces for this size.</div>
                        </div>
                        <Preview key={r.baseline.render ?? ""} src={r.baseline.render} width={width} height={height} maxHeight={wide ? 90 : 160} className="w-full" />
                      </div>
                    )}
                    {r.status==='unverified_missing_asset'&&<p role="status" className="mb-3 text-xs leading-relaxed text-amber-300">These layouts are visible, but they are not verified. A required original image is missing, so the renderer drew a placeholder.</p>}
                    {r.candidates[0]?.controller ? <ActiveDesign key={`${resp.run_id}-${ti}`} candidate={r.candidates[0]} runId={resp.run_id} targetIndex={ti} locked={Object.keys(locks)} onCandidate={candidate=>updateCandidate(ti,candidate,resp.run_id)} onLock={element=>setLocks({...locks,[element.id]:{x:element.x,y:element.y,width:element.width,height:element.height}})}/> : <div className={`grid gap-4 ${wide ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2 2xl:grid-cols-3"}`}>
                      {r.candidates.map((c, ci) => (
                        <CandidateCard
                          key={c.template}
                          c={c}
                          rank={ci}
                          name={sourceName}
                          wide={wide}
                          showOverlay={overlay}
                          roleOf={roleOf}
                          labelOf={labelOf}
                          onOpen={() => setOpen({ t: ti, c: ci })}
                        />
                      ))}
                    </div>}
                    {!r.candidates.length&&<p role="alert" className="panel p-4 text-amber-200">No verified design met the requirements for this size. Review the saved run or relax a constraint.</p>}
                  </section>
                );
              })}
              {resp.analysis.notes.length > 0 && (
                <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 text-xs text-zinc-400">
                  <div className="label mb-1.5">Engine notes</div>
                  {resp.analysis.notes.map((n) => (
                    <div key={n} className="flex gap-2">
                      <Icon name="info" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-300" /> {n}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {open && resp && source && (
        <DetailModal
          result={resp.results[open.t]}
          index={open.c}
          source={source}
          name={sourceName}
          roleOf={roleOf}
          labelOf={labelOf}
          onClose={() => setOpen(null)}
          onIndex={(c) => setOpen({ ...open, c })}
        />
      )}
      </>}
    </div>
  );
}

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="chip !py-0.5" title={ok ? `${label} available` : `${label} unavailable`}>
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,.8)]" : "bg-rose-400"}`} />
      {label}
    </span>
  );
}

function EmptyState({ source, targets }: { source: Design | null; targets: Target[] }) {
  const steps = [
    { icon: "layers", title: "Understand", body: "Infers roles and groups — headline, product, CTA, badge rows, stickers — from geometry and type, never from IDs." },
    { icon: "compare", title: "Propose", body: "Builds structurally different arrangements: stacked, split, banner rows, anchored, proportional." },
    { icon: "spark", title: "Fit & score", body: "Measures every line with the renderer's own glyph metrics, then scores readability, hierarchy and composition." },
  ];
  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col justify-center px-8 py-16">
      <div className="font-display text-5xl leading-[1.05] text-white">
        A naive program stretches the picture.
        <br />
        <span className="bg-gradient-to-r from-emerald-300 via-teal-200 to-violet-300 bg-clip-text italic text-transparent">A good one moves the furniture.</span>
      </div>
      <p className="mt-5 max-w-xl text-sm leading-relaxed text-zinc-400">
        {source ? (
          <>
            Source loaded ({source.elements.length} elements). Choose {targets.length ? "more " : ""}output sizes on the left and hit{" "}
            <span className="text-emerald-300">Generate</span>.
          </>
        ) : (
          "Pick a sample design or drop your own JSON to begin."
        )}
      </p>
      <div className="mt-10 grid gap-3 sm:grid-cols-3">
        {steps.map((s, i) => (
          <div key={s.title} className="panel p-4 animate-rise" style={{ animationDelay: `${i * 90}ms` }}>
            <div className="mb-3 grid h-8 w-8 place-items-center rounded-lg bg-white/5 text-emerald-300">
              <Icon name={s.icon} />
            </div>
            <div className="text-sm font-semibold text-white">{s.title}</div>
            <div className="mt-1 text-xs leading-relaxed text-zinc-500">{s.body}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LoadingState({ targets }: { targets: Target[] }) {
  return (
    <div className="space-y-10 px-8 py-6">
      <div className="h-9 w-40 rounded-lg skeleton" />
      {targets.map((t) => {
        const wide = t.width / t.height >= 2.2;
        return (
          <div key={t.id}>
            <div className="mb-3 text-sm text-zinc-500">
              {t.label} <span className="font-mono text-xs">{t.width}×{t.height}</span>
            </div>
            <div className={`grid gap-4 ${wide ? "grid-cols-1" : "md:grid-cols-2 2xl:grid-cols-3"}`}>
              {[0, 1, 2].map((i) => (
                <div key={i} className="panel overflow-hidden">
                  <div className="flex items-center justify-center bg-black/20 p-5">
                    <div className="w-full rounded-lg skeleton" style={{ aspectRatio: `${t.width}/${t.height}`, maxHeight: wide ? 200 : 300, maxWidth: wide ? undefined : (300 * t.width) / t.height }} />
                  </div>
                  <div className="space-y-2 p-4">
                    <div className="h-4 w-1/2 rounded skeleton" />
                    <div className="h-3 w-3/4 rounded skeleton" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function intentKey(value:AnalysisOverrides){
  return JSON.stringify({roles:Object.entries(value.roles).sort(([a],[b])=>a.localeCompare(b)),
    group_modes:Object.entries(value.group_modes).sort(([a],[b])=>a.localeCompare(b))});
}
