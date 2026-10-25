import type { Analysis, AnalysisOverrides, Candidate, Design, JobUpdate, JudgeResult, Preset, Sample, Target } from "./types";

async function req<T>(path: string, body?: unknown, signal?:AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<{ ok: boolean; renderer: boolean; text_metrics: boolean }>("/api/health"),
  samples: () => req<Sample[]>("/api/samples"),
  presets: () => req<Preset[]>("/api/presets"),
  analyze: (design: Design, groupPolicies:'legacy'|'hierarchical'='legacy', signal?:AbortSignal) =>
    req<{ analysis: Analysis; render: string | null; warnings: string[] }>("/api/analyze", { design, group_policies:groupPolicies },signal),
  generate: async (source: Design, targets: Target[], emphasis: string, locks: Record<string,Record<string,number>> = {}, allCandidates = false, options: Record<string,boolean|string|number> = {}, onProgress?: (id:string, progress:number,job:JobUpdate)=>void, signal?:AbortSignal, analysisOverrides:AnalysisOverrides={roles:{},group_modes:{}}) => {
    const started=await req<{id:string}>("/api/jobs",{ source, targets, emphasis, locks, top_k:allCandidates?36:3, all_candidates:allCandidates, options, analysis_overrides:analysisOverrides, idempotency_key:crypto.randomUUID() },signal);
    let after=-1;
    for (;;) {
      const job=await req<JobUpdate>(`/api/jobs/${started.id}?after_event=${after}`,undefined,signal);
      after=Math.max(after,...job.events.map(event=>event.event_id??-1));
      onProgress?.(started.id,job.progress,job);
      if(job.state==='done' && job.result)return job.result;
      if(job.state==='cancelled' && job.result?.results.length)return job.result;
      if(job.state==='cancelled')throw new Error('Generation cancelled. Saved revisions remain in history.');
      if(job.state==='failed' || job.state==='interrupted')throw new Error(job.error??'No verified design was produced. Inspect this run in history.');
      await new Promise<void>((resolve,reject)=>{
        const finish=()=>{signal?.removeEventListener('abort',abort);resolve();};
        const timer=setTimeout(finish,350);
        const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};
        signal?.addEventListener('abort',abort,{once:true});
        if(signal?.aborted)abort();
      });
    }
  },
  feedback:(runId:string,targetIndex:number,revisionHash:string,verdict:'like'|'dislike',reason?:string,resourceHash?:string)=>req(`/api/history/${runId}/feedback`,{target_index:targetIndex,revision_hash:revisionHash,resource_hash:resourceHash,verdict,reason,provenance:'human'}),
  edit:(runId:string,targetIndex:number,expected:string,patches:Record<string,Record<string,number>>)=>req<{candidate:Candidate;revision_hash:string}>(`/api/history/${runId}/targets/${targetIndex}/edit`,{expected_revision_hash:expected,patches}),
  restore:(runId:string,targetIndex:number,expected:string,revision:string)=>req<{candidate:Candidate;revision_hash:string}>(`/api/history/${runId}/targets/${targetIndex}/restore`,{expected_revision_hash:expected,revision_hash:revision}),
  cancel: (id:string)=>req(`/api/jobs/${id}/cancel`,{}),
  judgeStatus: () => req<{ available: boolean; model: string; detail?: string }>("/api/judge/status"),
  judge: (source: Design, design: Design) => req<JudgeResult>("/api/judge", { source, design }),
};

export function download(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ratioLabel(w: number, h: number) {
  const r = w / h;
  if (Math.abs(r - 1) < 0.01) return "1:1";
  return r >= 1 ? `${r.toFixed(r >= 10 ? 0 : 2).replace(/\.?0+$/, "")}:1` : `1:${(1 / r).toFixed(2).replace(/\.?0+$/, "")}`;
}
